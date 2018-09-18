from __future__ import division
from builtins import zip
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import nn
import util
from learner import CoreModel


class ValueRL(CoreModel):
  """
  Learn a state-action value function and its corresponding policy.
  """

  @property
  def saveid(self):
    return "valuerl"

  def create_params(self, env_config, learner_config):
    self.obs_dim = np.prod(env_config["obs_dims"])
    self.action_dim = env_config["action_dim"]
    self.reward_scale = env_config["reward_scale"]
    self.discount = env_config["discount"]

    self.hidden_dim = learner_config["hidden_dim"]
    self.bayesian_config = learner_config["bayesian"]
    self.value_expansion = learner_config["value_expansion"]
    self.explore_chance = learner_config["ddpg_explore_chance"]

    with tf.variable_scope(self.name):
      self.policy = nn.FeedForwardNet('policy', self.obs_dim, [self.action_dim], layers=4, hidden_dim=self.hidden_dim, get_uncertainty=False)

      if self.bayesian_config:
        self.Q = nn.EnsembleFeedForwardNet('Q', self.obs_dim + self.action_dim, [], layers=4, hidden_dim=self.hidden_dim, get_uncertainty=True, ensemble_size=self.bayesian_config["ensemble_size"], train_sample_count=self.bayesian_config["train_sample_count"], eval_sample_count=self.bayesian_config["eval_sample_count"])
        self.old_Q = nn.EnsembleFeedForwardNet('old_q', self.obs_dim + self.action_dim, [], layers=4, hidden_dim=self.hidden_dim, get_uncertainty=True, ensemble_size=self.bayesian_config["ensemble_size"], train_sample_count=self.bayesian_config["train_sample_count"], eval_sample_count=self.bayesian_config["eval_sample_count"])
      else:
        self.Q = nn.FeedForwardNet('Q', self.obs_dim + self.action_dim, [], layers=4, hidden_dim=self.hidden_dim, get_uncertainty=True)
        self.old_Q = nn.FeedForwardNet('old_q', self.obs_dim + self.action_dim, [], layers=4, hidden_dim=self.hidden_dim, get_uncertainty=True)

    self.policy_params = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name) if "policy" in v.name]
    self.Q_params = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name) if "Q" in v.name]
    self.agent_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    self.copy_to_old_ops = [tf.assign(p_old, p) for p_old, p in zip(self.old_Q.params_list, self.Q.params_list)]
    self.assign_epoch_op = [tf.assign(self.epoch_n, self.epoch_n_placeholder), tf.assign(self.update_n, self.update_n_placeholder), tf.assign(self.frame_n, self.frame_n_placeholder), tf.assign(self.hours, self.hours_placeholder)]

  def update_epoch(self, sess, epoch, updates, frames, hours):
    sess.run(self.assign_epoch_op, feed_dict={self.epoch_n_placeholder: int(epoch), self.update_n_placeholder: int(updates), self.frame_n_placeholder: int(frames), self.hours_placeholder: float(hours)})

  def copy_to_old(self, sess):
    sess.run(self.copy_to_old_ops)

  def build_evalution_graph(self, obs, get_full_info=False, mode="regular", n_samples=1):
    assert mode in {"regular", "explore", "exploit"}
    policy_actions_pretanh = self.policy(obs)

    if mode == "regular" or mode == "exploit":
      policy_actions = tf.tanh(policy_actions_pretanh)
    elif mode == "explore":
      _, _, exploring_policy_actions, _ = util.tanh_sample_info(policy_actions_pretanh, tf.zeros_like(policy_actions_pretanh), n_samples=n_samples)
      policy_actions = tf.where(tf.random_uniform(tf.shape(exploring_policy_actions)) < self.explore_chance, x=exploring_policy_actions, y=tf.tanh(policy_actions_pretanh))
    else: raise Exception('this should never happen')

    if get_full_info:     return policy_actions_pretanh, policy_actions
    else:                 return policy_actions

  def build_training_graph(self, obs, next_obs, empirical_actions, rewards, dones, data_size, worldmodel=None):
    average_model_use = tf.constant(0.)
    empirical_Q_info = tf.concat([obs, empirical_actions], 1)

    if worldmodel is None:
      policy_action_pretanh, policy_actions = self.build_evalution_graph(obs, get_full_info=True)
      policy_Q_info = tf.concat([obs, policy_actions], 1)
      state_value_estimate = self.Q(policy_Q_info, reduce_mode="mean")

      next_policy_actions = self.build_evalution_graph(next_obs)
      policy_next_Q_info = tf.concat([next_obs, next_policy_actions], 1)
      next_Q_estimate = self.old_Q(policy_next_Q_info, reduce_mode="mean")

      Q_guess = self.Q(empirical_Q_info, is_eval=False, reduce_mode="random")
      Q_target = rewards * self.reward_scale + self.discount * next_Q_estimate * (1. - dones)

      policy_losses = -state_value_estimate
      Q_losses = .5 * tf.square( Q_guess - tf.stop_gradient(Q_target) )

    else:
      targets, confidence, Q_guesses, reach_probs = self.build_Q_expansion_graph(next_obs, rewards, dones, worldmodel, rollout_len=self.value_expansion["rollout_len"], model_ensembling=worldmodel.bayesian_config is not False)

      # targets is a 3D matrix: [batch_i, start_timestep, end_timestep]. here, we reduce out the last dimension, turning
      # it into a [batch_i, start_timestep] matrix. in other words, we are taking a bunch of candidate targets and reducing
      # them into a single target. the four options here correspond to the four ways to do that reduction.
      if self.value_expansion["mean_k_return"]:
        target_counts = self.value_expansion["rollout_len"]+1 - tf.reshape(tf.range(self.value_expansion["rollout_len"]+1), [1, self.value_expansion["rollout_len"]+1])
        k_returns = tf.reduce_sum(targets, 2) / tf.cast(target_counts, tf.float32)
      elif self.value_expansion["lambda_return"]:
        cont_coeffs = self.value_expansion["lambda_return"] ** tf.cast(tf.reshape(tf.range(self.value_expansion["rollout_len"]+1), [1,1,self.value_expansion["rollout_len"]+1]), tf.float32)
        stop_coeffs = tf.concat([(1 - self.value_expansion["lambda_return"]) * tf.ones_like(targets)[:,:,:-1], tf.ones_like(targets)[:,:,-1:]], 2)
        k_returns = tf.reduce_sum(targets * stop_coeffs * cont_coeffs, 2)
      elif self.value_expansion["steve_reweight"]:
        k_returns = tf.reduce_sum(targets * confidence, 2)
        average_model_use = 1. - tf.reduce_mean(confidence[:,0,0])
      else:
        # MVE objective: just take the last one
        k_returns = targets[:,:,-1]

      # now we have [batch_i, start_timestep]. if we are using the TDK trick, then we want to use all of the targets,
      # so we construct a corresponding [batch_i, start_timestep] matrix of guesses. otherwise, we just take the targets
      # for the first timestep.
      Q_guess = self.Q(empirical_Q_info, is_eval=False, reduce_mode="random")
      if self.value_expansion["tdk_trick"]:
        Q_guess = tf.concat([tf.expand_dims(Q_guess, 1), Q_guesses], 1)
        reach_probs = tf.concat([tf.expand_dims(tf.ones_like(reach_probs[:,0]), 1), reach_probs[:,:-1]], 1)
        Q_target = k_returns
      else:
        # non-TDK trick means we just take the first one
        Q_target = k_returns[:,0]

      policy_action_pretanh, policy_actions = self.build_evalution_graph(obs, get_full_info=True)
      policy_Q_info = tf.concat([obs, policy_actions], 1)
      state_value_estimate = self.Q(policy_Q_info, stop_params_gradient=True, reduce_mode="mean")

      policy_losses = -state_value_estimate
      Q_losses = .5 * tf.square( Q_guess - tf.stop_gradient(Q_target) )
      if self.value_expansion["tdk_trick"]: Q_losses *= reach_probs # we downscale the various TDK-trick losses by
                                                                    # the likelihood of actually reaching the state
                                                                    # from which the guess was made
    policy_loss = tf.reduce_mean(policy_losses)
    Q_loss = tf.reduce_mean(Q_losses)
    policy_reg_loss = tf.reduce_mean(tf.square(policy_action_pretanh)) * .001 # a small regularization to make sure the
                                                                              # tanh does not saturate

    # anything in inspect gets logged
    inspect = (policy_loss, Q_loss, policy_reg_loss, average_model_use)

    return (policy_loss + policy_reg_loss, Q_loss), inspect


  def build_Q_expansion_graph(self, obs, first_rewards, first_done, worldmodel, rollout_len=1, model_ensembling=False):
    ### this sets up the machinery for having multiple parallel rollouts, each of which has a single consistent transition
    ensemble_idxs, transition_sample_n, reward_sample_n = worldmodel.get_ensemble_idx_info()
    q_sample_n = self.bayesian_config["eval_sample_count"] if self.bayesian_config is not False else 1
    first_rewards = tf.tile(tf.expand_dims(tf.expand_dims(first_rewards,1),1), [1,transition_sample_n,reward_sample_n])
    first_rewards.set_shape([None, transition_sample_n, reward_sample_n])
    if model_ensembling:
      obs = tf.tile(tf.expand_dims(obs,1), [1,transition_sample_n,1])
      obs.set_shape([None, transition_sample_n, self.obs_dim])
      first_done = tf.tile(tf.expand_dims(first_done, 1), [1, transition_sample_n])
      first_done.set_shape([None, transition_sample_n])

    ### below, we use a while loop to actually do the iterative model rollout
    extra_info = worldmodel.init_extra_info(obs)

    action_ta = tf.TensorArray(size=rollout_len, dynamic_size=False, dtype=tf.float32)
    obs_ta =       tf.TensorArray(size=rollout_len, dynamic_size=False, dtype=tf.float32)
    done_ta =     tf.TensorArray(size=rollout_len, dynamic_size=False, dtype=tf.float32)
    extra_info_ta =tf.TensorArray(size=rollout_len, dynamic_size=False, dtype=tf.float32)

    def rollout_loop_body(r_i, xxx_todo_changeme):
      (obs, done, extra_info, action_ta, obs_ta, dones_ta, extra_info_ta) = xxx_todo_changeme
      action_pretanh, action = self.build_evalution_graph(tf.stop_gradient(obs), get_full_info=True)

      if model_ensembling:
        next_obs, next_dones, next_extra_info = worldmodel.transition(obs, action, extra_info, ensemble_idxs=ensemble_idxs)
      else:
        next_obs, next_dones, next_extra_info = worldmodel.transition(obs, action, extra_info)
        next_obs = tf.reduce_mean(next_obs, -2)
        next_dones = tf.reduce_mean(next_dones, -1)

      action_ta = action_ta.write(r_i, action)
      obs_ta = obs_ta.write(r_i, obs)
      dones_ta = dones_ta.write(r_i, done)
      extra_info_ta = extra_info_ta.write(r_i, extra_info)
      return r_i+1, (next_obs, next_dones, next_extra_info, action_ta, obs_ta, dones_ta, extra_info_ta)

    _, (final_obs, final_done, final_extra_info, action_ta, obs_ta, done_ta, extra_info_ta) = tf.while_loop(
        lambda r_i, _: r_i < rollout_len,
        rollout_loop_body,
        [0, (obs, first_done, extra_info, action_ta, obs_ta, done_ta, extra_info_ta)]
    )

    final_action_pretanh, final_action = self.build_evalution_graph(tf.stop_gradient(final_obs), get_full_info=True)

    ### compile the TensorArrays into useful tensors
    obss = obs_ta.stack()
    obss = tf.reshape(obss, tf.stack([rollout_len, -1, transition_sample_n, self.obs_dim]))
    obss = tf.transpose(obss, [1, 0, 2, 3])
    final_obs = tf.reshape(final_obs, tf.stack([-1, 1, transition_sample_n, self.obs_dim]))
    all_obss = tf.concat([obss, final_obs],1)
    next_obss = all_obss[:,1:]

    dones = done_ta.stack()
    dones = tf.reshape(dones, tf.stack([rollout_len, -1, transition_sample_n]))
    dones = tf.transpose(dones, [1, 0, 2])
    final_done = tf.reshape(final_done, tf.stack([-1, 1, transition_sample_n]))
    all_dones = tf.concat([dones, final_done],1)

    actions = action_ta.stack()
    actions = tf.reshape(actions, tf.stack([rollout_len, -1, transition_sample_n, self.action_dim]))
    actions = tf.transpose(actions , [1, 0, 2, 3])
    final_action = tf.reshape(final_action, tf.stack([-1, 1, transition_sample_n, self.action_dim]))
    all_actions = tf.concat([actions, final_action],1)

    continue_probs = tf.cumprod(1. - all_dones, axis=1)
    rewards = worldmodel.get_rewards(obss, actions, next_obss)
    rawrew = rewards = tf.concat([tf.expand_dims(first_rewards, 1), rewards],1)

    ### TDK trick means we have to guess at every timestep
    if self.value_expansion["tdk_trick"]:
      guess_info = tf.concat([obss,actions], -1)
      Q_guesses = self.Q(guess_info, reduce_mode="random")
      Q_guesses = tf.reduce_mean(Q_guesses, -1) # make it so there's only one guess per rollout length, which is the mean of the guesses under all the various model rollouts
      reached_this_point_to_guess_prob = tf.reduce_mean(continue_probs, -1)
    else:
      Q_guesses = None
      reached_this_point_to_guess_prob = None

    ### use the Q function at every timestep to get value estimates
    target_info = tf.concat([all_obss, all_actions], -1)
    Q_targets = self.old_Q(target_info, reduce_mode="none")

    rollout_frames = rollout_len + 1 # if we take N steps, we have N+1 frames

    ### create "decay-exponent matrix" of size [1,ROLLOUT_FRAMES,ROLLOUT_FRAMES,1]. the first ROLLOUT_FRAMES corresponds to the index of the source, the second to the target.
    ts_count_mat = (tf.cast(tf.reshape(tf.range(rollout_frames), [1, rollout_frames]) - tf.reshape(tf.range(rollout_frames), [rollout_frames, 1]), tf.float32))
    reward_coeff_matrix = tf.matrix_band_part(tf.ones([rollout_frames, rollout_frames]), 0, -1) * self.discount ** ts_count_mat
    value_coeff_matrix = tf.matrix_band_part(tf.ones([rollout_frames, rollout_frames]), 0, -1) * self.discount ** (1. + ts_count_mat)
    reward_coeff_matrix = tf.reshape(reward_coeff_matrix, [1, rollout_frames, rollout_frames, 1, 1])
    value_coeff_matrix = tf.reshape(value_coeff_matrix, [1, rollout_frames, rollout_frames, 1, 1])

    ### similarly, create a "done" matrix
    shifted_continue_probs = tf.concat([tf.expand_dims(tf.ones_like(continue_probs[:,0]),1), continue_probs[:,:-1]], 1)
    reward_continue_matrix = tf.expand_dims(shifted_continue_probs, 1) / tf.expand_dims(shifted_continue_probs+1e-8, 2)
    value_continue_matrix = tf.expand_dims(continue_probs, 1) / tf.expand_dims(shifted_continue_probs+1e-8, 2)
    reward_continue_matrix = tf.expand_dims(reward_continue_matrix, -1)
    value_continue_matrix = tf.expand_dims(value_continue_matrix, -1)

    ### apply the discounting factors to the rewards and values
    rewards = tf.expand_dims(rewards, 1) * reward_coeff_matrix * reward_continue_matrix
    rewards = tf.cumsum(rewards, axis=2)
    values = tf.expand_dims(Q_targets, 1) * value_coeff_matrix * value_continue_matrix

    ### compute the targets using the Bellman equation
    sampled_targets = tf.expand_dims(rewards,-2) * self.reward_scale + tf.expand_dims(values,-1)

    ### flatten out the various sources of variance (transition, reward, and Q-function ensembles) to get a set of estimates for each candidate target
    sampled_targets = tf.reshape(sampled_targets, tf.stack([-1, rollout_frames, rollout_frames, transition_sample_n * reward_sample_n * q_sample_n]))

    ### compute the mean and variance for each candidate target
    target_means, target_variances = tf.nn.moments(sampled_targets, 3)

    ### compute the confidence, either using the full covariance matrix, or approximating all the estimators as independent
    if self.value_expansion["covariances"]:
      targetdiffs = sampled_targets - tf.expand_dims(target_means,3)
      target_covariances = tf.einsum("abij,abjk->abik", targetdiffs, tf.transpose(targetdiffs, [0,1,3,2]))
      target_confidence = tf.squeeze(tf.matrix_solve(target_covariances + tf.expand_dims(tf.expand_dims(tf.matrix_band_part(tf.ones(tf.shape(target_covariances)[-2:]),0,0) * 1e-3,0),0), tf.ones(tf.concat([tf.shape(target_covariances)[:-1], tf.constant([1])],0))),-1)
    else:
      target_confidence = 1./(target_variances + 1e-8)

    ### normalize so weights sum to 1
    target_confidence *= tf.matrix_band_part(tf.ones([1, rollout_frames, rollout_frames]), 0, -1)
    target_confidence = target_confidence / tf.reduce_sum(target_confidence, axis=2, keepdims=True)

    ### below here is a bunch of debugging Print statements that I use as a sanity check:
    # target_confidence = tf.Print(target_confidence, [], message="raw rewards")
    # target_confidence = tf.Print(target_confidence, [rawrew[0,:,0,0]], summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [], message="\n", summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [(1. - all_dones)[0,:,0]], message="contin", summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [continue_probs[0,:,0]], message="cum_contin", summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [shifted_continue_probs[0,:,0]], message="shifted contin", summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [], message="reward_coeff")
    # for i in range(rollout_len+1): target_means = tf.Print(target_means, [reward_coeff_matrix[0,i,:,0,0]], summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [], message="reward_continue")
    # for i in range(rollout_len+1): target_means = tf.Print(target_means, [reward_continue_matrix[0,i,:,0,0]], summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [], message="value_coeff")
    # for i in range(rollout_len+1): target_means = tf.Print(target_means, [value_coeff_matrix[0,i,:,0,0]], summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [], message="value_continue")
    # for i in range(rollout_len+1): target_means = tf.Print(target_means, [value_continue_matrix[0,i,:,0,0]], summarize=rollout_len+1)
    # target_confidence = tf.Print(target_confidence, [], message="rewards")
    # for i in range(rollout_len+1): target_confidence = tf.Print(target_confidence, [rewards[0,i,:,0,0]], summarize=rollout_len+1)
    # target_confidence = tf.Print(target_confidence, [], message="target Qs")
    # target_confidence = tf.Print(target_confidence, [Q_targets[0,:,0,0]], summarize=rollout_len+1)
    # target_confidence = tf.Print(target_confidence, [], message="values")
    # for i in range(rollout_len+1): target_confidence = tf.Print(target_confidence, [values[0,i,:,0,0]], summarize=rollout_len+1)
    # target_confidence = tf.Print(target_confidence, [], message="target_means")
    # for i in range(rollout_len+1): target_confidence = tf.Print(target_confidence, [target_means[0,i,:]], summarize=rollout_len+1)
    # target_confidence = tf.Print(target_confidence, [], message="target_variance")
    # for i in range(rollout_len+1): target_confidence = tf.Print(target_confidence, [target_variances[0,i,:]], summarize=rollout_len+1)
    # target_confidence = tf.Print(target_confidence, [], message="target_confidence")
    # for i in range(rollout_len+1): target_confidence = tf.Print(target_confidence, [target_confidence[0,i,:]], summarize=rollout_len+1)
    # target_means = tf.Print(target_means, [target_confidence, action_lls, tf.shape(Q_targets)], message="\n\n", summarize=10)

    return target_means, target_confidence, Q_guesses, reached_this_point_to_guess_prob