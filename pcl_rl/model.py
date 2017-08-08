# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Model is responsible for setting up Tensorflow graph.

Creates policy and value networks.  Also sets up all optimization
ops, including gradient ops, trust region ops, and value optimizers.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Model(object):

  def __init__(self, env_spec, global_step,
               target_network_lag=0.95,
               sample_from='online',
               get_policy=None,
               get_baseline=None,
               get_objective=None,
               get_trust_region_p_opt=None,
               get_value_opt=None):
    self.env_spec = env_spec

    self.global_step = global_step
    self.inc_global_step = self.global_step.assign_add(1)

    self.target_network_lag = target_network_lag
    self.sample_from = sample_from

    self.policy = get_policy()
    self.baseline = get_baseline()
    self.objective = get_objective()
    self.baseline.eps_lambda = self.objective.eps_lambda  # TODO: do this better
    self.trust_region_policy_opt = get_trust_region_p_opt()
    self.value_opt = get_value_opt()

  def setup_placeholders(self):
    """Create the Tensorflow placeholders."""
    # summary placeholder
    self.avg_episode_reward = tf.placeholder(
        tf.float32, [], 'avg_episode_reward')

    # sampling placeholders
    self.internal_state = tf.placeholder(tf.float32,
                                         [None, self.policy.rnn_state_dim],
                                         'internal_state')

    self.single_observation = []
    for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
      if self.env_spec.is_discrete(obs_type):
        self.single_observation.append(
            tf.placeholder(tf.int32, [None], 'obs%d' % i))
      elif self.env_spec.is_box(obs_type):
        self.single_observation.append(
            tf.placeholder(tf.float32, [None, obs_dim], 'obs%d' % i))
      else:
        assert False

    self.single_action = []
    for i, (action_dim, action_type) in \
        enumerate(self.env_spec.act_dims_and_types):
      if self.env_spec.is_discrete(action_type):
        self.single_action.append(
            tf.placeholder(tf.int32, [None], 'act%d' % i))
      elif self.env_spec.is_box(action_type):
        self.single_action.append(
            tf.placeholder(tf.float32, [None, action_dim], 'act%d' % i))
      else:
        assert False

    # training placeholders
    self.observations = []
    for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
      if self.env_spec.is_discrete(obs_type):
        self.observations.append(
            tf.placeholder(tf.int32, [None, None], 'all_obs%d' % i))
      else:
        self.observations.append(
            tf.placeholder(tf.float32, [None, None, obs_dim], 'all_obs%d' % i))

    self.actions = []
    self.other_logits = []
    for i, (action_dim, action_type) in \
        enumerate(self.env_spec.act_dims_and_types):
      if self.env_spec.is_discrete(action_type):
        self.actions.append(
            tf.placeholder(tf.int32, [None, None], 'all_act%d' % i))
      if self.env_spec.is_box(action_type):
        self.actions.append(
            tf.placeholder(tf.float32, [None, None, action_dim],
                           'all_act%d' % i))
      self.other_logits.append(
          tf.placeholder(tf.float32, [None, None, None],
                         'other_logits%d' % i))

    self.rewards = tf.placeholder(tf.float32, [None, None], 'rewards')
    self.terminated = tf.placeholder(tf.float32, [None], 'terminated')
    self.pads = tf.placeholder(tf.float32, [None, None], 'pads')

    self.prev_log_probs = tf.placeholder(tf.float32, [None, None],
                                         'prev_log_probs')

  def setup(self):
    """Setup Tensorflow Graph."""

    self.setup_placeholders()

    tf.summary.scalar('avg_episode_reward', self.avg_episode_reward)

    with tf.variable_scope('model', reuse=None):
      # policy network
      with tf.variable_scope('policy_net'):
        (self.policy_internal_states, self.logits, self.log_probs,
         self.entropies, self.self_kls) = \
            self.policy.multi_step(self.observations,
                                   self.internal_state,
                                   self.actions)
        self.out_log_probs = sum(self.log_probs)
        self.kl = self.policy.calculate_kl(self.other_logits, self.logits)
        self.avg_kl = (tf.reduce_sum(sum(self.kl)[:-1] * (1 - self.pads)) /
                       tf.reduce_sum(1 - self.pads))

      # value network
      with tf.variable_scope('value_net'):
        (self.values,
         self.regression_input,
         self.regression_weight) = self.baseline.get_values(
            self.observations, self.actions,
            self.policy_internal_states, self.logits)

      # target policy network
      with tf.variable_scope('target_policy_net'):
        (self.target_policy_internal_states,
         self.target_logits, self.target_log_probs,
         _, _) = \
            self.policy.multi_step(self.observations,
                                   self.internal_state,
                                   self.actions)

      # target value network
      with tf.variable_scope('target_value_net'):
        (self.target_values, _, _) = self.baseline.get_values(
            self.observations, self.actions,
            self.target_policy_internal_states, self.target_logits)

      # construct copy op online --> target
      all_vars = tf.trainable_variables()
      online_vars = [p for p in all_vars if
                     '/policy_net' in p.name or '/value_net' in p.name]
      target_vars = [p for p in all_vars if
                     'target_policy_net' in p.name or 'target_value_net' in p.name]
      online_vars.sort(key=lambda p: p.name)
      target_vars.sort(key=lambda p: p.name)
      aa = self.target_network_lag
      self.copy_op = tf.group(*[
          target_p.assign(aa * target_p + (1 - aa) * online_p)
          for online_p, target_p in zip(online_vars, target_vars)])

      # evaluate objective
      (self.loss, self.raw_loss, self.regression_target,
       self.gradient_ops, self.summary) = self.objective.get(
          self.rewards, self.pads,
          self.values[:-1, :],
          self.values[-1, :] * (1 - self.terminated),
          self.log_probs, self.prev_log_probs, self.target_log_probs,
          self.entropies,
          self.logits)

      self.regression_target = tf.reshape(self.regression_target, [-1])

      self.policy_vars = [
          v for v in tf.trainable_variables()
          if '/policy_net' in v.name]
      self.value_vars = [
          v for v in tf.trainable_variables()
          if '/value_net' in v.name]

    # trust region optimizer
    if self.trust_region_policy_opt is not None:
      with tf.variable_scope('trust_region_policy', reuse=None):
        avg_self_kl = (
            tf.reduce_sum(sum(self.self_kls) * (1 - self.pads)) /
            tf.reduce_sum(1 - self.pads))

        self.trust_region_policy_opt.setup(
            self.policy_vars, self.raw_loss, avg_self_kl,
            self.avg_kl)

    # value optimizer
    if self.value_opt is not None:
      with tf.variable_scope('trust_region_value', reuse=None):
        self.value_opt.setup(
            self.value_vars,
            tf.reshape(self.values[:-1, :], [-1]),
            self.regression_target,
            tf.reshape(self.pads, [-1]),
            self.regression_input, self.regression_weight)

    # we re-use variables for the sampling operations
    with tf.variable_scope('model', reuse=True):
      scope = ('target_policy_net' if self.sample_from == 'target'
               else 'policy_net')
      with tf.variable_scope(scope):
        self.next_internal_state, self.sampled_actions = \
            self.policy.sample_step(self.single_observation,
                                    self.internal_state,
                                    self.single_action)
        self.greedy_next_internal_state, self.greedy_sampled_actions = \
            self.policy.sample_step(self.single_observation,
                                    self.internal_state,
                                    self.single_action,
                                    greedy=True)

  def sample_step(self, sess,
                  single_observation, internal_state, single_action,
                  greedy=False):
    """Sample batch of steps from policy."""
    if greedy:
      outputs = [self.greedy_next_internal_state, self.greedy_sampled_actions]
    else:
      outputs = [self.next_internal_state, self.sampled_actions]

    feed_dict = {self.internal_state: internal_state}
    for action_place, action in zip(self.single_action, single_action):
      feed_dict[action_place] = action
    for obs_place, obs in zip(self.single_observation, single_observation):
      feed_dict[obs_place] = obs

    return sess.run(outputs, feed_dict=feed_dict)

  def train_step(self, sess,
                 observations, internal_state, actions,
                 rewards, terminated, pads,
                 avg_episode_reward=0):
    """Train network using standard gradient descent."""
    outputs = [self.raw_loss, self.gradient_ops, self.summary]
    feed_dict = {self.internal_state: internal_state,
                 self.rewards: rewards,
                 self.terminated: terminated,
                 self.pads: pads,
                 self.avg_episode_reward: avg_episode_reward}
    for action_place, action in zip(self.actions, actions):
      feed_dict[action_place] = action
    for obs_place, obs in zip(self.observations, observations):
      feed_dict[obs_place] = obs

    return sess.run(outputs, feed_dict=feed_dict)


  def trust_region_step(self, sess,
                        observations, internal_state, actions,
                        rewards, terminated, pads,
                        avg_episode_reward=0):
    """Train policy using trust region step."""
    feed_dict = {self.internal_state: internal_state,
                 self.rewards: rewards,
                 self.terminated: terminated,
                 self.pads: pads,
                 self.avg_episode_reward: avg_episode_reward}
    for action_place, action in zip(self.actions, actions):
      feed_dict[action_place] = action
    for obs_place, obs in zip(self.observations, observations):
      feed_dict[obs_place] = obs

    (prev_log_probs, prev_logits) = sess.run(
        [self.out_log_probs, self.logits], feed_dict=feed_dict)
    feed_dict[self.prev_log_probs] = prev_log_probs
    for other_logit, prev_logit in zip(self.other_logits, prev_logits):
      feed_dict[other_logit] = prev_logit

    # fit policy
    self.trust_region_policy_opt.optimize(sess, feed_dict)

    ret = sess.run([self.raw_loss, self.summary], feed_dict=feed_dict)
    ret = [ret[0], None, ret[1]]
    return ret

  def fit_values(self, sess,
                 observations, internal_state, actions,
                 rewards, terminated, pads):
    """Train value network using value-specific optimizer."""
    feed_dict = {self.internal_state: internal_state,
                 self.rewards: rewards,
                 self.terminated: terminated,
                 self.pads: pads}
    for action_place, action in zip(self.actions, actions):
      feed_dict[action_place] = action
    for obs_place, obs in zip(self.observations, observations):
      feed_dict[obs_place] = obs

    # fit values
    if self.value_opt is None:
      raise ValueError('Specific value optimizer does not exist')
    self.value_opt.optimize(sess, feed_dict)
