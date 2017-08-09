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

"""Policy neural network.

Implements network which takes in input and produces actions
and log probabilities given a sampling distribution parameterization.
"""

import tensorflow as tf
import numpy as np


class Policy(object):
  def __init__(self, env_spec, internal_dim,
               fixed_std=True, recurrent=True,
               input_prev_actions=True):
    self.env_spec = env_spec
    self.internal_dim = internal_dim
    self.rnn_state_dim = self.internal_dim
    self.fixed_std = fixed_std
    self.recurrent = recurrent
    self.input_prev_actions = input_prev_actions

    self.matrix_init = tf.truncated_normal_initializer(stddev=0.01)
    self.vector_init = tf.constant_initializer(0.0)

  @property
  def input_dim(self):
    return (self.env_spec.total_obs_dim +
            self.env_spec.total_sampled_act_dim * self.input_prev_actions)

  @property
  def output_dim(self):
    return self.env_spec.total_sampling_act_dim

  def get_cell(self):
    """Get RNN cell."""
    self.cell_input_dim = self.internal_dim // 2
    cell = tf.contrib.rnn.LSTMCell(self.cell_input_dim,
                                   state_is_tuple=False,
                                   reuse=tf.get_variable_scope().reuse)

    cell = tf.contrib.rnn.OutputProjectionWrapper(
        cell, self.output_dim,
        reuse=tf.get_variable_scope().reuse)

    return cell

  def core(self, obs, prev_internal_state, prev_actions):
    """Core neural network taking in inputs and outputting sampling
    distribution parameters."""
    batch_size = tf.shape(obs[0])[0]
    if not self.recurrent:
      prev_internal_state = tf.zeros([batch_size, self.rnn_state_dim])

    cell = self.get_cell()

    b = tf.get_variable('input_bias', [self.cell_input_dim],
                        initializer=self.vector_init)
    cell_input = tf.nn.bias_add(tf.zeros([batch_size, self.cell_input_dim]), b)

    for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
      w = tf.get_variable('w_state%d' % i, [obs_dim, self.cell_input_dim],
                          initializer=self.matrix_init)
      if self.env_spec.is_discrete(obs_type):
        cell_input += tf.matmul(tf.one_hot(obs[i], obs_dim), w)
      elif self.env_spec.is_box(obs_type):
        cell_input += tf.matmul(obs[i], w)
      else:
        assert False

    if self.input_prev_actions:
      if self.env_spec.combine_actions:  # TODO(ofir): clean this up
        prev_action = prev_actions[0]
        for i, action_dim in enumerate(self.env_spec.orig_act_dims):
          act = tf.mod(prev_action, action_dim)
          w = tf.get_variable('w_prev_action%d' % i, [action_dim, self.cell_input_dim],
                              initializer=self.matrix_init)
          cell_input += tf.matmul(tf.one_hot(act, action_dim), w)
          prev_action = tf.to_int32(prev_action / action_dim)
      else:
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
          w = tf.get_variable('w_prev_action%d' % i, [act_dim, self.cell_input_dim],
                              initializer=self.matrix_init)
          if self.env_spec.is_discrete(act_type):
            cell_input += tf.matmul(tf.one_hot(prev_actions[i], act_dim), w)
          elif self.env_spec.is_box(act_type):
            cell_input += tf.matmul(prev_actions[i], w)
          else:
            assert False

    output, next_state = cell(cell_input, prev_internal_state)

    return output, next_state

  def sample_action(self, logits, sampling_dim,
                    act_dim, act_type, greedy=False):
    """Sample an action from a distribution."""
    if self.env_spec.is_discrete(act_type):
      if greedy:
        act = tf.argmax(logits, 1)
      else:
        act = tf.reshape(tf.multinomial(logits, 1), [-1])
    elif self.env_spec.is_box(act_type):
      means = logits[:, :sampling_dim / 2]
      std = logits[:, sampling_dim / 2:]
      if greedy:
        act = means
      else:
        batch_size = tf.shape(logits)[0]
        act = means + std * tf.random_normal([batch_size, act_dim])
    else:
      assert False

    return act

  def entropy(self, logits,
              sampling_dim, act_dim, act_type):
    """Calculate entropy of distribution."""
    if self.env_spec.is_discrete(act_type):
      entropy = tf.reduce_sum(
          -tf.nn.softmax(logits) * tf.nn.log_softmax(logits), -1)
    elif self.env_spec.is_box(act_type):
      means = logits[:, :sampling_dim / 2]
      std = logits[:, sampling_dim / 2:]
      entropy = tf.reduce_sum(
          0.5 * (1 + tf.log(2 * np.pi * tf.square(std))), -1)
    else:
      assert False

    return entropy

  def self_kl(self, logits,
              sampling_dim, act_dim, act_type):
    """Calculate KL of distribution with itself.

    Used layer only for the gradients.
    """

    if self.env_spec.is_discrete(act_type):
      probs = tf.nn.softmax(logits)
      log_probs = tf.nn.log_softmax(logits)
      self_kl = tf.reduce_sum(
          tf.stop_gradient(probs) *
          (tf.stop_gradient(log_probs) - log_probs), -1)
    elif self.env_spec.is_box(act_type):
      means = logits[:, :sampling_dim / 2]
      std = logits[:, sampling_dim / 2:]
      my_means = tf.stop_gradient(means)
      my_std = tf.stop_gradient(std)
      self_kl = tf.reduce_sum(
          tf.log(std / my_std) +
          (tf.square(my_std) + tf.square(my_means - means)) /
          (2.0 * tf.square(std)) - 0.5,
          -1)
    else:
      assert False

    return self_kl

  def log_prob_action(self, action, logits,
                      sampling_dim, act_dim, act_type):
    """Calculate log-prob of action sampled from distribution."""
    if self.env_spec.is_discrete(act_type):
      act_log_prob = tf.reduce_sum(
          tf.one_hot(action, act_dim) * tf.nn.log_softmax(logits), -1)
    elif self.env_spec.is_box(act_type):
      means = logits[:, :sampling_dim / 2]
      std = logits[:, sampling_dim / 2:]
      act_log_prob = (- 0.5 * tf.log(2 * np.pi * tf.square(std))
                      - 0.5 * tf.square(action - means) / tf.square(std))
      act_log_prob = tf.reduce_sum(act_log_prob, -1)
    else:
      assert False

    return act_log_prob

  def sample_actions(self, output, actions=None, greedy=False):
    """Sample all actions given output of core network."""
    sampled_actions = []
    logits = []
    log_probs = []
    entropy = []
    self_kl = []

    start_idx = 0
    for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
      sampling_dim = self.env_spec.sampling_dim(act_dim, act_type)
      if self.fixed_std and self.env_spec.is_box(act_type):
        act_logits = output[:, start_idx:start_idx + act_dim]

        log_std = tf.get_variable('std%d' % i, [1, sampling_dim // 2])
        # fix standard deviations to variable
        act_logits = tf.concat(
            [act_logits,
             1e-6 + tf.exp(log_std) + 0 * act_logits], 1)
      else:
        act_logits = output[:, start_idx:start_idx + sampling_dim]

      if actions is None:
        act = self.sample_action(act_logits, sampling_dim,
                                 act_dim, act_type,
                                 greedy=greedy)
      else:
        act = actions[i]

      ent = self.entropy(act_logits, sampling_dim, act_dim, act_type)
      kl = self.self_kl(act_logits, sampling_dim, act_dim, act_type)

      act_log_prob = self.log_prob_action(
          act, act_logits,
          sampling_dim, act_dim, act_type)

      sampled_actions.append(act)
      logits.append(act_logits)
      log_probs.append(act_log_prob)
      entropy.append(ent)
      self_kl.append(kl)

      start_idx += sampling_dim

    assert start_idx == self.env_spec.total_sampling_act_dim

    return sampled_actions, logits, log_probs, entropy, self_kl

  def get_kl(self, my_logits, other_logits):
    """Calculate KL between one policy output and another."""
    kl = []
    for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
      sampling_dim = self.env_spec.sampling_dim(act_dim, act_type)
      single_my_logits = my_logits[i]
      single_other_logits = other_logits[i]
      if self.env_spec.is_discrete(act_type):
        my_probs = tf.nn.softmax(single_my_logits)
        my_log_probs = tf.nn.log_softmax(single_my_logits)
        other_log_probs = tf.nn.log_softmax(single_other_logits)
        my_kl = tf.reduce_sum(my_probs * (my_log_probs - other_log_probs), -1)
      elif self.env_spec.is_box(act_type):
        my_means = single_my_logits[:, :sampling_dim / 2]
        my_std = single_my_logits[:, sampling_dim / 2:]
        other_means = single_other_logits[:, :sampling_dim / 2]
        other_std = single_other_logits[:, sampling_dim / 2:]
        my_kl = tf.reduce_sum(
            tf.log(other_std / my_std) +
            (tf.square(my_std) + tf.square(my_means - other_means)) /
            (2.0 * tf.square(other_std)) - 0.5,
            -1)
      else:
        assert False

      kl.append(my_kl)

    return kl

  def single_step(self, prev, cur, greedy=False):
    """Single RNN step.  Equivalently, single-time-step sampled actions."""
    prev_internal_state, prev_actions, _, _, _, _ = prev
    obs, actions = cur  # state observed and action taken at this time step

    # feed into RNN cell
    output, next_state = self.core(
        obs, prev_internal_state, prev_actions)

    # sample actions with values and log-probs
    (actions, logits, log_probs,
     entropy, self_kl) = self.sample_actions(
        output, actions=actions, greedy=greedy)

    return (next_state, tuple(actions), tuple(logits), tuple(log_probs),
            tuple(entropy), tuple(self_kl))

  def sample_step(self, obs, prev_internal_state, prev_actions, greedy=False):
    """Sample single step from policy."""
    (next_state, sampled_actions, logits, log_probs,
     entropies, self_kls) = self.single_step(
        (prev_internal_state, prev_actions, None, None, None, None),
        (obs, None), greedy=greedy)
    return next_state, sampled_actions

  def multi_step(self, all_obs, initial_state, all_actions):
    """Calculate log-probs and other calculations on batch of episodes."""
    batch_size = tf.shape(initial_state)[0]
    time_length = tf.shape(all_obs[0])[0]
    initial_actions = [act[0] for act in all_actions]
    all_actions = [tf.concat([act[1:], act[0:1]], 0)
                   for act in all_actions]  # "final" action is dummy

    (internal_states, _, logits, log_probs,
     entropies, self_kls) = tf.scan(
        self.single_step,
        (all_obs, all_actions),
        initializer=self.get_initializer(
            batch_size, initial_state, initial_actions))

    # remove "final" computations
    log_probs = [log_prob[:-1] for log_prob in log_probs]
    entropies = [entropy[:-1] for entropy in entropies]
    self_kls = [self_kl[:-1] for self_kl in self_kls]

    return internal_states, logits, log_probs, entropies, self_kls

  def get_initializer(self, batch_size, initial_state, initial_actions):
    """Get initializer for RNN."""
    logits_init = []
    log_probs_init = []
    for act_dim, act_type in self.env_spec.act_dims_and_types:
      sampling_dim = self.env_spec.sampling_dim(act_dim, act_type)
      logits_init.append(tf.zeros([batch_size, sampling_dim]))
      log_probs_init.append(tf.zeros([batch_size]))
    entropy_init = [tf.zeros([batch_size]) for _ in self.env_spec.act_dims]
    self_kl_init = [tf.zeros([batch_size]) for _ in self.env_spec.act_dims]

    return (initial_state,
            tuple(initial_actions),
            tuple(logits_init), tuple(log_probs_init),
            tuple(entropy_init),
            tuple(self_kl_init))

  def calculate_kl(self, my_logits, other_logits):
    """Calculate KL between one policy and another on batch of episodes."""
    batch_size = tf.shape(my_logits[0])[1]
    time_length = tf.shape(my_logits[0])[0]

    reshaped_my_logits = [
        tf.reshape(my_logit, [batch_size * time_length, -1])
        for my_logit in my_logits]
    reshaped_other_logits = [
        tf.reshape(other_logit, [batch_size * time_length, -1])
        for other_logit in other_logits]

    kl = self.get_kl(reshaped_my_logits, reshaped_other_logits)
    kl = [tf.reshape(kkl, [time_length, batch_size])
          for kkl in kl]
    return kl


class MLPPolicy(Policy):
  """Non-recurrent policy."""

  def get_cell(self):
    self.cell_input_dim = self.internal_dim

    def mlp(cell_input, prev_internal_state):
      w1 = tf.get_variable('w1', [self.cell_input_dim, self.internal_dim])
      b1 = tf.get_variable('b1', [self.internal_dim])

      w2 = tf.get_variable('w2', [self.internal_dim, self.internal_dim])
      b2 = tf.get_variable('b2', [self.internal_dim])

      w3 = tf.get_variable('w3', [self.internal_dim, self.internal_dim])
      b3 = tf.get_variable('b3', [self.internal_dim])

      proj = tf.get_variable(
          'proj', [self.internal_dim, self.output_dim])

      hidden = cell_input
      hidden = tf.tanh(tf.nn.bias_add(tf.matmul(hidden, w1), b1))
      hidden = tf.tanh(tf.nn.bias_add(tf.matmul(hidden, w2), b2))

      output = tf.matmul(hidden, proj)

      return output, hidden

    return mlp

  def single_step(self, obs, actions, prev_actions, greedy=False):
    """Single step."""
    batch_size = tf.shape(obs[0])[0]
    prev_internal_state = tf.zeros([batch_size, self.internal_dim])

    output, next_state = self.core(
        obs, prev_internal_state, prev_actions)

    # sample actions with values and log-probs
    (actions, logits, log_probs,
     entropy, self_kl) = self.sample_actions(
        output, actions=actions, greedy=greedy)

    return (next_state, tuple(actions), tuple(logits), tuple(log_probs),
            tuple(entropy), tuple(self_kl))

  def sample_step(self, obs, prev_internal_state, prev_actions, greedy=False):
    """Sample single step from policy."""
    (next_state, sampled_actions, logits, log_probs,
     entropies, self_kls) = self.single_step(obs, None, prev_actions,
                                             greedy=greedy)
    return next_state, sampled_actions

  def multi_step(self, all_obs, initial_state, all_actions):
    """Calculate log-probs and other calculations on batch of episodes."""
    batch_size = tf.shape(initial_state)[0]
    time_length = tf.shape(all_obs[0])[0]

    # first reshape inputs as a single batch
    reshaped_obs = []
    for obs, (obs_dim, obs_type) in zip(all_obs, self.env_spec.obs_dims_and_types):
      if self.env_spec.is_discrete(obs_type):
        reshaped_obs.append(tf.reshape(obs, [time_length * batch_size]))
      elif self.env_spec.is_box(obs_type):
        reshaped_obs.append(tf.reshape(obs, [time_length * batch_size, obs_dim]))

    reshaped_act = []
    reshaped_prev_act = []
    for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
      act = tf.concat([all_actions[i][1:], all_actions[i][0:1]], 0)
      prev_act = all_actions[i]
      if self.env_spec.is_discrete(act_type):
        reshaped_act.append(tf.reshape(act, [time_length * batch_size]))
        reshaped_prev_act.append(
            tf.reshape(prev_act, [time_length * batch_size]))
      elif self.env_spec.is_box(act_type):
        reshaped_act.append(
            tf.reshape(act, [time_length * batch_size, act_dim]))
        reshaped_prev_act.append(
            tf.reshape(prev_act, [time_length * batch_size, act_dim]))

    # now inputs go into single step as one large batch
    (internal_states, _, logits, log_probs,
     entropies, self_kls) = self.single_step(
         reshaped_obs, reshaped_act, reshaped_prev_act)

    # reshape the outputs back to original time-major format
    internal_states = tf.reshape(internal_states, [time_length, batch_size, -1])
    logits = [tf.reshape(logit, [time_length, batch_size, -1])
              for logit in logits]
    log_probs = [tf.reshape(log_prob, [time_length, batch_size])[:-1]
                 for log_prob in log_probs]
    entropies = [tf.reshape(ent, [time_length, batch_size])[:-1]
                 for ent in entropies]
    self_kls = [tf.reshape(self_kl, [time_length, batch_size])[:-1]
                for self_kl in self_kls]

    return internal_states, logits, log_probs, entropies, self_kls
