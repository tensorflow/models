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

"""Baseline model for value estimates.

Implements the value component of the neural network.
In some cases this is just an additional linear layer on the policy.
In other cases, it is a completely separate neural network.
"""

import tensorflow as tf
import numpy as np


class Baseline(object):
  def __init__(self, env_spec, internal_policy_dim,
               input_prev_actions=True,
               input_time_step=False,
               input_policy_state=True,
               n_hidden_layers=0,
               hidden_dim=64,
               tau=0.0):
    self.env_spec = env_spec
    self.internal_policy_dim = internal_policy_dim
    self.input_prev_actions = input_prev_actions
    self.input_time_step = input_time_step
    self.input_policy_state = input_policy_state
    self.n_hidden_layers = n_hidden_layers
    self.hidden_dim = hidden_dim
    self.tau = tau

    self.matrix_init = tf.truncated_normal_initializer(stddev=0.01)

  def get_inputs(self, time_step, obs, prev_actions,
                 internal_policy_states):
    """Get inputs to network as single tensor."""
    inputs = [tf.ones_like(time_step)]
    input_dim = 1

    if not self.input_policy_state:
      for i, (obs_dim, obs_type) in enumerate(self.env_spec.obs_dims_and_types):
        if self.env_spec.is_discrete(obs_type):
          inputs.append(
              tf.one_hot(obs[i], obs_dim))
          input_dim += obs_dim
        elif self.env_spec.is_box(obs_type):
          cur_obs = obs[i]
          inputs.append(cur_obs)
          inputs.append(cur_obs ** 2)
          input_dim += obs_dim * 2
        else:
          assert False

      if self.input_prev_actions:
        for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
          if self.env_spec.is_discrete(act_type):
            inputs.append(
                tf.one_hot(prev_actions[i], act_dim))
            input_dim += act_dim
          elif self.env_spec.is_box(act_type):
            inputs.append(prev_actions[i])
            input_dim += act_dim
          else:
            assert False

    if self.input_policy_state:
      inputs.append(internal_policy_states)
      input_dim += self.internal_policy_dim

    if self.input_time_step:
      scaled_time = 0.01 * time_step
      inputs.extend([scaled_time, scaled_time ** 2, scaled_time ** 3])
      input_dim += 3

    return input_dim, tf.concat(inputs, 1)

  def reshape_batched_inputs(self, all_obs, all_actions,
                             internal_policy_states, policy_logits):
    """Reshape inputs from [time_length, batch_size, ...] to
    [time_length * batch_size, ...].

    This allows for computing the value estimate in one go.
    """
    batch_size = tf.shape(all_obs[0])[1]
    time_length = tf.shape(all_obs[0])[0]

    reshaped_obs = []
    for obs, (obs_dim, obs_type) in zip(all_obs, self.env_spec.obs_dims_and_types):
      if self.env_spec.is_discrete(obs_type):
        reshaped_obs.append(tf.reshape(obs, [time_length * batch_size]))
      elif self.env_spec.is_box(obs_type):
        reshaped_obs.append(tf.reshape(obs, [time_length * batch_size, obs_dim]))

    reshaped_prev_act = []
    reshaped_policy_logits = []
    for i, (act_dim, act_type) in enumerate(self.env_spec.act_dims_and_types):
      prev_act = all_actions[i]
      if self.env_spec.is_discrete(act_type):
        reshaped_prev_act.append(
            tf.reshape(prev_act, [time_length * batch_size]))
      elif self.env_spec.is_box(act_type):
        reshaped_prev_act.append(
            tf.reshape(prev_act, [time_length * batch_size, act_dim]))

      reshaped_policy_logits.append(
          tf.reshape(policy_logits[i], [time_length * batch_size, -1]))

    reshaped_internal_policy_states = tf.reshape(
        internal_policy_states,
        [time_length * batch_size, self.internal_policy_dim])

    time_step = (float(self.input_time_step) *
                 tf.expand_dims(
                     tf.to_float(tf.range(time_length * batch_size) /
                                 batch_size), -1))

    return (time_step, reshaped_obs, reshaped_prev_act,
            reshaped_internal_policy_states,
            reshaped_policy_logits)

  def get_values(self, all_obs, all_actions, internal_policy_states,
                 policy_logits):
    """Get value estimates given input."""
    batch_size = tf.shape(all_obs[0])[1]
    time_length = tf.shape(all_obs[0])[0]

    (time_step, reshaped_obs, reshaped_prev_act,
     reshaped_internal_policy_states,
     reshaped_policy_logits) = self.reshape_batched_inputs(
         all_obs, all_actions, internal_policy_states, policy_logits)

    input_dim, inputs = self.get_inputs(
        time_step, reshaped_obs, reshaped_prev_act,
        reshaped_internal_policy_states)

    for depth in xrange(self.n_hidden_layers):
      with tf.variable_scope('value_layer%d' % depth):
        w = tf.get_variable('w', [input_dim, self.hidden_dim])
        inputs = tf.nn.tanh(tf.matmul(inputs, w))
        input_dim = self.hidden_dim

    w_v = tf.get_variable('w_v', [input_dim, 1],
                          initializer=self.matrix_init)
    values = tf.matmul(inputs, w_v)
    values = tf.reshape(values, [time_length, batch_size])

    inputs = inputs[:-batch_size]  # remove "final vals"
    return values, inputs, w_v


class UnifiedBaseline(Baseline):
  """Baseline for Unified PCL."""

  def get_values(self, all_obs, all_actions, internal_policy_states,
                 policy_logits):
    batch_size = tf.shape(all_obs[0])[1]
    time_length = tf.shape(all_obs[0])[0]

    (time_step, reshaped_obs, reshaped_prev_act,
     reshaped_internal_policy_states,
     reshaped_policy_logits) = self.reshape_batched_inputs(
         all_obs, all_actions, internal_policy_states, policy_logits)

    def f_transform(q_values, tau):
      max_q = tf.reduce_max(q_values, -1, keep_dims=True)
      return tf.squeeze(max_q, [-1]) + tau * tf.log(
          tf.reduce_sum(tf.exp((q_values - max_q) / tau), -1))

    assert len(reshaped_policy_logits) == 1
    values = f_transform((self.tau + self.eps_lambda) * reshaped_policy_logits[0],
                         (self.tau + self.eps_lambda))
    values = tf.reshape(values, [time_length, batch_size])

    # not used
    input_dim, inputs = self.get_inputs(
        time_step, reshaped_obs, reshaped_prev_act,
        reshaped_internal_policy_states)

    w_v = tf.get_variable('w_v', [input_dim, 1],
                          initializer=self.matrix_init)

    inputs = inputs[:-batch_size]  # remove "final vals"

    return values, inputs, w_v
