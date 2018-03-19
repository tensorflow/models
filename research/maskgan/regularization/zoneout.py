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

"""Zoneout Wrapper"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ZoneoutWrapper(tf.contrib.rnn.RNNCell):
  """Add Zoneout to a RNN cell."""

  def __init__(self, cell, zoneout_drop_prob, is_training=True):
    self._cell = cell
    self._zoneout_prob = zoneout_drop_prob
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    output, new_state = self._cell(inputs, state, scope)
    if not isinstance(self._cell.state_size, tuple):
      new_state = tf.split(value=new_state, num_or_size_splits=2, axis=1)
      state = tf.split(value=state, num_or_size_splits=2, axis=1)
    final_new_state = [new_state[0], new_state[1]]
    if self._is_training:
      for i, state_element in enumerate(state):
        random_tensor = 1 - self._zoneout_prob  # keep probability
        random_tensor += tf.random_uniform(tf.shape(state_element))
        # 0. if [zoneout_prob, 1.0) and 1. if [1.0, 1.0 + zoneout_prob)
        binary_tensor = tf.floor(random_tensor)
        final_new_state[
            i] = (new_state[i] - state_element) * binary_tensor + state_element
    else:
      for i, state_element in enumerate(state):
        final_new_state[
            i] = state_element * self._zoneout_prob + new_state[i] * (
                1 - self._zoneout_prob)
    if isinstance(self._cell.state_size, tuple):
      return output, tf.contrib.rnn.LSTMStateTuple(
          final_new_state[0], final_new_state[1])

    return output, tf.concat([final_new_state[0], final_new_state[1]], 1)
