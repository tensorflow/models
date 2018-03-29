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

"""Variational Dropout Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class VariationalDropoutWrapper(tf.contrib.rnn.RNNCell):
  """Add variational dropout to a RNN cell."""

  def __init__(self, cell, batch_size, input_size, recurrent_keep_prob,
               input_keep_prob):
    self._cell = cell
    self._recurrent_keep_prob = recurrent_keep_prob
    self._input_keep_prob = input_keep_prob

    def make_mask(keep_prob, units):
      random_tensor = keep_prob
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      random_tensor += tf.random_uniform(tf.stack([batch_size, units]))
      return tf.floor(random_tensor) / keep_prob

    self._recurrent_mask = make_mask(recurrent_keep_prob,
                                     self._cell.state_size[0])
    self._input_mask = self._recurrent_mask

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    dropped_inputs = inputs * self._input_mask
    dropped_state = (state[0], state[1] * self._recurrent_mask)
    new_h, new_state = self._cell(dropped_inputs, dropped_state, scope)
    return new_h, new_state
