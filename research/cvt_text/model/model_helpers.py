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

"""Utilities for building the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def project(input_layers, size, name='projection'):
  return tf.add_n([tf.layers.dense(layer, size, name=name + '_' + str(i))
                   for i, layer in enumerate(input_layers)])


def lstm_cell(cell_size, keep_prob, num_proj):
  return tf.contrib.rnn.DropoutWrapper(
      tf.contrib.rnn.LSTMCell(cell_size, num_proj=min(cell_size, num_proj)),
      output_keep_prob=keep_prob)


def multi_lstm_cell(cell_sizes, keep_prob, num_proj):
  return tf.contrib.rnn.MultiRNNCell([lstm_cell(cell_size, keep_prob, num_proj)
                                      for cell_size in cell_sizes])


def masked_ce_loss(logits, labels, mask, sparse=False, roll_direction=0):
  if roll_direction != 0:
    labels = _roll(labels, roll_direction, sparse)
    mask *= _roll(mask, roll_direction, True)
  ce = ((tf.nn.sparse_softmax_cross_entropy_with_logits if sparse
         else tf.nn.softmax_cross_entropy_with_logits_v2)
        (logits=logits, labels=labels))
  return tf.reduce_sum(mask * ce) / tf.to_float(tf.reduce_sum(mask))


def _roll(arr, direction, sparse=False):
  if sparse:
    return tf.concat([arr[:, direction:], arr[:, :direction]], axis=1)
  return tf.concat([arr[:, direction:, :], arr[:, :direction, :]], axis=1)
