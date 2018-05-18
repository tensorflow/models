# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Transformer model helper methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

_NEG_INF = -1e9


def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  position = tf.to_float(tf.range(length))
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal


def get_decoder_self_attention_bias(length):
  """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = _NEG_INF * (1.0 - valid_locs)
  return decoder_bias


def get_padding(x, padding_value=0):
  """Return float tensor representing the padding values in x.

  Args:
    x: int tensor with any shape
    padding_value: int value that

  Returns:
    flaot tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
  with tf.name_scope("padding"):
    return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
  """Calculate bias tensor from padding values in tensor.

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.

  Args:
    x: int tensor with shape [batch_size, length]

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF
    attention_bias = tf.expand_dims(
        tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias
