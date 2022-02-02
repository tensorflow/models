# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
# Lint as: python3
"""Layers for embedding."""
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module


class AttentionPooling(base_layers.BaseLayer):
  """A basic attention pooling layer."""

  def __init__(self, scalar=True, **kwargs):
    self.scalar = scalar
    # Attention logits should not have activation post linear layer so it can
    # be positive or negative. This would enable the attention distribution to
    # be anything that the network likes. Using relu activation makes the
    # attention distribution biased towards uniform distribution.
    # This gets better results for attention pooling. Though some outputs are
    # emphasized for making classification decision, all other outputs have
    # a non zero probability of influencing the class. This seems to result
    # in better backprop.
    self.attention = dense_layers.BaseQDenseVarLen(units=1, rank=3, **kwargs)
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    super(AttentionPooling, self).__init__(**kwargs)

  def build(self, input_shapes):
    self.feature_size = input_shapes[-1]

  def call(self, inputs, mask, inverse_normalizer):
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    batch_size = self.get_batch_dimension(inputs)
    attn_logits = self.attention(inputs, mask, inverse_normalizer)
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      invalid_mask = (1 - mask) * self.parameters.invalid_logit
      attn_logits = attn_logits * mask + invalid_mask
    attn_logits = tf.reshape(attn_logits, [batch_size, -1])
    attention = tf.nn.softmax(attn_logits, axis=-1)
    attention = self.qrange_sigmoid(attention, tf_only=True)
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      inputs = tf.reshape(inputs, [-1, self.feature_size])
    else:
      attention = tf.expand_dims(attention, axis=1)
    pre_logits = self.qactivation(tf.matmul(attention, inputs))
    return tf.reshape(pre_logits, [batch_size, self.feature_size])


class TreeInductionLayer(base_layers.BaseLayer):
  """A basic tree induction layer."""

  def __init__(self, **kwargs):
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    super(TreeInductionLayer, self).__init__(**kwargs)

  def call(self, keys, queries, sequence_length):
    key_dim = keys.get_shape().as_list()[-1]
    query_dim = queries.get_shape().as_list()[-1]
    assert key_dim == query_dim, "Last dimension of keys/queries should match."

    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      sequence_mask = tf.sequence_mask(
          sequence_length, maxlen=tf.shape(keys)[1], dtype=tf.float32)
      sequence_mask = tf.expand_dims(sequence_mask, axis=2)
      attn_mask = tf.matmul(sequence_mask, sequence_mask, transpose_b=True)

      attn_logits = self.qactivation(tf.matmul(keys, queries, transpose_b=True))
      invalid_attn_mask = (1 - attn_mask) * self.parameters.invalid_logit
      return attn_logits * attn_mask + invalid_attn_mask
    else:
      assert self.get_batch_dimension(keys) == 1
      assert self.get_batch_dimension(queries) == 1
      keys = tf.reshape(keys, [-1, key_dim])
      queries = tf.reshape(queries, [-1, key_dim])

      result = self.qactivation(tf.matmul(keys, queries, transpose_b=True))
      # TODO(b/171063452): Bug needs to be fixed to handle this correctly.
      # seq_dim = tf.shape(result)[1]
      # result = tf.reshape(result, [1, seq_dim, seq_dim])
      return result
