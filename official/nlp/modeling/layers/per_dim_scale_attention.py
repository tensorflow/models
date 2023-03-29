# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Keras-based attention layer with learnable per dim scaling."""
import gin
import numpy as np
import tensorflow as tf


@gin.configurable
@tf.keras.utils.register_keras_serializable(package='Text')
class PerDimScaleAttention(tf.keras.layers.MultiHeadAttention):
  """Learn scales for individual dims.

     It can improve quality but might hurt training stability.
  """

  def _build_from_signature(self, query, value, key=None):
    super()._build_from_signature(query=query, value=value, key=key)  # pytype: disable=attribute-error
    self._scale_dim = self._key_dim
    with tf.init_scope():
      self.per_dim_scale = self.add_weight(
          name='per_dim_scale',
          shape=(self._scale_dim,),
          initializer='zeros',
          dtype=self.dtype,
          trainable=True)

  def _scale_query(self, query):
    # 1.0/tf.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
    # can avoid unnecessary XLA op fusion mess on TPU.
    r_softplus_0 = 1.442695041
    scale = tf.constant(
        r_softplus_0 / np.sqrt(float(self._scale_dim)), dtype=query.dtype)

    scale *= tf.nn.softplus(self.per_dim_scale)
    return query * scale

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         attention_mask=None,
                         training=None):
    query = self._scale_query(query)

    attention_scores = tf.einsum(self._dot_product_equation, key, query)

    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum(self._combine_equation,
                                 attention_scores_dropout, value)
    return attention_output, attention_scores

  def call(
      self,
      query,
      value,
      key=None,
      attention_mask=None,
      return_attention_scores=False,
      training=None,
  ):
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, H]
    value = self._value_dense(value)

    attention_output, attention_scores = self._compute_attention(
        query, key, value, attention_mask, training)
    attention_output = self._output_dense(attention_output)

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output
