# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""
Modified from attention.py
"""

"""Keras-based attention layer."""
# pylint: disable=g-classes-have-attributes
import math

import tensorflow as tf

EinsumDense = tf.keras.layers.experimental.EinsumDense
MultiHeadAttention = tf.keras.layers.MultiHeadAttention


@tf.keras.utils.register_keras_serializable(package="Text")
class RoformerAttention(tf.keras.layers.MultiHeadAttention):
  def roformer_recompute_qkv(self,
                             q,
                             k,
                             v):
      input_shape = tf_utils.get_shape_list(q)
      batch_size = input_shape[0]
      length = input_shape[1]
      num_heads1 = input_shape[2]
      head_size = input_shape[3]

      input_shape2 = tf_utils.get_shape_list(k)
      length2 = input_shape2[1]
      num_heads2 = input_shape2[2]
      head_size2 = input_shape2[3]

      position_ids = tf.cast(tf.range(length), tf.float32)[None]  # (1, length)
      num_timescales = self._hidden_size // 2
      indices = tf.cast(tf.range(num_timescales), tf.float32)  # (d/2)
      indices = tf.pow(10000.0, -2 * indices / num_timescales)  # (d/2,)
      embeddings = tf.einsum('bn,d->bnd', position_ids, indices)  # (1, length, d/2)
      sin_emb = tf.repeat(tf.sin(embeddings), repeats=2, axis=-1)
      sin_emb = tf.expand_dims(sin_emb, 2)  # (1, length, 1, d/2)
      cos_emb = tf.repeat(tf.cos(embeddings), repeats=2, axis=-1)
      cos_emb = tf.expand_dims(cos_emb, 2)  # (1, length, 1, d/2)
      q2 = tf.stack([-q[..., 1::2], q[..., ::2]], axis=4)
      q2 = tf.reshape(q2, (batch_size, length, num_heads1, head_size))
      k2 = tf.stack([-k[..., 1::2], k[..., ::2]], axis=4)
      k2 = tf.reshape(k2, (batch_size, length2, num_heads2, head_size2))
      ret_q = q * cos_emb + q2 * sin_emb
      ret_w = k * cos_emb + k2 * sin_emb
      return ret_q, ret_w, v


  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None):
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

    # TODO Roformer Implementation here
    query, key, value = self.roformer_recompute_qkv(query, key, value)

    attention_output, attention_scores = self._compute_attention(
        query, key, value, attention_mask, training)
    attention_output = self._output_dense(attention_output)

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output