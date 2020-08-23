# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras-based attention layer."""
# pylint: disable=g-classes-have-attributes

import math
import string

import tensorflow as tf


EinsumDense = tf.keras.layers.experimental.EinsumDense
_CHR_IDX = string.ascii_lowercase


MultiHeadAttention = tf.keras.layers.MultiHeadAttention


@tf.keras.utils.register_keras_serializable(package="Text")
class CachedAttention(tf.keras.layers.MultiHeadAttention):
  """Attention layer with cache used for auto-agressive decoding.

  Arguments are the same as `MultiHeadAttention` layer.
  """

  def _update_cache(self, key, value, cache, decode_loop_step):
    """Updates cache states and gets full-length key/value tensors."""
    # Combines cached keys and values with new keys and values.
    if decode_loop_step is not None:
      # TPU special case.
      key_seq_dim = cache["key"].shape.as_list()[1]
      indices = tf.reshape(
          tf.one_hot(decode_loop_step, key_seq_dim, dtype=key.dtype),
          [1, key_seq_dim, 1, 1])
      key = cache["key"] + key * indices
      value_seq_dim = cache["value"].shape.as_list()[1]
      indices = tf.reshape(
          tf.one_hot(decode_loop_step, value_seq_dim, dtype=value.dtype),
          [1, value_seq_dim, 1, 1])
      value = cache["value"] + value * indices
    else:
      key = tf.concat([tf.cast(cache["key"], key.dtype), key], axis=1)
      value = tf.concat([tf.cast(cache["value"], value.dtype), value], axis=1)

    # Update cache
    cache["key"] = key
    cache["value"] = value

    return key, value

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           cache=None,
           decode_loop_step=None,
           return_attention_scores=False):
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, F, N ,H]
    query = self._query_dense(query)

    # `key` = [B, T, N, H]
    key = self._key_dense(key)

    # `value` = [B, T, N, H]
    value = self._value_dense(value)

    if cache:
      key, value = self._update_cache(key, value, cache, decode_loop_step)

    query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum(self._dot_product_equation, key, query)

    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, F, T]
    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores = self._dropout_layer(attention_scores)
    # `context_layer` = [B, F, N, H]
    attention_output = tf.einsum(self._combine_equation, attention_scores,
                                 value)
    attention_output = self._output_dense(attention_output)
    if return_attention_scores:
      return attention_output, attention_scores, cache
    return attention_output, cache
