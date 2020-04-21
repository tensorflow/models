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
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math
import tensorflow as tf

from official.nlp.modeling.layers import dense_einsum
from official.nlp.modeling.layers import masked_softmax


@tf.keras.utils.register_keras_serializable(package="Text")
class MultiHeadAttention(tf.keras.layers.Layer):
  """MultiHeadAttention layer.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `query`, `key,` `value` are the same, then
  this is self-attention. Each timestep in `query` attends to the
  corresponding sequence in `key`, and returns a fixed-width vector.

  This layer first projects `query`, `key` and `value`. These are
  (effectively) a list of tensors of length `num_attention_heads`, where the
  corresponding shapes are [batch_size, query_seq_length, key_size],
  [batch_size, seq_length, key_size], [batch_size, seq_length, value_size].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor.

  Finally, the result tensor with the last dimension as value_size can take an
  linear projection and return.

  Arguments:
    num_heads: Number of attention heads.
    key_size: Size of each attention head for query and key.
    value_size:  Size of each attention head for value.
    dropout: Dropout probability.
    use_bias: Boolean, whether the dense layers use bias vectors.
    output_shape: The expected shape of an output tensor, besides the batch and
      sequence dims. If not specified, projects back to the key feature dim.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def __init__(self,
               num_heads,
               key_size,
               value_size=None,
               dropout_rate=0.0,
               use_bias=True,
               output_shape=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self._num_heads = num_heads
    self._key_size = key_size
    self._value_size = value_size if value_size else key_size
    self._dropout_rate = dropout_rate
    self._use_bias = use_bias
    self._output_shape = output_shape
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    self._query_dense = dense_einsum.DenseEinsum(
        output_shape=(self._num_heads, self._key_size),
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="query")

    self._key_dense = dense_einsum.DenseEinsum(
        output_shape=(self._num_heads, self._key_size),
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="key")

    self._value_dense = dense_einsum.DenseEinsum(
        output_shape=(self._num_heads, self._value_size),
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="value")

    self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])

    self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

  def get_config(self):
    config = {
        "num_heads":
            self._num_heads,
        "key_size":
            self._key_size,
        "value_size":
            self._value_size,
        "dropout_rate":
            self._dropout_rate,
        "use_bias":
            self._use_bias,
        "output_shape":
            self._output_shape,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint)
    }
    base_config = super(MultiHeadAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    if self._output_shape:
      output_shape = self._output_shape
    else:
      input_shape = tf.TensorShape(input_shape[0])
      output_shape = input_shape[-1]
    self._output_dense = dense_einsum.DenseEinsum(
        output_shape=output_shape,
        num_summed_dimensions=2,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="attention_output")
    super(MultiHeadAttention, self).build(input_shape)

  def call(self, inputs, attention_mask=None):
    """Implements the forward pass.

    Size glossary:
      * Number of heads (H): the number of attention heads.
      * Value size (V): the size of each value embedding per head.
      * Key size (K): the size of each key embedding per head. Equally, the size
          of each query embedding per head. Typically K <= V.
      * Batch size (B).
      * Query (target) sequence length (T).
      * Value (source) sequence length (S).

    Args:
      inputs: List of the following tensors:
        * query: Query `Tensor` of shape `[B, T, dim]`.
        * value: Value `Tensor` of shape `[B, S, dim]`.
        * key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will
          use `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
        attention to certain positions.

    Returns:
      attention_output: The result of the computation, of shape [B, F, N, V] or
        [B, F, E], where `N` is the number of heads and `E` is the query input
        last dimension.
    """
    inputs_len = len(inputs)
    if inputs_len > 3 or inputs_len < 2:
      raise ValueError(
          "Expects inputs list of length 2 or 3, namely [query, value] or "
          "[query, value, key]. "
          "Given length: %d" % inputs_len)
    query = inputs[0]
    value = inputs[1]
    key = inputs[2] if inputs_len == 3 else value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, T, N ,H]
    query_tensor = self._query_dense(query)

    # `key_tensor` = [B, S, N, H]
    key_tensor = self._key_dense(key)

    # `value_tensor` = [B, S, N, H]
    value_tensor = self._value_dense(value)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BSNH,BTNH->BNTS", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, T, S]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum("BNTS,BSNH->BTNH", attention_probs,
                                 value_tensor)

    attention_output = self._output_dense(attention_output)
    return attention_output


@tf.keras.utils.register_keras_serializable(package="Text")
class CachedAttention(MultiHeadAttention):
  """Attention layer with cache used for auto-agressive decoding.

  Arguments are the same as `MultiHeadAttention` layer.
  """

  def _update_cache(self, key_tensor, value_tensor, cache, decode_loop_step):
    """Updates cache states and gets full-length key/value tensors."""
    # Combines cached keys and values with new keys and values.
    if decode_loop_step is not None:
      # TPU special case.
      key_seq_dim = cache["key"].shape.as_list()[1]
      indices = tf.reshape(
          tf.one_hot(decode_loop_step, key_seq_dim, dtype=key_tensor.dtype),
          [1, key_seq_dim, 1, 1])
      key_tensor = cache["key"] + key_tensor * indices
      value_seq_dim = cache["value"].shape.as_list()[1]
      indices = tf.reshape(
          tf.one_hot(decode_loop_step, value_seq_dim, dtype=value_tensor.dtype),
          [1, value_seq_dim, 1, 1])
      value_tensor = cache["value"] + value_tensor * indices
    else:
      key_tensor = tf.concat(
          [tf.cast(cache["key"], key_tensor.dtype), key_tensor], axis=1)
      value_tensor = tf.concat(
          [tf.cast(cache["value"], value_tensor.dtype), value_tensor], axis=1)

    # Update cache
    cache["key"] = key_tensor
    cache["value"] = value_tensor

    return key_tensor, value_tensor

  def call(self, inputs, decode_loop_step=None):
    from_tensor = inputs[0]
    to_tensor = inputs[1]
    attention_mask = inputs[2] if len(inputs) >= 3 else None
    cache = inputs[3] if len(inputs) >= 4 else None
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self._query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self._key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self._value_dense(to_tensor)

    if cache:
      key_tensor, value_tensor = self._update_cache(key_tensor, value_tensor,
                                                    cache, decode_loop_step)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)
    # `context_layer` = [B, F, N, H]
    attention_output = tf.einsum("BNFT,BTNH->BFNH", attention_probs,
                                 value_tensor)
    attention_output = self._output_dense(attention_output)
    return attention_output, cache
