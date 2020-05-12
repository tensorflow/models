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
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections
import math
import string

import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import masked_softmax

EinsumDense = tf.keras.layers.experimental.EinsumDense
_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(qkv_rank, attn_axes):
  """Builds einsum equations for the attention computation.

  Query, key, value inputs after projection are expected to have the shape as:
  (bs, <non-attention dims>, <attention dims>, num_heads, channels).
  bs and <non-attention dims> are treated as <batch dims>.
  The attention operations can be generalized:
  (1) Query-key dot product:
  (<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
  <key attention dims>, num_heads, channels) -> (<batch dims>,
  num_heads, <query attention dims>, <key attention dims>)
  (2) Combination:
  (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
  (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
  <query attention dims>, num_heads, channels)

  Args:
    qkv_rank: the rank of query, key, value tensors.
    attn_axes: a list/tuple of axes, [1, rank), that will do attention.
  Returns:
    Einsum equations.
  """
  target_notation = _CHR_IDX[:qkv_rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(qkv_rank), attn_axes + (qkv_rank - 1,)))

  letter_offset = qkv_rank
  source_notation = ""
  for i in range(qkv_rank):
    if i in batch_dims or i == qkv_rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                        product_notation)
  combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                    target_notation)
  return dot_product_equation, combine_equation


def _build_proj_equation(free_dims, bound_dims, output_dims):
  """Builds an einsum equation for projections inside multi-head attention."""
  input_str = ""
  kernel_str = ""
  output_str = ""
  bias_axes = ""
  letter_offset = 0
  for i in range(free_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    output_str += char

  letter_offset += free_dims
  for i in range(bound_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    kernel_str += char

  letter_offset += bound_dims
  for i in range(output_dims):
    char = _CHR_IDX[i + letter_offset]
    kernel_str += char
    output_str += char
    bias_axes += char
  equation = "%s,%s->%s" % (input_str, kernel_str, output_str)
  # The output rank does not consider the batch dimension.
  output_rank = len(output_str) - 1

  return equation, bias_axes, output_rank


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


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
    use_bias: Boolean, whether the dense layers use bias vectors/matrices.
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
    inputs_len = len(input_shape)
    if inputs_len > 3 or inputs_len < 2:
      raise ValueError(
          "Expects inputs list of length 2 or 3, namely [query, value] or "
          "[query, value, key]. "
          "Given length: %d" % inputs_len)
    tensor_shapes = tf.nest.map_structure(tf.TensorShape, input_shape)
    query_shape = tensor_shapes[0]
    value_shape = tensor_shapes[1]
    key_shape = tensor_shapes[2] if inputs_len == 3 else value_shape

    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)

    free_dims = query_shape.rank - 1
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=1, output_dims=2)
    self._query_dense = EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank,
                                       [self._num_heads, self._key_size]),
        bias_axes=bias_axes if self._use_bias else None,
        name="query",
        **common_kwargs)
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        key_shape.rank - 1, bound_dims=1, output_dims=2)
    self._key_dense = EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank,
                                       [self._num_heads, self._key_size]),
        bias_axes=bias_axes if self._use_bias else None,
        name="key",
        **common_kwargs)
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        value_shape.rank - 1, bound_dims=1, output_dims=2)
    self._value_dense = EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank,
                                       [self._num_heads, self._value_size]),
        bias_axes=bias_axes if self._use_bias else None,
        name="value",
        **common_kwargs)
    self._dot_product_equation, self._combine_equation = (
        _build_attention_equation(output_rank + 1, attn_axes=(1,)))

    if self._output_shape:
      if not isinstance(self._output_shape, collections.abc.Sized):
        output_shape = [self._output_shape]
      else:
        output_shape = self._output_shape
    else:
      output_shape = [query_shape[-1]]
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=2, output_dims=len(output_shape))
    self._output_dense = EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank, output_shape),
        bias_axes=bias_axes if self._use_bias else None,
        name="attention_output",
        **common_kwargs)
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
      attention_output: The result of the computation, of shape [B, T, N, V] or
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
    attention_scores = tf.einsum(self._dot_product_equation, key_tensor,
                                 query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, T, S]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum(self._combine_equation, attention_probs,
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

  def call(self,
           inputs,
           attention_mask=None,
           cache=None,
           decode_loop_step=None):
    from_tensor = inputs[0]
    to_tensor = inputs[1]

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
    attention_scores = tf.einsum(self._dot_product_equation, key_tensor,
                                 query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)
    # `context_layer` = [B, F, N, H]
    attention_output = tf.einsum(self._combine_equation, attention_probs,
                                 value_tensor)
    attention_output = self._output_dense(attention_output)
    return attention_output, cache
