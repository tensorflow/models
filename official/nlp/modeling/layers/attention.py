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

from official.nlp.modeling.layers import masked_softmax


EinsumDense = tf.keras.layers.experimental.EinsumDense
MultiHeadAttention = tf.keras.layers.MultiHeadAttention
_CHR_IDX = string.ascii_lowercase


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


def _rel_shift(x, klen=-1):
  """Performs relative shift to form the relative attention score."""

  x = tf.transpose(x, perm=[1, 2, 0, 3])
  x_size = tf.shape(x)

  x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
  x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])
  x = tf.transpose(x, perm=[2, 0, 1, 3])

  return x


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

  return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


@tf.keras.utils.register_keras_serializable(package="Text")
class MultiHeadRelativeAttention(MultiHeadAttention):
  """A multi-head attention layer with relative attention + position encoding.

  This layer shares the same input/output projections as the common
  MultiHeadAttention layer.

  When it calculates attention logits, position encoding is projected to form
  relative keys. The logits are composed by shifted relative logits and content
  logits.

  **Note: This layer is currently experimental.

  Arguments:
    num_heads: The number of attention heads.
    key_dim: Size of each attention head for query and key.
    value_dim: Size of attention head for value.
    dropout: Dropout probability for attention.
    use_bias: Boolean, whether the dense layers use bias vectors/matrices.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
  Call args:
    query: Query `Tensor` of shape `[B, T, dim]`.
    value: Value `Tensor` of shape `[B, S, dim]`.
    content_attention_bias: Bias `Tensor` for content based attention of shape
      `[num_heads, dim]`.
    position_attention_bias: Bias `Tensor` for position based attention of shape
      `[num_heads, dim]`.
    relative_position_encoding: Relative positional encoding `Tensor` of shape
      `[B, L, dim]`.
    state: Optional `Tensor` of shape [B, M, E] where M is the length of the
      state or memory.
      If passed, this is also attended over as in Transformer XL.
    key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
      `value` for both `key` and `value`, which is the most common case.
    attention_mask: a boolean mask of shape `[B, T, S]`, that prevents attention
      to certain positions.
  """

  def _build_from_signature(self, query, value, key=None):
    super(MultiHeadRelativeAttention, self)._build_from_signature(
        query=query,
        value=value,
        key=key)
    if hasattr(query, "shape"):
      query_shape = tf.TensorShape(query.shape)
    else:
      query_shape = query
    if hasattr(value, "shape"):
      value_shape = tf.TensorShape(value.shape)
    else:
      value_shape = value
    if key is None:
      key_shape = value_shape
    elif hasattr(key, "shape"):
      key_shape = tf.TensorShape(key.shape)
    else:
      key_shape = key

    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)

    with tf.init_scope():
      free_dims = query_shape.rank - 1
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          key_shape.rank - 1, bound_dims=1, output_dims=2)
      self._encoding_dense = EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="encoding",
          **common_kwargs)

      output_shape = [query_shape[-1]]
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          free_dims, bound_dims=2, output_dims=len(output_shape))
      # TODO(allencwang) - replace all einsums with programmatic equations.
      einsum_equation = "abcd,ecd->abe"

      self._output_dense = EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1, output_shape),
          bias_axes=bias_axes if self._use_bias else None,
          name="attention_output",
          **common_kwargs)

  def _build_attention(self, rank):
    self._masked_softmax = masked_softmax.MaskedSoftmax(
        mask_expansion_axes=[1], normalization_axes=[2])
    self._dropout_layer = tf.keras.layers.Dropout(
        rate=self._dropout)

  def compute_attention(self,
                        query,
                        key,
                        value,
                        position,
                        content_attention_bias,
                        positional_attention_bias,
                        attention_mask=None):
    """Computes the attention.

    This function defines the computation inside `call` with projected
    multihead Q, K, V, R inputs.

    Args:
      query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
      key: Projected key `Tensor` of shape `[B, S + M, N, key_dim]`.
      value: Projected value `Tensor` of shape `[B, S + M, N, key_dim]`.
      position: Projected position `Tensor` of shape `[B, L, N, key_dim]`.
      content_attention_bias: Trainable bias parameter added to the query head
        when calculating the content-based attention score.
      positional_attention_bias: Trainable bias parameter added to the query
        head when calculating the position-based attention score.
      attention_mask: (default None) Optional mask that is added to attention
        logits. If state is not None, the mask source sequence dimension should
        extend M.

    Returns:
      attention_output: Multi-headed output of attention computation of shape
        `[B, T, N, key_dim]`.

    """
    content_attention = tf.einsum("bind,bjnd->bijn",
                                  query + content_attention_bias,
                                  key)

    positional_attention = tf.einsum("bind,bjnd->bijn",
                                     query + positional_attention_bias,
                                     position)

    positional_attention = _rel_shift(
        positional_attention, klen=tf.shape(content_attention)[2])

    attention_scores = tf.multiply((content_attention + positional_attention),
                                   1.0 / math.sqrt(float(self._key_dim)))
    attention_scores = self._masked_softmax(attention_scores, attention_mask)
    attention_output = self._dropout_layer(attention_scores)

    attention_output = tf.einsum("bijn,bjnd->bind", attention_output, value)

    return attention_output

  def call(self,
           query,
           value,
           content_attention_bias,
           positional_attention_bias,
           key=None,
           relative_position_encoding=None,
           state=None,
           attention_mask=None):
    """Compute multi-head relative attention over inputs.

    Size glossary:
      * Number of heads (H): the number of attention heads.
      * Value size (V): the size of each value embedding per head.
      * Key size (K): the size of each key embedding per head. Equally, the size
        of each query embedding per head. Typically K <= V.
      * Batch dimensions (B).
      * Query (target) attention axes shape (T).
      * Value (source) attention axes shape (S), the rank must match the target.
      * Encoding length (L): The relative positional encoding length.

    Args:
      query: attention input.
      value: attention input.
      content_attention_bias: A trainable bias parameter added to the query
        head when calculating the content-based attention score.
      positional_attention_bias: A trainable bias parameter added to the query
        head when calculating the position-based attention score.
      key: attention input.
      relative_position_encoding: relative positional encoding for key and
        value.
      state: (default None) optional state. If passed, this is also attended
        over as in TransformerXL.
      attention_mask: (default None) Optional mask that is added to attention
        logits. If state is not None, the mask source sequence dimension should
        extend M.

    Returns:
      attention_output: The result of the computation, of shape [B, T, E],
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are projected to the shape specified by `output_shape`.
    """
    if not self._built_from_signature:
      self._build_from_signature(query, value, key=key)
    if key is None:
      key = value
    if state is not None and state.shape.ndims > 1:
      value = tf.concat([state, value], 1)
      key = tf.concat([state, key], 1)

    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S + M, N, H]
    key = self._key_dense(key)

    # `value` = [B, S + M, N, H]
    value = self._value_dense(value)

    # `position` = [B, L, N, H]
    position = self._encoding_dense(relative_position_encoding)

    attention_output = self.compute_attention(
        query=query,
        key=key,
        value=value,
        position=position,
        content_attention_bias=content_attention_bias,
        positional_attention_bias=positional_attention_bias,
        attention_mask=attention_mask)
    attention_output = self._output_dense(attention_output)

    return attention_output
