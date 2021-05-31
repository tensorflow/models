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

"""Keras-based relative attention layers."""
import math
import string
import tensorflow as tf

_CHR_IDX = string.ascii_lowercase


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


def _rel_shift(x, klen=-1):
  """Performs relative shift to form the relative attention score."""

  x = tf.transpose(x, perm=[2, 3, 0, 1])
  x_size = tf.shape(x)

  x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
  x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

  x = tf.transpose(x, perm=[2, 3, 0, 1])

  return x


@tf.keras.utils.register_keras_serializable(package="Text")
class MultiHeadRelativeAttention(tf.keras.layers.MultiHeadAttention):
  """A multi-head attention layer with relative attention + position encoding.

  This layer shares the same input/output projections as the common
  `tf.keras.layers.MultiHeadAttention` layer.

  When it calculates attention logits, position encoding is projected to form
  relative keys. The logits are composed by shifted relative logits and content
  logits.

  **Note: This layer is currently experimental.

  Attributes:
    kernel_initializer: The kernel initializer. Defaults to variance_scaling.

  Call args:
    query: Query `Tensor` of shape `[B, T, dim]`.
    value: Value `Tensor` of shape `[B, S, dim]`.
    content_attention_bias: Bias `Tensor` for content based attention of shape
      `[num_heads, dim]`.
    positional_attention_bias: Bias `Tensor` for position based attention of
      shape `[num_heads, dim]`.
    key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
      `value` for both `key` and `value`, which is the most common case.
    relative_position_encoding: Relative positional encoding `Tensor` of shape
      `[B, L, dim]`.
    segment_matrix: Optional `Tensor` representing segmentation IDs used in
      XLNet of shape `[B, S, S + M]`.
    segment_encoding: Optional `Tensor` representing the segmentation
      encoding as used in XLNet of shape `[2, num_heads, dim]`.
    segment_attention_bias: Optional trainable bias parameter added to the
      query had when calculating the segment-based attention score used in
      XLNet of shape `[num_heads, dim]`.
    state: Optional `Tensor` of shape `[B, M, E]` where M is the length of the
      state or memory.
      If passed, this is also attended over as in Transformer XL.
    attention_mask: A boolean mask of shape `[B, T, S]` that prevents attention
      to certain positions.
  """

  def __init__(self,
               kernel_initializer="variance_scaling",
               **kwargs):
    super().__init__(kernel_initializer=kernel_initializer,
                     **kwargs)

  def _build_from_signature(self, query, value, key=None):
    super(MultiHeadRelativeAttention, self)._build_from_signature(
        query=query,
        value=value,
        key=key)
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
      einsum_equation, _, output_rank = _build_proj_equation(
          key_shape.rank - 1, bound_dims=1, output_dims=2)
      self._encoding_dense = tf.keras.layers.experimental.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=None,
          name="encoding",
          **common_kwargs)

  def compute_attention(self,
                        query,
                        key,
                        value,
                        position,
                        content_attention_bias,
                        positional_attention_bias,
                        segment_matrix=None,
                        segment_encoding=None,
                        segment_attention_bias=None,
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
      segment_matrix: Optional `Tensor` representing segmentation IDs used in
        XLNet.
      segment_encoding: Optional trainable `Tensor` representing the
        segmentation encoding as used in XLNet.
      segment_attention_bias: Optional trainable bias parameter added to the
        query had when calculating the segment-based attention score used in
        XLNet.
      attention_mask: (default None) Optional mask that is added to attention
        logits. If state is not None, the mask source sequence dimension should
        extend M.

    Returns:
      attention_output: Multi-headed output of attention computation of shape
        `[B, S, N, key_dim]`.

    """
    content_attention = tf.einsum(self._dot_product_equation,
                                  key,
                                  query + content_attention_bias)
    positional_attention = tf.einsum(self._dot_product_equation,
                                     position,
                                     query + positional_attention_bias)
    positional_attention = _rel_shift(
        positional_attention, klen=tf.shape(content_attention)[3])

    if segment_matrix is not None:
      segment_attention = tf.einsum("bind,snd->bnis",
                                    query + segment_attention_bias,
                                    segment_encoding)
      target_shape = tf.shape(positional_attention)
      segment_attention = tf.where(
          tf.broadcast_to(tf.expand_dims(segment_matrix, 1), target_shape),
          tf.broadcast_to(segment_attention[:, :, :, 1:], target_shape),
          tf.broadcast_to(segment_attention[:, :, :, :1], target_shape))
      attention_sum = (
          content_attention + positional_attention + segment_attention)
    else:
      attention_sum = content_attention + positional_attention

    attention_scores = tf.multiply(
        attention_sum, 1.0 / math.sqrt(float(self._key_dim)))

    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    attention_output = self._dropout_layer(attention_scores)

    attention_output = tf.einsum(self._combine_equation,
                                 attention_output,
                                 value)
    return attention_output

  def call(self,
           query,
           value,
           content_attention_bias,
           positional_attention_bias,
           key=None,
           relative_position_encoding=None,
           segment_matrix=None,
           segment_encoding=None,
           segment_attention_bias=None,
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
      segment_matrix: Optional `Tensor` representing segmentation IDs used in
        XLNet.
      segment_encoding: Optional `Tensor` representing the segmentation
        encoding as used in XLNet.
      segment_attention_bias: Optional trainable bias parameter added to the
        query had when calculating the segment-based attention score used in
        XLNet.
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
        segment_matrix=segment_matrix,
        segment_encoding=segment_encoding,
        segment_attention_bias=segment_attention_bias,
        attention_mask=attention_mask)

    # `attention_output` = [B, S, N, H]
    attention_output = self._output_dense(attention_output)

    return attention_output


@tf.keras.utils.register_keras_serializable(package="Text")
class TwoStreamRelativeAttention(MultiHeadRelativeAttention):
  """Two-stream relative self-attention for XLNet.

  In XLNet, each token has two associated vectors at each self-attention layer,
  the content stream (h) and the query stream (g).

  The content stream is the self-attention stream as in Transformer XL and
  represents the context and content (the token itself).

  The query stream only has access to contextual information and the position,
  but not the content.

  This layer shares the same build signature as
  `tf.keras.layers.MultiHeadAttention` but has different input/output
  projections.

  **Note: This layer is currently experimental.

  Call args:
    content_stream: `Tensor` of shape `[B, T, dim]`.
    content_attention_bias: Bias `Tensor` for content based attention of shape
      `[num_heads, dim]`.
    positional_attention_bias: Bias `Tensor` for position based attention of
      shape `[num_heads, dim]`.
    query_stream: `Tensor` of shape `[B, P, dim]`.
    target_mapping: `Tensor` of shape `[B, P, S]`.
    relative_position_encoding: Relative positional encoding `Tensor` of shape
      `[B, L, dim]`.
    segment_matrix: Optional `Tensor` representing segmentation IDs used in
      XLNet of shape `[B, S, S + M]`.
    segment_encoding: Optional `Tensor` representing the segmentation
      encoding as used in XLNet of shape `[2, num_heads, dim]`.
    segment_attention_bias: Optional trainable bias parameter added to the
      query had when calculating the segment-based attention score used in
      XLNet of shape `[num_heads, dim]`.
    state: Optional `Tensor` of shape [B, M, E] where M is the length of the
      state or memory.
      If passed, this is also attended over as in Transformer XL.
    content_attention_mask: a boolean mask of shape `[B, T, S]` that
      prevents attention to certain positions for content attention computation.
    query_attention_mask: a boolean mask of shape `[B, T, S]` that
      prevents attention to certain position for query attention computation.
  """

  def call(self,
           content_stream,
           content_attention_bias,
           positional_attention_bias,
           query_stream,
           relative_position_encoding,
           target_mapping=None,
           segment_matrix=None,
           segment_encoding=None,
           segment_attention_bias=None,
           state=None,
           content_attention_mask=None,
           query_attention_mask=None):
    """Compute multi-head relative attention over inputs.

    Size glossary:
      * Number of heads (H): the number of attention heads.
      * Value size (V): the size of each value embedding per head.
      * Key size (K): the size of each key embedding per head. Equally, the size
        of each query embedding per head. Typically K <= V.
      * Number of predictions (P): the number of predictions.
      * Batch dimensions (B).
      * Query (target) attention axes shape (T).
      * Value (source) attention axes shape (S), the rank must match the target.
      * Encoding length (L): The relative positional encoding length.

    Args:
      content_stream: The content representation, commonly referred to as h.
        This serves a similar role to the standard hidden states in
        Transformer-XL.
      content_attention_bias: A trainable bias parameter added to the query
        head when calculating the content-based attention score.
      positional_attention_bias: A trainable bias parameter added to the query
        head when calculating the position-based attention score.
      query_stream: The query representation, commonly referred to as g.
        This only has access to contextual information and position, but not
        content. If not provided, then this is MultiHeadRelativeAttention with
        self-attention.
      relative_position_encoding: relative positional encoding for key and
        value.
      target_mapping: Optional `Tensor` representing the target mapping used
        in partial prediction.
      segment_matrix: Optional `Tensor` representing segmentation IDs used in
        XLNet.
      segment_encoding: Optional `Tensor` representing the segmentation
        encoding as used in XLNet.
      segment_attention_bias: Optional trainable bias parameter added to the
        query head when calculating the segment-based attention score.
      state: (default None) optional state. If passed, this is also attended
        over as in TransformerXL and XLNet.
      content_attention_mask: (default None) Optional mask that is added to
        content attention logits. If state is not None, the mask source sequence
        dimension should extend M.
      query_attention_mask: (default None) Optional mask that is added to
        query attention logits. If state is not None, the mask source sequence
        dimension should extend M.

    Returns:
      content_attention_output, query_attention_output: the results of the
        computation, both of shape [B, T, E]. `T` is for target sequence shapes,
        `E` is the query input last dimension if `output_shape` is `None`.
        Otherwise, the multi-head outputs are projected to the shape specified
        by `output_shape`.
    """
    if not self._built_from_signature:
      self._build_from_signature(content_stream, content_stream, content_stream)
    if state is not None and state.shape.ndims > 1:
      content_and_memory_stream = tf.concat([state, content_stream], 1)
    else:
      content_and_memory_stream = content_stream

    # `query` = [B, T, N, H]
    query = self._query_dense(content_stream)

    # `key` = [B, S + M, N, H]
    key = self._key_dense(content_and_memory_stream)

    # `value` = [B, S + M, N, H]
    value = self._value_dense(content_and_memory_stream)

    # `position` = [B, L, N, H]
    position = self._encoding_dense(relative_position_encoding)

    content_attention_output = self.compute_attention(
        query=query,
        key=key,
        value=value,
        position=position,
        content_attention_bias=content_attention_bias,
        positional_attention_bias=positional_attention_bias,
        segment_matrix=segment_matrix,
        segment_encoding=segment_encoding,
        segment_attention_bias=segment_attention_bias,
        attention_mask=content_attention_mask)

    # `content_attention_output` = [B, S, N, H]
    content_attention_output = self._output_dense(content_attention_output)

    query_attention_output = None
    if query_stream is not None:
      query = self._query_dense(query_stream)
      if target_mapping is not None:
        query = tf.einsum("bmnd,bml->blnd", query, target_mapping)
        query_attention_output = self.compute_attention(
            query=query,
            key=key,
            value=value,
            position=position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            segment_matrix=segment_matrix,
            segment_encoding=segment_encoding,
            segment_attention_bias=segment_attention_bias,
            attention_mask=query_attention_mask)
        query_attention_output = tf.einsum("blnd,bml->bmnd",
                                           query_attention_output,
                                           target_mapping)
      else:
        query_attention_output = self.compute_attention(
            query=query,
            key=key,
            value=value,
            position=position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            segment_matrix=segment_matrix,
            segment_encoding=segment_encoding,
            segment_attention_bias=segment_attention_bias,
            attention_mask=query_attention_mask)
      query_attention_output = self._output_dense(query_attention_output)

    return content_attention_output, query_attention_output

