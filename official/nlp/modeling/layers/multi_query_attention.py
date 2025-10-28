# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Keras-based attention layers to support multi-query attention.

Based on https://arxiv.org/pdf/1911.02150.pdf and
https://arxiv.org/pdf/2305.13245.pdf.
"""

import math
import string
from typing import Optional, Sequence, Union

import gin
import tensorflow as tf, tf_keras
from official.modeling import tf_utils

_CHR_IDX = string.ascii_lowercase


def _build_proj_equation(
    free_dims: int, bound_dims: int, output_dims: int
) -> ...:
  """Builds an einsum equation for projections inside attention layer.

  Args:
    free_dims: The number of free dimensions which are copied from input to
      output.
    bound_dims: The number of bound dimensions part of input which are combined
      with the kernel to produce output.
    output_dims: The number of output dimensions.

  Returns:
    A tuple of einsum equation, bias axes and output rank.
  """

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
  equation = f"{input_str},{kernel_str}->{output_str}"

  return equation, bias_axes, len(output_str)


def _get_output_shape(
    output_rank: int, known_last_dims: Sequence[int]
) -> list[Optional[int]]:
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


class MultiHeadAttention(tf_keras.layers.MultiHeadAttention):
  """Multi-query attention layer."""

  def __init__(
      self, num_kv_heads=None, enable_gqa_optimization=False, **kwargs
  ):
    # num_kv_heads defines the number of key/value heads. A value of 1 means
    # that the key/value heads are shared across all query heads. Any other
    # value must be less than num_heads and must divide num_heads exactly. If
    # num_kv_heads is greater than 1, query heads are split into groups of
    # num_kv_heads.
    super().__init__(**kwargs)
    self._num_kv_heads = num_kv_heads or self._num_heads
    # TODO(akandoor): Remove this flag once the GQA optimization is rolled out.
    # This flag is used to enable order of K,G in the einsum equations.
    # This optimization is only used in GQA, and is disabled by default.
    # If enabled, the einsum equations are:
    #   1. Dot product: "...SKH,...TKGH->...KGTS"
    #   2. Combine: "...KGTS,...SKH->...TKGH"
    # If disabled, the einsum equations are:
    #   1. Dot product: "...SKH,...TKnH->...nKTS"
    #   2. Combine: "...nKTS,...SKH->...TnKH"
    self._enable_gqa_optimization = enable_gqa_optimization
    assert (
        self._num_kv_heads < self._num_heads
    ), "num_kv_heads must be less than num_heads."
    assert (
        self._num_heads % self._num_kv_heads == 0
    ), "num_kv_heads needs to divide num_heads exactly."

  def get_config(self):
    config = super().get_config()
    config.update({"num_kv_heads": self._num_kv_heads})
    return config

  def _build_from_signature(
      self,
      query: Union[tf.Tensor, tf.TensorShape],
      value: Union[tf.Tensor, tf.TensorShape],
      key: Optional[Union[tf.Tensor, tf.TensorShape]] = None,
  ):
    """Builds layers and variables.

    Once the method is called, self._built_from_signature will be set to
    True.

    Args:
        query: Query tensor or TensorShape.
        value: Value tensor or TensorShape.
        key: Key tensor or TensorShape.
    """
    # pytype: disable=attribute-error
    super()._build_from_signature(query=query, value=value, key=key)
    # pytype: enable=attribute-error

    with tf.init_scope():
      # Key, value are shared across heads in multi-query attention.
      # Overwrite the K, V projections, logits & attend einsum equations to
      # remove the number of attention head dimension in K, V related tensors.
      #
      # The following capital letters are used to denote the tensor dimension
      # parameters:
      # B = batch size
      # S = length of the key/value (source)
      # T = length of the query (target)
      # N = number of query attention heads
      # K = number of key/value heads
      # n = N // K
      # H = dimensions of each attention head.
      #
      if self._num_kv_heads == 1:
        output_dims = 1
        key_last_dims = [self._key_dim]
        value_last_dims = [self._value_dim]
        self._dot_product_equation = "...SH,...TNH->...NTS"
        self._combine_equation = "...NTS,...SH->...TNH"
      else:
        output_dims = 2
        key_last_dims = [self._num_kv_heads, self._key_dim]
        value_last_dims = [self._num_kv_heads, self._value_dim]
        if self._enable_gqa_optimization:
          self._dot_product_equation = "...SKH,...TKGH->...KGTS"
          self._combine_equation = "...KGTS,...SKH->...TKGH"
        else:
          self._dot_product_equation = "...SKH,...TKnH->...nKTS"
          self._combine_equation = "...nKTS,...SKH->...TnKH"

      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          free_dims=self._key_shape.rank - 1,
          bound_dims=1,
          output_dims=output_dims,
      )
      self._key_dense = tf_keras.layers.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1, key_last_dims),
          bias_axes=bias_axes if self._use_bias else None,
          name="key",
          **self._get_common_kwargs_for_sublayer(),
      )
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          free_dims=self._value_shape.rank - 1,
          bound_dims=1,
          output_dims=output_dims,
      )
      self._value_dense = tf_keras.layers.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1, value_last_dims),
          bias_axes=bias_axes if self._use_bias else None,
          name="value",
          **self._get_common_kwargs_for_sublayer(),
      )
      self._qkv_rank = (
          output_rank if self._num_kv_heads > 1 else output_rank + 1
      )

  def _compute_attention(
      self, query, key, value, attention_mask=None, training=None
  ):
    if self._num_kv_heads > 1:
      query = tf.reshape(
          query,
          [
              tf.shape(query)[0],
              tf.shape(query)[1],
              self._num_kv_heads,
              self._num_heads // self._num_kv_heads,
              tf.shape(query)[-1],
          ],
      )

    # pytype: disable=attribute-error
    attention_output, attention_scores = super()._compute_attention(
        query, key, value, attention_mask=attention_mask, training=training
    )
    # pytype: enable=attribute-error
    if self._num_kv_heads != 1:
      attention_output = tf.reshape(
          attention_output,
          [
              tf.shape(attention_output)[0],
              tf.shape(attention_output)[1],
              self._num_heads,
              tf.shape(attention_output)[-1],
          ],
      )
      attention_scores = tf.reshape(
          attention_scores,
          [
              tf.shape(attention_scores)[0],
              self._num_heads,
              tf.shape(attention_scores)[-2],
              tf.shape(attention_scores)[-1],
          ],
      )
    return attention_output, attention_scores


@tf_keras.utils.register_keras_serializable(package="Text")
@gin.configurable
class TalkingHeadsMultiQueryAttention(MultiHeadAttention):
  """Implements Talking-Heads Attention combined with Multi-Query Attention.

  See https://arxiv.org/pdf/2003.02436 for more details.
  TODO(akandoor): Make num talking heads configurable. Currently, num talking
  heads is fixed to num query heads.

  This class inherits from MultiQueryAttention to get the MQA-specific
  logic for __init__, get_config.

  It then overrides _build_from_signature to add the talking-heads weights
  and overrides _compute_attention to merge the MQA wrapper (for
  reshaping) with the THA computation (for pre/post-softmax projections).
  """

  def _build_from_signature(
      self,
      query: Union[tf.Tensor, tf.TensorShape],
      value: Union[tf.Tensor, tf.TensorShape],
      key: Optional[Union[tf.Tensor, tf.TensorShape]] = None,
  ):
    """Builds layers and variables."""
    # Call the parent (MultiQueryAttention) _build_from_signature.
    super()._build_from_signature(query=query, value=value, key=key)
    # Now, *after* all MQA setup is done, we add the THA setup logic.
    qkv_rank = self._qkv_rank
    # TalkingHeadsAttention logic to the MQA build logic.
    num_batch_dims = qkv_rank - len(self._attention_axes) - 2
    attn_scores_rank = num_batch_dims + 1 + len(self._attention_axes) * 2
    scores_notation = _CHR_IDX[:attn_scores_rank]
    projection_notation = scores_notation[num_batch_dims] + (
        _CHR_IDX[attn_scores_rank])
    projected_scores_notation = scores_notation[:num_batch_dims] + (
        _CHR_IDX[attn_scores_rank] + scores_notation[num_batch_dims + 1:])
    self._talking_heads_equation = "%s,%s->%s" % (
        scores_notation, projection_notation, projected_scores_notation)

    with tf.init_scope():
      self._pre_softmax_weight = self.add_weight(
          "pre_softmax_weight",
          shape=(self._num_heads, self._num_heads),
          initializer=tf_utils.clone_initializer(self._kernel_initializer),
          regularizer=self._kernel_regularizer,
          constraint=self._kernel_constraint,
          dtype=self.dtype,
          trainable=True)
      self._post_softmax_weight = self.add_weight(
          "post_softmax_weight",
          shape=(self._num_heads, self._num_heads),
          initializer=tf_utils.clone_initializer(self._kernel_initializer),
          regularizer=self._kernel_regularizer,
          constraint=self._kernel_constraint,
          dtype=self.dtype,
          trainable=True)

  def _compute_attention(
      self, query, key, value, attention_mask=None, training=None
  ):
    """Applies Dot-product attention, merging MQA wrapper and THA computation.

    Args:
      query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
      key: Projected key `Tensor` of shape `[B, T, N, key_dim]`.
      value: Projected value `Tensor` of shape `[B, T, N, value_dim]`.
      attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
        attention to certain positions.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    # This is the MQA "wrapper" logic for grouped queries
    query_shape = tf.shape(query)
    if self._num_kv_heads > 1:
      query = tf.reshape(
          query,
          [
              query_shape[0],
              query_shape[1],
              self._num_kv_heads,
              self._num_heads // self._num_kv_heads,
              query_shape[-1],
          ],
      )

    # This is the THA "computation" logic
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = tf.multiply(
        query, 1.0 / math.sqrt(float(self._key_dim))
    )

    # Note: self._dot_product_equation was set by _build_from_signature
    # (from MQA) to be MQA-compatible.
    attention_scores = tf.einsum(self._dot_product_equation, key, query)

    # --- Talking-Heads modification for MQA ---
    # The THA _talking_heads_equation expects scores of shape [B, N, T, S].
    # The MQA _dot_product_equation produces [B, K, G, T, S].
    # We must reshape before and after applying TH logic.
    scores_shape = tf.shape(attention_scores)
    if self._num_kv_heads > 1:
      # Reshape from [B, K, G, T, S] to [B, N, T, S]
      attention_scores = tf.reshape(
          attention_scores,
          [
              scores_shape[0],  # Batch
              self._num_heads,  # N = K * G
              scores_shape[-2],  # T
              scores_shape[-1]   # S
          ]
      )

    # Apply linear projection before softmax
    attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
                                 self._pre_softmax_weight)

    # Normalize the attention scores to probabilities.
    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # Apply linear projection after softmax
    attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
                                 self._post_softmax_weight)

    # Reshape back to MQA-compatible shape [B, K, G, T, S]
    # before the final combine_equation
    if self._num_kv_heads > 1:
      if self._enable_gqa_optimization:
        attention_scores = tf.reshape(
            attention_scores,
            [
                scores_shape[0],  # B
                self._num_kv_heads,  # K
                self._num_heads // self._num_kv_heads,  # G
                scores_shape[-2],  # T
                scores_shape[-1]   # S
            ]
        )
      else:
        attention_scores = tf.reshape(
            attention_scores,
            [
                scores_shape[0],  # B
                self._num_heads // self._num_kv_heads,  # G
                self._num_kv_heads,  # K
                scores_shape[-2],  # T
                scores_shape[-1]   # S
            ]
        )

    # This is actually dropping out entire tokens to attend to.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # Note: self._combine_equation was set by _build_from_signature
    # (from MQA) to be MQA-compatible.
    attention_output = tf.einsum(self._combine_equation,
                                 attention_scores_dropout, value)

    # This is the MQA "wrapper" logic for grouped queries
    if self._num_kv_heads > 1:
      attention_output = tf.reshape(
          attention_output,
          [
              query_shape[0],
              query_shape[1],
              self._num_heads,
              tf.shape(attention_output)[-1],
          ],
      )
      # We also need to reshape the final scores back to [B, N, T, S]
      # for the return value.
      attention_scores = tf.reshape(
          attention_scores,
          [
              query_shape[0],
              self._num_heads,
              tf.shape(attention_scores)[-2],
              tf.shape(attention_scores)[-1],
          ],
      )

    return attention_output, attention_scores

