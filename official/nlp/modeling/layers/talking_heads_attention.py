# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Talking Head Attention layer."""
# pylint: disable=g-classes-have-attributes
import math
import string

import gin
import tensorflow as tf

from official.nlp.modeling.layers import attention

_CHR_IDX = string.ascii_lowercase


@tf.keras.utils.register_keras_serializable(package="Text")
@gin.configurable
class TalkingHeadsAttention(attention.MultiHeadAttention):
  """Implements Talking-Heads Attention.

  This is an implementation of Talking-Heads Attention based on the paper
  Talking-Heads Attention (https://arxiv.org/abs/2003.02436): it enhanced
  multi-head attention by including linearprojections across the attention-heads
  dimension, immediately before and after the softmax operation.

  See the base class `MultiHeadAttention` for more details.

  Arguments:
    num_heads: Number of attention heads.
    key_size: Size of each attention head for query and key.
    value_size:  Size of each attention head for value.
    dropout: Dropout probability.
    use_bias: Boolean, whether the dense layers use bias vectors/matrices.
    output_shape: The expected shape of an output tensor, besides the batch and
      sequence dims. If not specified, projects back to the key feature dim.
    attention_axes: axes over which the attention is applied. `None` means
      attention over all axes, but batch, heads, and features.
    return_attention_scores: bool, if `True`, returns the multi-head attention
      scores as an additional output argument.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def build_attention(self, qkv_rank):
    """Builds multi-head dot-product attention computations.

    This function overrides base class to create additional linear projection
    that will be applied on attention scores before and after softmax.

    Args:
      qkv_rank: the rank of query, key, value tensors after projection.
    """
    super(TalkingHeadsAttention, self).build_attention(qkv_rank)

    # Build an equation:
    # (<batch_dims>, num_heads_a, ...),(num_heads_a, num_heads_b) ->
    # (<batch_dims>, num_heads_b, ...)
    # qkv_ranks has `batch_dims`, `attention_dims`, `num_heads` and `channels`.
    num_batch_dims = qkv_rank - len(self._attention_axes) - 2

    # The shape of attn_scores is:
    # (<batch_dims>, num_heads, <query_attn_dims>, <key_attn_dims>)
    attn_scores_rank = num_batch_dims + 1 + len(self._attention_axes) * 2
    scores_notation = _CHR_IDX[:attn_scores_rank]
    projection_notation = scores_notation[num_batch_dims] + (
        _CHR_IDX[attn_scores_rank])
    projected_scores_notation = scores_notation[:num_batch_dims] + (
        _CHR_IDX[attn_scores_rank] + scores_notation[num_batch_dims + 1:])
    self._talking_heads_equation = "%s,%s->%s" % (
        scores_notation, projection_notation, projected_scores_notation)

    self._pre_softmax_weight = self.add_weight(
        "pre_softmax_weight",
        shape=(self._num_heads, self._num_heads),
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        constraint=self._kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self._post_softmax_weight = self.add_weight(
        "post_softmax_weight",
        shape=(self._num_heads, self._num_heads),
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        constraint=self._kernel_constraint,
        dtype=self.dtype,
        trainable=True)

  def compute_attention(self,
                        query_tensor,
                        key_tensor,
                        value_tensor,
                        attention_mask=None):
    """Applies Dot-product attention with query, key, value tensors.

    This function overrides base class to apply additional linear projection
    on attention scores before and after softmax.

    Args:
      query_tensor: Projected query `Tensor` of shape `[B, T, N, key_size]`.
      key_tensor: Projected key `Tensor` of shape `[B, T, N, key_size]`.
      value_tensor: Projected value `Tensor` of shape `[B, T, N, value_size]`.
      attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
        attention to certain positions.

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum(self._dot_product_equation, key_tensor,
                                 query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Apply linear projection before softmax
    attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
                                 self._pre_softmax_weight)

    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, T, S]
    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # Apply linear projection after softmax
    attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
                                 self._post_softmax_weight)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(attention_scores)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum(self._combine_equation,
                                 attention_scores_dropout, value_tensor)
    return attention_output, attention_scores
