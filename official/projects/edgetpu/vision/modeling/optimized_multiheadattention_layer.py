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

"""MultiHeadAttention layer optimized for EdgeTPU.

Compared to tf_keras.layers.MultiHeadAttention, this layer performs query-key
multiplication instead of key-query multiplication to remove an unnecessary
transpose.
"""
import math
import string
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf, tf_keras

_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(
    rank: int, attn_axes: Tuple[int, ...]) -> Tuple[str, str, int]:
  """Builds einsum equations for the attention computation.

  Query, key, value inputs after projection are expected to have the shape as:
  `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
  `bs` and `<non-attention dims>` are treated as `<batch dims>`.

  The attention operations can be generalized:
  (1) Query-key dot product:
  `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
  <key attention dims>, num_heads, channels) -> (<batch dims>,
  num_heads, <query attention dims>, <key attention dims>)`
  (2) Combination:
  `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
  (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
  dims>, <query attention dims>, num_heads, channels)`

  Args:
    rank: Rank of query, key, value tensors.
    attn_axes: List/tuple of axes, `[-1, rank)`, that attention will be
      applied to.

  Returns:
    Einsum equations.
  """
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ""
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (
      target_notation,
      source_notation,
      product_notation,
  )
  attn_scores_rank = len(product_notation)
  combine_equation = "%s,%s->%s" % (
      product_notation,
      source_notation,
      target_notation,
  )
  return dot_product_equation, combine_equation, attn_scores_rank


class OptimizedMultiHeadAttention(tf_keras.layers.MultiHeadAttention):
  """MultiHeadAttention with query-key multiplication.

  Currently, this layer only works for self-attention but not for
  cross-attention. TODO(b/243166060).
  """

  def _build_attention(self, rank: int) -> None:
    """Builds multi-head dot-product attention computations.

    This function builds attributes necessary for `_compute_attention` to
    customize attention computation to replace the default dot-product
    attention.

    Args:
      rank: the rank of query, key, value tensors.
    """
    if self._attention_axes is None:
      self._attention_axes = tuple(range(1, rank - 2))
    else:
      self._attention_axes = tuple(self._attention_axes)
    (
        self._dot_product_equation,
        self._combine_equation,
        attn_scores_rank,
    ) = _build_attention_equation(
        rank, attn_axes=self._attention_axes)
    norm_axes = tuple(
        range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = tf_keras.layers.Softmax(axis=norm_axes)
    self._dropout_layer = tf_keras.layers.Dropout(rate=self._dropout)

  def _compute_attention(
      self,
      query: tf.Tensor,
      key: tf.Tensor,
      value: tf.Tensor,
      attention_mask: Optional[tf.Tensor] = None,
      training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies Dot-product attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for
    customized attention implementation.

    Args:
      query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
      key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
      value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. It is generally not needed if the
        `query` and `value` (and/or `key`) are masked.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum(self._dot_product_equation, query, key)

    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # `context_layer` = [B, T, N, H]
    attention_output = tf.einsum(self._combine_equation,
                                 attention_scores_dropout, value)
    return attention_output, attention_scores
