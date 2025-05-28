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

import string
from typing import Optional, Sequence, Union

import tensorflow as tf, tf_keras

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

  def __init__(self, num_kv_heads=None, **kwargs):
    # num_kv_heads defines the number of key/value heads. A value of 1 means
    # that the key/value heads are shared across all query heads. Any other
    # value must be less than num_heads and must divide num_heads exactly. If
    # num_kv_heads is greater than 1, query heads are split into groups of
    # num_kv_heads.
    super().__init__(**kwargs)
    self._num_kv_heads = num_kv_heads or self._num_heads
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
