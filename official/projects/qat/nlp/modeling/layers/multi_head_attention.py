# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Quantized multi head attention layer."""
import math

import tensorflow as tf, tf_keras

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from official.projects.qat.nlp.quantization import helper


# -6 for mask adder before softmax on int8 model. (e^-6 < 1/256)
_MASK_CONSTANT_FOR_INT8_QUANTIZATION = 6


class MultiHeadAttentionQuantized(helper.LayerQuantizerHelper,
                                  tf_keras.layers.MultiHeadAttention):
  """Quantized multi head attention layer.

   This layer only quantized _compute_attention part. EinsumDense child layers
  should be quantized from the QuantizeConfig.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._compute_attention_first_call = True

  def _build_from_signature(self, *args, **kwargs):
    super()._build_from_signature(  # pytype: disable=attribute-error  # typed-keras
        *args, **kwargs)
    self._add_quantizer('query')
    self._add_quantizer('attention_scores')
    self._add_quantizer('attention_output')

    self._add_quantizer('masked_softmax_attention_mask',
                        all_value_quantizer=True)
    self._add_quantizer('masked_softmax_sub1')
    self._add_quantizer('masked_softmax_mask1')
    self._add_quantizer('masked_softmax_sub2')
    self._add_quantizer('masked_softmax_clamp', all_value_quantizer=True)
    self._add_quantizer('masked_softmax_mask2', all_value_quantizer=True)
    self._add_quantizer('masked_softmax_adder_sub', all_value_quantizer=True)
    self._add_quantizer('masked_softmax_adder_mul', all_value_quantizer=True)
    self._add_quantizer('masked_softmax_add', all_value_quantizer=True)

  def _masked_softmax(
      self, attention_scores, attention_mask=None, training=None):
    """Normalize the attention scores to probabilities."""
    # `attention_scores` = [B, N, T, S]
    if attention_mask is None:
      return self._softmax(attention_scores)

    # The expand dim happens starting from the `num_heads` dimension,
    # (<batch_dims>, num_heads, <query_attention_dims, key_attention_dims>)
    mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
    for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
      attention_mask = array_ops.expand_dims(
          attention_mask, axis=mask_expansion_axes)
    if attention_scores.dtype != attention_mask.dtype:
      attention_mask = tf.cast(attention_mask, attention_scores.dtype)
    attention_mask = self._apply_quantizer(
        'masked_softmax_attention_mask', attention_mask, training)

    # Makes attention_scores >= 0 to avoid masked maximum value be 0.
    attention_scores -= math_ops.reduce_min(
        attention_scores, axis=-1, keepdims=True)
    attention_scores = self._apply_quantizer(
        'masked_softmax_sub1', attention_scores, training)
    attention_scores *= attention_mask
    attention_scores = self._apply_quantizer(
        'masked_softmax_mask1', attention_scores, training)

    # Makes attention_scores <= 0, and become max value be 0.
    attention_scores -= math_ops.reduce_max(
        attention_scores, axis=-1, keepdims=True)
    attention_scores = self._apply_quantizer(
        'masked_softmax_sub2', attention_scores, training)

    # Clip the range of values [-6, 0].
    attention_scores = tf.clip_by_value(
        attention_scores, clip_value_min=-6, clip_value_max=0)
    attention_scores = self._apply_quantizer(
        'masked_softmax_clamp', attention_scores, training)
    # We basically hard-code the to-be-masked-out part have -6.
    # Maximum number is 0. It"s reasonable for 8 bit quantization because
    # e^(0) / e^(-6) < 1/256
    attention_scores *= attention_mask
    attention_scores = self._apply_quantizer(
        'masked_softmax_mask2', attention_scores, training)
    adder = attention_mask - 1.0
    adder = self._apply_quantizer('masked_softmax_adder_sub', adder, training)
    adder *= _MASK_CONSTANT_FOR_INT8_QUANTIZATION
    adder = self._apply_quantizer('masked_softmax_adder_mul', adder, training)
    attention_scores += adder
    attention_scores = self._apply_quantizer(
        'masked_softmax_add', attention_scores, training)

    return self._softmax(attention_scores)

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         attention_mask=None,
                         training=None):
    """Applies Dot-product attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for customized
    attention implementation.

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
    if self._compute_attention_first_call:
      self._build_quantizer_vars()
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

    query = self._apply_quantizer('query', query, training)
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = special_math_ops.einsum(self._dot_product_equation, key,
                                               query)
    attention_scores = self._apply_quantizer(
        'attention_scores', attention_scores, training)

    attention_scores = self._masked_softmax(
        attention_scores, attention_mask, training)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # `context_layer` = [B, T, N, H]
    attention_output = special_math_ops.einsum(self._combine_equation,
                                               attention_scores_dropout, value)
    attention_output = self._apply_quantizer(
        'attention_output', attention_output, training)

    self._compute_attention_first_call = False
    return attention_output, attention_scores
