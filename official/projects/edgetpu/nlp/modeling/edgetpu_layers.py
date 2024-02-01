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

"""Customized MobileBERT-EdgeTPU layers.

There are two reasons for us to customize the layers instead of using the well-
defined layers used in baseline MobileBERT.
1. The layer introduces compiler sharding failures. For example, the gather in
   OnDeviceEmbedding.
2. The layer contains ops that need to have bounded input/output ranges. For
   example, softmax op.
"""
import string

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling import layers

_CHR_IDX = string.ascii_lowercase


# This function is directly copied from the tf_keras.layers.MultiHeadAttention
# implementation.
def _build_attention_equation(rank, attn_axes):
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
  (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
  <query attention dims>, num_heads, channels)`

  Args:
    rank: Rank of query, key, value tensors.
    attn_axes: List/tuple of axes, `[-1, rank)`,
      that attention will be applied to.

  Returns:
    Einsum equations.
  """
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ''
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = ''.join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = '%s,%s->%s' % (source_notation, target_notation,
                                        product_notation)
  attn_scores_rank = len(product_notation)
  combine_equation = '%s,%s->%s' % (product_notation, source_notation,
                                    target_notation)
  return dot_product_equation, combine_equation, attn_scores_rank


@tf_keras.utils.register_keras_serializable(package='Text')
class EdgeTPUSoftmax(tf_keras.layers.Softmax):
  """EdgeTPU/Quantization friendly implementation for the SoftMax.

  When export quant model, use -120 mask value.
  When export float model and run inference with bf16 on device, use -10000.
  """

  def __init__(self,
               mask_value: int = -120,
               **kwargs):
    self._mask_value = mask_value
    super(EdgeTPUSoftmax, self).__init__(**kwargs)

  def get_config(self):
    config = {
        'mask_value': self._mask_value
    }
    base_config = super(EdgeTPUSoftmax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, mask=None):
    if mask is not None:
      adder = (1.0 - tf.cast(mask, inputs.dtype)) * self._mask_value
      inputs += adder
    if isinstance(self.axis, (tuple, list)):
      if len(self.axis) > 1:
        return tf.exp(inputs - tf.reduce_logsumexp(
            inputs, axis=self.axis, keepdims=True))
      else:
        return tf_keras.backend.softmax(inputs, axis=self.axis[0])
    return tf_keras.backend.softmax(inputs, axis=self.axis)


@tf_keras.utils.register_keras_serializable(package='Text')
class EdgeTPUMultiHeadAttention(tf_keras.layers.MultiHeadAttention):
  """Quantization friendly implementation for the MultiHeadAttention."""

  def _build_attention(self, rank):
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
    self._dot_product_equation, self._combine_equation, attn_scores_rank = (
        _build_attention_equation(
            rank, attn_axes=self._attention_axes))
    norm_axes = tuple(
        range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = EdgeTPUSoftmax(axis=norm_axes)
    self._dropout_layer = tf_keras.layers.Dropout(rate=self._dropout)


class EdgetpuMobileBertTransformer(layers.MobileBertTransformer):
  """Quantization friendly MobileBertTransformer.

  Inherits from the MobileBertTransformer but use our customized MHA.
  """

  def __init__(self, **kwargs):
    super(EdgetpuMobileBertTransformer, self).__init__(**kwargs)
    attention_head_size = int(
        self.intra_bottleneck_size / self.num_attention_heads)
    attention_layer = EdgeTPUMultiHeadAttention(
        num_heads=self.num_attention_heads,
        key_dim=attention_head_size,
        value_dim=attention_head_size,
        dropout=self.attention_probs_dropout_prob,
        output_shape=self.intra_bottleneck_size,
        kernel_initializer=self.initializer,
        name='attention')
    layer_norm = self.block_layers['attention'][1]
    self.block_layers['attention'] = [attention_layer, layer_norm]

