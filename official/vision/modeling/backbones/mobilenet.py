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

"""Contains definitions of MobileNet Networks."""

import dataclasses
from typing import Any

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.backbones import factory
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers

layers = tf_keras.layers

#  pylint: disable=pointless-string-statement


@dataclasses.dataclass
class BlockSpec(hyperparams.Config):
  """A container class that specifies the block configuration for MobileNet.

  Attributes:
    block_fn: Block function name.
    kernel_size: One side of a 2d kernel size (if relevant).
    strides: One side of a 2d stride (if relevant).
    filters: Number of output filters in a kernel (if relevant).
    use_bias: If True, include a bias term in relevant blocks.
    use_normalization: If True, include a normalization term in relevant blocks.
    activation: Block activation function name.
    expand_ratio: Factor to multiply incoming filter size by in
      InvertBottlenecks and related blocks.
    se_ratio: Filter multiplication factor for use in squeeze and excitation.
    use_depthwise: If True, create an inverted bottleneck structure with 1x1
      conv2ds between the filters and kxk depthwise convs across the spatial
      extent. If False, create a fused inverted bottleneck structure with a kxk
      conv2d followed by a 1x1 conv2d.
    use_residual: If True, create a residual connection which adds the input
      filters to the output filters of this block.
    is_output: If True, add the output filters from this block to the model
      endpoints.
    middle_dw_downsample: True if the middle depthwise op should be the strided
      operation instead of the first depthwise op in
      UniversalInvertedBottleneckBlocks with strides > 1.
    start_dw_kernel_size: One side of the 2d kernel size in the first depthwise
      op in a UniversalInvertedBottleneckBlock.
    middle_dw_kernel_size: One side of the 2d kernel size in the second
      depthwise op in a UniversalInvertedBottleneckBlock.
    end_dw_kernel_size: One side of the 2d kernel size in the second depthwise
      op in a UniversalInvertedBottleneckBlock.
    use_layer_scale: True if layer scale should be included in a
      UniversalInvertedBottleneckBlock or a MultiHeadSelfAttentionBlock.
    use_multi_query: True if Multi Query Attention should be used in a
      MultiHeadSelfAttentionBlock.
    use_downsampling: bool = False
    downsampling_dw_kernel_size: int = 3
    num_heads: Number of attention heads to use in a
      MultiHeadSelfAttentionBlock.
    key_dim: Size of the key dimension used in a MultiHeadSelfAttentionBlock.
    value_dim: Size of the value dimension used in a
      MultiHeadSelfAttentionBlock.
    query_h_strides: The size of the vertical stride used to compute the query
      when use_multi_query is True in a MultiHeadSelfAttentionBlock.
    query_w_strides: The size of the horizontal stride used to compute the query
      when use_multi_query is True in a MultiHeadSelfAttentionBlock.
    kv_strides: One size of the 2d stride used to compute the key and value when
      use_multi_query is True in a MultiHeadSelfAttentionBlock.
  """

  block_fn: str = 'convbn'
  kernel_size: int = 3
  strides: int = 1
  filters: int = 32
  use_bias: bool = False
  use_normalization: bool = True
  activation: str = 'relu6'
  # Used for block type InvertedResConv.
  expand_ratio: float | None = 6.0
  # Used for block type InvertedResConv with SE.
  se_ratio: float | None = None
  use_depthwise: bool = True
  use_residual: bool = True
  is_output: bool = True

  # Parameters for a UniversalInvertedBottleneckBlock block.
  middle_dw_downsample: bool = True
  start_dw_kernel_size: int = 0
  middle_dw_kernel_size: int = 0
  end_dw_kernel_size: int = 0

  # layer scale currently only supports uib and mhsa.
  use_layer_scale: bool = False

  # Fields only relevant to mhsa blocks.
  use_multi_query: bool = False
  use_downsampling: bool = False
  downsampling_dw_kernel_size: int = 3
  num_heads: int = 8
  key_dim: int = 64
  value_dim: int = 64
  query_h_strides: int = 1
  query_w_strides: int = 1
  kv_strides: int = 1


def block_spec_field_list() -> list[str]:
  """Returns the list of field names used in `BlockSpec`."""
  return [field.name for field in dataclasses.fields(BlockSpec)]


def block_spec_values_to_list(
    block_specs: list[BlockSpec],
) -> list[tuple[Any, ...]]:
  """Creates a list field value tuples from a list of `BlockSpec`s."""
  return [dataclasses.astuple(bs) for bs in block_specs]


@tf_keras.utils.register_keras_serializable(package='Vision')
class Conv2DBNBlock(tf_keras.layers.Layer):
  """A convolution block with batch normalization."""

  def __init__(
      self,
      filters: int,
      kernel_size: int = 3,
      strides: int = 1,
      use_bias: bool = False,
      use_explicit_padding: bool = False,
      activation: str = 'relu6',
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: tf_keras.regularizers.Regularizer | None = None,
      bias_regularizer: tf_keras.regularizers.Regularizer | None = None,
      use_normalization: bool = True,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs,
  ):
    """A convolution block with batch normalization.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      kernel_size: An `int` specifying the height and width of the 2D
        convolution window.
      strides: An `int` of block stride. If greater than 1, this block will
        ultimately downsample the input.
      use_bias: If True, use bias in the convolution layer.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      activation: A `str` name of the activation function.
      kernel_initializer: A `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      use_normalization: If True, use batch normalization.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(Conv2DBNBlock, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._use_bias = use_bias
    self._use_explicit_padding = use_explicit_padding
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_normalization = use_normalization
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._norm = tf_keras.layers.BatchNormalization

    if use_explicit_padding and kernel_size > 1:
      self._padding = 'valid'
    else:
      self._padding = 'same'
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'use_bias': self._use_bias,
        'use_explicit_padding': self._use_explicit_padding,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'use_normalization': self._use_normalization,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(Conv2DBNBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    if self._use_explicit_padding and self._kernel_size > 1:
      padding_size = nn_layers.get_padding_for_kernel_size(self._kernel_size)
      self._pad = tf_keras.layers.ZeroPadding2D(padding_size)
    self._conv0 = tf_keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding=self._padding,
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    if self._use_normalization:
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          synchronized=self._use_sync_bn)
    self._activation_layer = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    super(Conv2DBNBlock, self).build(input_shape)

  def call(self, inputs, training=None):
    if self._use_explicit_padding and self._kernel_size > 1:
      inputs = self._pad(inputs)
    x = self._conv0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    return self._activation_layer(x)

"""
Architecture: https://arxiv.org/abs/1704.04861.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications" Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
"""
MNV1_BLOCK_SPECS = {
    'spec_name': 'MobileNetV1',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides',
                          'filters', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 32, False),
        ('depsepconv', 3, 1, 64, False),
        ('depsepconv', 3, 2, 128, False),
        ('depsepconv', 3, 1, 128, True),
        ('depsepconv', 3, 2, 256, False),
        ('depsepconv', 3, 1, 256, True),
        ('depsepconv', 3, 2, 512, False),
        ('depsepconv', 3, 1, 512, False),
        ('depsepconv', 3, 1, 512, False),
        ('depsepconv', 3, 1, 512, False),
        ('depsepconv', 3, 1, 512, False),
        ('depsepconv', 3, 1, 512, True),
        ('depsepconv', 3, 2, 1024, False),
        ('depsepconv', 3, 1, 1024, True),
    ]
}

"""
Architecture: https://arxiv.org/abs/1801.04381

"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
"""
MNV2_BLOCK_SPECS = {
    'spec_name': 'MobileNetV2',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'expand_ratio', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 32, None, False),
        ('invertedbottleneck', 3, 1, 16, 1., False),
        ('invertedbottleneck', 3, 2, 24, 6., False),
        ('invertedbottleneck', 3, 1, 24, 6., True),
        ('invertedbottleneck', 3, 2, 32, 6., False),
        ('invertedbottleneck', 3, 1, 32, 6., False),
        ('invertedbottleneck', 3, 1, 32, 6., True),
        ('invertedbottleneck', 3, 2, 64, 6., False),
        ('invertedbottleneck', 3, 1, 64, 6., False),
        ('invertedbottleneck', 3, 1, 64, 6., False),
        ('invertedbottleneck', 3, 1, 64, 6., False),
        ('invertedbottleneck', 3, 1, 96, 6., False),
        ('invertedbottleneck', 3, 1, 96, 6., False),
        ('invertedbottleneck', 3, 1, 96, 6., True),
        ('invertedbottleneck', 3, 2, 160, 6., False),
        ('invertedbottleneck', 3, 1, 160, 6., False),
        ('invertedbottleneck', 3, 1, 160, 6., False),
        ('invertedbottleneck', 3, 1, 320, 6., True),
        ('convbn', 1, 1, 1280, None, False),
    ]
}

"""
Architecture: https://arxiv.org/abs/1905.02244

"Searching for MobileNetV3"
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan,
Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
"""
MNV3Large_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Large',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 16,
         'hard_swish', None, None, True, False, False),
        ('invertedbottleneck', 3, 1, 16,
         'relu', None, 1., None, False, False),
        ('invertedbottleneck', 3, 2, 24,
         'relu', None, 4., None, False, False),
        ('invertedbottleneck', 3, 1, 24,
         'relu', None, 3., None, False, True),
        ('invertedbottleneck', 5, 2, 40,
         'relu', 0.25, 3., None, False, False),
        ('invertedbottleneck', 5, 1, 40,
         'relu', 0.25, 3., None, False, False),
        ('invertedbottleneck', 5, 1, 40,
         'relu', 0.25, 3., None, False, True),
        ('invertedbottleneck', 3, 2, 80,
         'hard_swish', None, 6., None, False, False),
        ('invertedbottleneck', 3, 1, 80,
         'hard_swish', None, 2.5, None, False, False),
        ('invertedbottleneck', 3, 1, 80,
         'hard_swish', None, 2.3, None, False, False),
        ('invertedbottleneck', 3, 1, 80,
         'hard_swish', None, 2.3, None, False, False),
        ('invertedbottleneck', 3, 1, 112,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 3, 1, 112,
         'hard_swish', 0.25, 6., None, False, True),
        ('invertedbottleneck', 5, 2, 160,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 5, 1, 160,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 5, 1, 160,
         'hard_swish', 0.25, 6., None, False, True),
        ('convbn', 1, 1, 960,
         'hard_swish', None, None, True, False, False),
        ('gpooling', None, None, None,
         None, None, None, None, None, False),
        ('convbn', 1, 1, 1280,
         'hard_swish', None, None, False, True, False),
    ]
}

MNV3Small_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Small',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 16,
         'hard_swish', None, None, True, False, False),
        ('invertedbottleneck', 3, 2, 16,
         'relu', 0.25, 1, None, False, True),
        ('invertedbottleneck', 3, 2, 24,
         'relu', None, 72. / 16, None, False, False),
        ('invertedbottleneck', 3, 1, 24,
         'relu', None, 88. / 24, None, False, True),
        ('invertedbottleneck', 5, 2, 40,
         'hard_swish', 0.25, 4., None, False, False),
        ('invertedbottleneck', 5, 1, 40,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 5, 1, 40,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 5, 1, 48,
         'hard_swish', 0.25, 3., None, False, False),
        ('invertedbottleneck', 5, 1, 48,
         'hard_swish', 0.25, 3., None, False, True),
        ('invertedbottleneck', 5, 2, 96,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 5, 1, 96,
         'hard_swish', 0.25, 6., None, False, False),
        ('invertedbottleneck', 5, 1, 96,
         'hard_swish', 0.25, 6., None, False, True),
        ('convbn', 1, 1, 576,
         'hard_swish', None, None, True, False, False),
        ('gpooling', None, None, None,
         None, None, None, None, None, False),
        ('convbn', 1, 1, 1024,
         'hard_swish', None, None, False, True, False),
    ]
}

"""
The EdgeTPU version is taken from
github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
"""
MNV3EdgeTPU_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3EdgeTPU',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_residual', 'use_depthwise', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, None, None, None, False),
        ('invertedbottleneck', 3, 1, 16, 'relu', None, 1., True, False, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', None, 8., True, False, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', None, 4., True, False, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', None, 4., True, False, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', None, 4., True, False, True),
        ('invertedbottleneck', 3, 2, 48, 'relu', None, 8., True, False, False),
        ('invertedbottleneck', 3, 1, 48, 'relu', None, 4., True, False, False),
        ('invertedbottleneck', 3, 1, 48, 'relu', None, 4., True, False, False),
        ('invertedbottleneck', 3, 1, 48, 'relu', None, 4., True, False, True),
        ('invertedbottleneck', 3, 2, 96, 'relu', None, 8., True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 8., False, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True, True),
        ('invertedbottleneck', 5, 2, 160, 'relu', None, 8., True, True, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', None, 4., True, True, False),
        ('invertedbottleneck', 3, 1, 192, 'relu', None, 8., True, True, True),
        ('convbn', 1, 1, 1280, 'relu', None, None, None, None, False),
    ]
}

"""
Architecture: https://arxiv.org/pdf/2008.08178.pdf

"Discovering Multi-Hardware Mobile Models via Architecture Search"
Grace Chu, Okan Arikan, Gabriel Bender, Weijun Wang,
Achille Brighton, Pieter-Jan Kindermans, Hanxiao Liu,
Berkin Akin, Suyog Gupta, and Andrew Howard
"""
MNMultiMAX_BLOCK_SPECS = {
    'spec_name': 'MobileNetMultiMAX',
    'block_spec_schema': [
        'block_fn', 'kernel_size', 'strides', 'filters', 'activation',
        'expand_ratio', 'use_normalization', 'use_bias', 'is_output'
    ],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, True, False, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', 3., None, False, True),
        ('invertedbottleneck', 5, 2, 64, 'relu', 6., None, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., None, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., None, False, True),
        ('invertedbottleneck', 5, 2, 128, 'relu', 6., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 4., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 6., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False, True),
        ('invertedbottleneck', 3, 2, 160, 'relu', 6., None, False, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', 4., None, False, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 5., None, False, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', 4., None, False, True),
        ('convbn', 1, 1, 960, 'relu', None, True, False, False),
        ('gpooling', None, None, None, None, None, None, None, False),
        # Remove bias and add batch norm for the last layer to support QAT
        # and achieve slightly better accuracy.
        ('convbn', 1, 1, 1280, 'relu', None, True, False, False),
    ]
}

MNMultiAVG_BLOCK_SPECS = {
    'spec_name': 'MobileNetMultiAVG',
    'block_spec_schema': [
        'block_fn', 'kernel_size', 'strides', 'filters', 'activation',
        'expand_ratio', 'use_normalization', 'use_bias', 'is_output'
    ],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, True, False, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', 2., None, False, True),
        ('invertedbottleneck', 5, 2, 64, 'relu', 5., None, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., None, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 3., None, False, True),
        ('invertedbottleneck', 5, 2, 128, 'relu', 6., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 6., None, False, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 4., None, False, True),
        ('invertedbottleneck', 3, 2, 192, 'relu', 6., None, False, False),
        ('invertedbottleneck', 5, 1, 192, 'relu', 4., None, False, False),
        ('invertedbottleneck', 5, 1, 192, 'relu', 4., None, False, False),
        ('invertedbottleneck', 5, 1, 192, 'relu', 4., None, False, True),
        ('convbn', 1, 1, 960, 'relu', None, True, False, False),
        ('gpooling', None, None, None, None, None, None, None, False),
        # Remove bias and add batch norm for the last layer to support QAT
        # and achieve slightly better accuracy.
        ('convbn', 1, 1, 1280, 'relu', None, True, False, False),
    ]
}

# Similar to MobileNetMultiAVG and used for segmentation task.
# Reduced the filters by a factor of 2 in the last block.
MNMultiAVG_SEG_BLOCK_SPECS = {
    'spec_name':
        'MobileNetMultiAVGSeg',
    'block_spec_schema': [
        'block_fn', 'kernel_size', 'strides', 'filters', 'activation',
        'expand_ratio', 'use_normalization', 'use_bias', 'is_output'
    ],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, True, False, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', 2., True, False, True),
        ('invertedbottleneck', 5, 2, 64, 'relu', 5., True, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., True, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 3., True, False, True),
        ('invertedbottleneck', 5, 2, 128, 'relu', 6., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 6., True, False, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 4., True, False, True),
        ('invertedbottleneck', 3, 2, 192, 'relu', 6., True, False, False),
        ('invertedbottleneck', 5, 1, 96, 'relu', 2., True, False, False),
        ('invertedbottleneck', 5, 1, 96, 'relu', 4., True, False, False),
        ('invertedbottleneck', 5, 1, 96, 'relu', 4., True, False, True),
        ('convbn', 1, 1, 448, 'relu', None, True, False, True),
        ('gpooling', None, None, None, None, None, None, None, False),
        # Remove bias and add batch norm for the last layer to support QAT
        # and achieve slightly better accuracy.
        ('convbn', 1, 1, 1280, 'relu', None, True, False, False),
    ]
}

# Similar to MobileNetMultiMax and used for segmentation task.
# Reduced the filters by a factor of 2 in the last block.
MNMultiMAX_SEG_BLOCK_SPECS = {
    'spec_name':
        'MobileNetMultiMAXSeg',
    'block_spec_schema': [
        'block_fn', 'kernel_size', 'strides', 'filters', 'activation',
        'expand_ratio', 'use_normalization', 'use_bias', 'is_output'
    ],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, True, False, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', 3., True, False, True),
        ('invertedbottleneck', 5, 2, 64, 'relu', 6., True, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., True, False, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., True, False, True),
        ('invertedbottleneck', 5, 2, 128, 'relu', 6., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 4., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 6., True, False, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., True, False, True),
        ('invertedbottleneck', 3, 2, 160, 'relu', 6., True, False, False),
        ('invertedbottleneck', 5, 1, 96, 'relu', 2., True, False, False),
        ('invertedbottleneck', 3, 1, 96, 'relu', 4., True, False, False),
        ('invertedbottleneck', 5, 1, 96, 'relu', 320.0 / 96, True, False, True),
        ('convbn', 1, 1, 448, 'relu', None, True, False, True),
        ('gpooling', None, None, None, None, None, None, None, False),
        # Remove bias and add batch norm for the last layer to support QAT
        # and achieve slightly better accuracy.
        ('convbn', 1, 1, 1280, 'relu', None, True, False, False),
    ]
}

# A smaller MNV3Small, with reduced filters for the last few layers
MNV3SmallReducedFilters = {
    'spec_name':
        'MobilenetV3SmallReducedFilters',
    'block_spec_schema': [
        'block_fn', 'kernel_size', 'strides', 'filters', 'activation',
        'se_ratio', 'expand_ratio', 'use_normalization', 'use_bias', 'is_output'
    ],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, True, False, False),
        ('invertedbottleneck', 3, 2, 16, 'relu', 0.25, 1, None, False, True),
        ('invertedbottleneck', 3, 2, 24, 'relu', None, 72. / 16, None, False,
         False),
        ('invertedbottleneck', 3, 1, 24, 'relu', None, 88. / 24, None, False,
         True),
        ('invertedbottleneck', 5, 2, 40, 'hard_swish', 0.25, 4, None, False,
         False),
        ('invertedbottleneck', 5, 1, 40, 'hard_swish', 0.25, 6, None, False,
         False),
        ('invertedbottleneck', 5, 1, 40, 'hard_swish', 0.25, 6, None, False,
         False),
        ('invertedbottleneck', 5, 1, 48, 'hard_swish', 0.25, 3, None, False,
         False),
        ('invertedbottleneck', 5, 1, 48, 'hard_swish', 0.25, 3, None, False,
         True),
        # Layers below are different from MobileNetV3Small and have
        # half as many filters
        ('invertedbottleneck', 5, 2, 48, 'hard_swish', 0.25, 3, None, False,
         False),
        ('invertedbottleneck', 5, 1, 48, 'hard_swish', 0.25, 6, None, False,
         False),
        ('invertedbottleneck', 5, 1, 48, 'hard_swish', 0.25, 6, None, False,
         True),
        ('convbn', 1, 1, 288, 'hard_swish', None, None, True, False, False),
        ('gpooling', None, None, None, None, None, None, None, None, False),
        ('convbn', 1, 1, 1024, 'hard_swish', None, None, False, True, False),
    ]
}


"""
Architecture: https://arxiv.org/abs/2404.10518

"MobileNetV4 - Universal Models for the Mobile Ecosystem"
Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan
Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal,
Tenghui Zhu, Daniele Moro, Andrew Howard
"""
MNV4ConvSmall_BLOCK_SPECS = {
    'spec_name': 'MobileNetV4ConvSmall',
    'block_spec_schema': [
        'block_fn',
        'activation',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'strides',
        'filters',
        'expand_ratio',
        'is_output',
    ],
    'block_specs': [
        # 112px after stride 2.
        ('convbn', 'relu', 3, None, None, False, 2, 32, None, False),
        # 56px.
        ('convbn', 'relu', 3, None, None, False, 2, 32, None, False),
        ('convbn', 'relu', 1, None, None, False, 1, 32, None, True),
        # 28px.
        ('convbn', 'relu', 3, None, None, False, 2, 96, None, False),
        ('convbn', 'relu', 1, None, None, False, 1, 64, None, True),
        # 14px.
        ('uib', 'relu', None, 5, 5, True, 2, 96, 3.0, False),  # ExtraDW
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 96, 2.0, False),  # IB
        ('uib', 'relu', None, 3, 0, True, 1, 96, 4.0, True),  # ConvNext
        # 7px
        ('uib', 'relu', None, 3, 3, True, 2, 128, 6.0, False),  # ExtraDW
        ('uib', 'relu', None, 5, 5, True, 1, 128, 4.0, False),  # ExtraDW
        ('uib', 'relu', None, 0, 5, True, 1, 128, 4.0, False),  # IB
        ('uib', 'relu', None, 0, 5, True, 1, 128, 3.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 128, 4.0, False),  # IB
        ('uib', 'relu', None, 0, 3, True, 1, 128, 4.0, True),  # IB
        ('convbn', 'relu', 1, None, None, False, 1, 960, None, False),  # Conv
        (
            'gpooling',
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        ),  # Avg
        ('convbn', 'relu', 1, None, None, False, 1, 1280, None, False),  # Conv
    ],
}


def _mnv4_conv_medium_block_specs():
  """Medium-sized MobileNetV4 using only convolutional operations."""

  def convbn(kernel_size, strides, filters):
    return BlockSpec(
        block_fn='convbn',
        activation='relu',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        is_output=False,
    )

  def fused_ib(kernel_size, strides, filters, output=False):
    return BlockSpec(
        block_fn='fused_ib',
        activation='relu',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        expand_ratio=4.0,
        is_output=output,
    )

  def uib(
      start_dw_ks, middle_dw_ks, strides, filters, expand_ratio, output=False
  ):
    return BlockSpec(
        block_fn='uib',
        activation='relu',
        start_dw_kernel_size=start_dw_ks,
        middle_dw_kernel_size=middle_dw_ks,
        filters=filters,
        strides=strides,
        expand_ratio=expand_ratio,
        use_layer_scale=False,
        is_output=output,
    )

  blocks = [
      convbn(3, 2, 32),
      fused_ib(3, 2, 48, output=True),
      # 3rd stage
      uib(3, 5, 2, 80, 4.0),
      uib(3, 3, 1, 80, 2.0, output=True),
      # 4th stage
      uib(3, 5, 2, 160, 6.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 5, 1, 160, 4.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 0, 1, 160, 4.0),
      uib(0, 0, 1, 160, 2.0),
      uib(3, 0, 1, 160, 4.0, output=True),
      # 5th stage
      uib(5, 5, 2, 256, 6.0),
      uib(5, 5, 1, 256, 4.0),
      uib(3, 5, 1, 256, 4.0),
      uib(3, 5, 1, 256, 4.0),
      uib(0, 0, 1, 256, 4.0),
      uib(3, 0, 1, 256, 4.0),
      uib(3, 5, 1, 256, 2.0),
      uib(5, 5, 1, 256, 4.0),
      uib(0, 0, 1, 256, 4.0),
      uib(0, 0, 1, 256, 4.0),
      uib(5, 0, 1, 256, 2.0, output=True),
      # FC layers
      convbn(1, 1, 960),
      BlockSpec(block_fn='gpooling', is_output=False),
      convbn(1, 1, 1280),
  ]
  return {
      'spec_name': 'MobileNetV4ConvMedium',
      'block_spec_schema': block_spec_field_list(),
      'block_specs': block_spec_values_to_list(blocks),
  }


def _mnv4_conv_medium_seg_block_specs():
  """Tailored MobileNetV4ConvMedium for dense prediction, e.g. segmentation."""

  def convbn(kernel_size, strides, filters, output=False):
    return BlockSpec(
        block_fn='convbn',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        is_output=output,
    )

  def fused_ib(kernel_size, strides, filters, output=False):
    return BlockSpec(
        block_fn='fused_ib',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        expand_ratio=4.0,
        is_output=output,
    )

  def uib(
      start_dw_ks, middle_dw_ks, strides, filters, expand_ratio, output=False
  ):
    return BlockSpec(
        block_fn='uib',
        start_dw_kernel_size=start_dw_ks,
        middle_dw_kernel_size=middle_dw_ks,
        filters=filters,
        strides=strides,
        expand_ratio=expand_ratio,
        use_layer_scale=False,
        is_output=output,
    )

  blocks = [
      convbn(3, 2, 32),
      fused_ib(3, 2, 48, output=True),
      # 3rd stage
      uib(3, 5, 2, 80, 4.0),
      uib(3, 3, 1, 80, 2.0, output=True),
      # 4th stage
      uib(3, 5, 2, 160, 6.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 5, 1, 160, 4.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 0, 1, 160, 4.0),
      uib(3, 0, 1, 160, 4.0, output=True),
      # 5th stage
      uib(5, 5, 2, 256, 6.0),
      uib(5, 5, 1, 128, 4.0),
      uib(3, 5, 1, 128, 4.0),
      uib(3, 5, 1, 128, 4.0),
      uib(3, 0, 1, 128, 4.0),
      uib(3, 5, 1, 128, 2.0),
      uib(5, 5, 1, 128, 4.0),
      uib(5, 0, 1, 128, 2.0, output=False),
      # FC layers
      convbn(1, 1, 448, output=True),
      BlockSpec(block_fn='gpooling', is_output=False),
      convbn(1, 1, 1280),
  ]
  return {
      'spec_name': 'MobileNetV4ConvMediumSeg',
      'block_spec_schema': block_spec_field_list(),
      'block_specs': block_spec_values_to_list(blocks),
  }


MNV4ConvLarge_BLOCK_SPECS = {
    'spec_name': 'MobileNetV4ConvLarge',
    'block_spec_schema': [
        'block_fn',
        'activation',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'strides',
        'filters',
        'expand_ratio',
        'is_output',
    ],
    'block_specs': [
        ('convbn', 'relu', 3, None, None, False, 2, 24, None, False),
        ('fused_ib', 'relu', 3, None, None, False, 2, 48, 4.0, True),
        ('uib', 'relu', None, 3, 5, True, 2, 96, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 96, 4.0, True),
        ('uib', 'relu', None, 3, 5, True, 2, 192, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 5, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 192, 4.0, False),
        ('uib', 'relu', None, 3, 0, True, 1, 192, 4.0, True),
        ('uib', 'relu', None, 5, 5, True, 2, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 3, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 5, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, False),
        ('uib', 'relu', None, 5, 0, True, 1, 512, 4.0, True),
        ('convbn', 'relu', 1, None, None, False, 1, 960, None, False),
        ('gpooling', None, None, None, None, None, None, None, None, False),
        ('convbn', 'relu', 1, None, None, False, 1, 1280, None, False),
    ],
}


def _mnv4_hybrid_medium_block_specs():
  """Medium-sized MobileNetV4 using only attention and convolutional operations."""

  def convbn(kernel_size, strides, filters):
    return BlockSpec(
        block_fn='convbn',
        activation='relu',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        is_output=False,
    )

  def fused_ib(kernel_size, strides, filters, output=False):
    return BlockSpec(
        block_fn='fused_ib',
        activation='relu',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        expand_ratio=4.0,
        is_output=output,
    )

  def uib(
      start_dw_ks, middle_dw_ks, strides, filters, expand_ratio, output=False
  ):
    return BlockSpec(
        block_fn='uib',
        activation='relu',
        start_dw_kernel_size=start_dw_ks,
        middle_dw_kernel_size=middle_dw_ks,
        filters=filters,
        strides=strides,
        expand_ratio=expand_ratio,
        use_layer_scale=True,
        is_output=output,
    )

  def mhsa_24px():
    return BlockSpec(
        block_fn='mhsa',
        activation='relu',
        filters=160,
        key_dim=64,
        value_dim=64,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=2,
        num_heads=4,
        use_layer_scale=True,
        use_multi_query=True,
        is_output=False,
    )

  def mhsa_12px():
    return BlockSpec(
        block_fn='mhsa',
        activation='relu',
        filters=256,
        key_dim=64,
        value_dim=64,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=1,
        num_heads=4,
        use_layer_scale=True,
        use_multi_query=True,
        is_output=False,
    )

  blocks = [
      convbn(3, 2, 32),
      fused_ib(3, 2, 48, output=True),
      # 3rd stage
      uib(3, 5, 2, 80, 4.0),
      uib(3, 3, 1, 80, 2.0, output=True),
      # 4th stage
      uib(3, 5, 2, 160, 6.0),
      uib(0, 0, 1, 160, 2.0),
      uib(3, 3, 1, 160, 4.0),
      uib(3, 5, 1, 160, 4.0),
      mhsa_24px(),
      uib(3, 3, 1, 160, 4.0),
      mhsa_24px(),
      uib(3, 0, 1, 160, 4.0),
      mhsa_24px(),
      uib(3, 3, 1, 160, 4.0),
      mhsa_24px(),
      uib(3, 0, 1, 160, 4.0, output=True),
      # 5th stage
      uib(5, 5, 2, 256, 6.0),
      uib(5, 5, 1, 256, 4.0),
      uib(3, 5, 1, 256, 4.0),
      uib(3, 5, 1, 256, 4.0),
      uib(0, 0, 1, 256, 2.0),
      uib(3, 5, 1, 256, 2.0),
      uib(0, 0, 1, 256, 2.0),
      uib(0, 0, 1, 256, 4.0),
      mhsa_12px(),
      uib(3, 0, 1, 256, 4.0),
      mhsa_12px(),
      uib(5, 5, 1, 256, 4.0),
      mhsa_12px(),
      uib(5, 0, 1, 256, 4.0),
      mhsa_12px(),
      uib(5, 0, 1, 256, 4.0, output=True),
      convbn(1, 1, 960),
      BlockSpec(block_fn='gpooling', is_output=False),
      convbn(1, 1, 1280),
  ]
  return {
      'spec_name': 'MobileNetV4HybridMedium',
      'block_spec_schema': block_spec_field_list(),
      'block_specs': block_spec_values_to_list(blocks),
  }


def _mnv4_hybrid_large_block_specs():
  """Large-sized MobileNetV4 using only attention and convolutional operations."""

  def convbn(kernel_size, strides, filters):
    return BlockSpec(
        block_fn='convbn',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        activation='gelu',
        is_output=False,
    )

  def fused_ib(kernel_size, strides, filters, output=False):
    return BlockSpec(
        block_fn='fused_ib',
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        expand_ratio=4.0,
        is_output=output,
        activation='gelu',
    )

  def uib(
      start_dw_ks,
      middle_dw_ks,
      strides,
      filters,
      expand_ratio=4.0,
      output=False,
  ):
    return BlockSpec(
        block_fn='uib',
        start_dw_kernel_size=start_dw_ks,
        middle_dw_kernel_size=middle_dw_ks,
        filters=filters,
        strides=strides,
        expand_ratio=expand_ratio,
        use_layer_scale=True,
        is_output=output,
        activation='gelu',
    )

  def mhsa_24px():
    return BlockSpec(
        block_fn='mhsa',
        activation='relu',
        filters=192,
        key_dim=48,
        value_dim=48,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=2,
        num_heads=8,
        use_layer_scale=True,
        use_multi_query=True,
        is_output=False,
    )

  def mhsa_12px():
    return BlockSpec(
        block_fn='mhsa',
        activation='relu',
        filters=512,
        key_dim=64,
        value_dim=64,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=1,
        num_heads=8,
        use_layer_scale=True,
        use_multi_query=True,
        is_output=False,
    )

  blocks = [
      convbn(3, 2, 24),
      fused_ib(3, 2, 48, output=True),
      uib(3, 5, 2, 96),
      uib(3, 3, 1, 96, output=True),
      uib(3, 5, 2, 192),
      uib(3, 3, 1, 192),
      uib(3, 3, 1, 192),
      uib(3, 3, 1, 192),
      uib(3, 5, 1, 192),
      uib(5, 3, 1, 192),
      uib(5, 3, 1, 192),
      # add attention blocks to 2nd last stage
      mhsa_24px(),
      uib(5, 3, 1, 192),
      mhsa_24px(),
      uib(5, 3, 1, 192),
      mhsa_24px(),
      uib(5, 3, 1, 192),
      mhsa_24px(),
      uib(3, 0, 1, 192, output=True),
      # last stage
      uib(5, 5, 2, 512),
      uib(5, 5, 1, 512),
      uib(5, 5, 1, 512),
      uib(5, 5, 1, 512),
      uib(5, 0, 1, 512),
      uib(5, 3, 1, 512),
      uib(5, 0, 1, 512),
      uib(5, 0, 1, 512),
      uib(5, 3, 1, 512),
      uib(5, 5, 1, 512),
      mhsa_12px(),
      uib(5, 0, 1, 512),
      mhsa_12px(),
      uib(5, 0, 1, 512),
      mhsa_12px(),
      uib(5, 0, 1, 512),
      mhsa_12px(),
      uib(5, 0, 1, 512, output=True),
      convbn(1, 1, 960),
      BlockSpec(block_fn='gpooling', is_output=False),
      convbn(1, 1, 1280),
  ]
  return {
      'spec_name': 'MobileNetV4HybridLarge',
      'block_spec_schema': block_spec_field_list(),
      'block_specs': block_spec_values_to_list(blocks),
  }


SUPPORTED_SPECS_MAP = {
    'MobileNetV1': MNV1_BLOCK_SPECS,
    'MobileNetV2': MNV2_BLOCK_SPECS,
    'MobileNetV3Large': MNV3Large_BLOCK_SPECS,
    'MobileNetV3Small': MNV3Small_BLOCK_SPECS,
    'MobileNetV3EdgeTPU': MNV3EdgeTPU_BLOCK_SPECS,
    'MobileNetMultiMAX': MNMultiMAX_BLOCK_SPECS,
    'MobileNetMultiAVG': MNMultiAVG_BLOCK_SPECS,
    'MobileNetMultiAVGSeg': MNMultiAVG_SEG_BLOCK_SPECS,
    'MobileNetMultiMAXSeg': MNMultiMAX_SEG_BLOCK_SPECS,
    'MobileNetV3SmallReducedFilters': MNV3SmallReducedFilters,
    'MobileNetV4ConvSmall': MNV4ConvSmall_BLOCK_SPECS,
    'MobileNetV4ConvMedium': _mnv4_conv_medium_block_specs(),
    'MobileNetV4ConvLarge': MNV4ConvLarge_BLOCK_SPECS,
    'MobileNetV4HybridMedium': _mnv4_hybrid_medium_block_specs(),
    'MobileNetV4HybridLarge': _mnv4_hybrid_large_block_specs(),
    'MobileNetV4ConvMediumSeg': _mnv4_conv_medium_seg_block_specs(),
}


def block_spec_decoder(
    specs: dict[Any, Any],
    filter_size_scale: float,
    # Set to 1 for mobilenetv1.
    divisible_by: int = 8,
    finegrain_classification_mode: bool = True,
):
  """Decodes specs for a block.

  Args:
    specs: A `dict` specification of block specs of a mobilenet version.
    filter_size_scale: A `float` multiplier for the filter size for all
      convolution ops. The value must be greater than zero. Typical usage will
      be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
    divisible_by: An `int` that ensures all inner dimensions are divisible by
      this number.
    finegrain_classification_mode: If True, the model will keep the last layer
      large even for small multipliers, following
      https://arxiv.org/abs/1801.04381.

  Returns:
    A list of `BlockSpec` that defines structure of the base network.
  """

  spec_name = specs['spec_name']
  block_spec_schema = specs['block_spec_schema']
  block_specs = specs['block_specs']

  if not block_specs:
    raise ValueError(
        'The block spec cannot be empty for {} !'.format(spec_name))

  for block_spec in block_specs:
    if len(block_spec) != len(block_spec_schema):
      raise ValueError(
          'The block spec values {} do not match with the schema {}'.format(
              block_spec, block_spec_schema
          )
      )

  decoded_specs = []

  for s in block_specs:
    kw_s = dict(zip(block_spec_schema, s))
    decoded_specs.append(BlockSpec(**kw_s))

  # This adjustment applies to V2, V3, and V4
  if (spec_name != 'MobileNetV1'
      and finegrain_classification_mode
      and filter_size_scale < 1.0):
    decoded_specs[-1].filters /= filter_size_scale  # pytype: disable=annotation-type-mismatch

  for ds in decoded_specs:
    if ds.filters:
      ds.filters = nn_layers.round_filters(filters=ds.filters,
                                           multiplier=filter_size_scale,
                                           divisor=divisible_by,
                                           min_depth=8)

  return decoded_specs


@tf_keras.utils.register_keras_serializable(package='Vision')
class MobileNet(tf_keras.Model):
  """Creates a MobileNet family model."""

  def __init__(
      self,
      model_id: str = 'MobileNetV2',
      filter_size_scale: float = 1.0,
      input_specs: tf_keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]
      ),
      # The followings are for hyper-parameter tuning.
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: tf_keras.regularizers.Regularizer | None = None,
      bias_regularizer: tf_keras.regularizers.Regularizer | None = None,
      # The followings should be kept the same most of the times.
      output_stride: int | None = None,
      min_depth: int = 8,
      # divisible is not used in MobileNetV1.
      divisible_by: int = 8,
      stochastic_depth_drop_rate: float = 0.0,
      flat_stochastic_depth_drop_rate: bool = True,
      regularize_depthwise: bool = False,
      use_sync_bn: bool = False,
      # finegrain is not used in MobileNetV1.
      finegrain_classification_mode: bool = True,
      output_intermediate_endpoints: bool = False,
      **kwargs,
  ):
    """Initializes a MobileNet model.

    Args:
      model_id: A `str` of MobileNet version. The supported values are
        `MobileNetV1`, `MobileNetV2`, `MobileNetV3Large`, `MobileNetV3Small`,
        `MobileNetV3EdgeTPU`, `MobileNetMultiMAX` and `MobileNetMultiAVG`.
      filter_size_scale: A `float` of multiplier for the filters (number of
        channels) for all convolution ops. The value must be greater than zero.
        Typical usage will be to set this value in (0, 1) to reduce the number
        of parameters or computation cost of the model.
      input_specs: A `tf_keras.layers.InputSpec` of specs of the input tensor.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      output_stride: An `int` that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous
        convolution if necessary to prevent the network from reducing the
        spatial resolution of activation maps. Allowed values are 8 (accurate
        fully convolutional mode), 16 (fast fully convolutional mode), 32
        (classification mode).
      min_depth: An `int` of minimum depth (number of channels) for all
        convolution ops. Enforced when filter_size_scale < 1, and not an active
        constraint when filter_size_scale >= 1.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      stochastic_depth_drop_rate: A `float` of drop rate for drop connect layer.
      flat_stochastic_depth_drop_rate: A `bool`, indicating that the stochastic
        depth drop rate will be fixed and equal to all blocks.
      regularize_depthwise: If Ture, apply regularization on depthwise.
      use_sync_bn: If True, use synchronized batch normalization.
      finegrain_classification_mode: If True, the model will keep the last layer
        large even for small multipliers, following
        https://arxiv.org/abs/1801.04381.
      output_intermediate_endpoints: A `bool` of whether or not output the
        intermediate endpoints.
      **kwargs: Additional keyword arguments to be passed.
    """
    if model_id not in SUPPORTED_SPECS_MAP:
      raise ValueError('The MobileNet version {} '
                       'is not supported'.format(model_id))

    if filter_size_scale <= 0:
      raise ValueError('filter_size_scale is not greater than zero.')

    if output_stride is not None:
      if model_id == 'MobileNetV1':
        if output_stride not in [8, 16, 32]:
          raise ValueError('Only allowed output_stride values are 8, 16, 32.')
      else:
        if output_stride == 0 or (output_stride > 1 and output_stride % 2):
          raise ValueError('Output stride must be None, 1 or a multiple of 2.')

    self._model_id = model_id
    self._input_specs = input_specs
    self._filter_size_scale = filter_size_scale
    self._min_depth = min_depth
    self._output_stride = output_stride
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._flat_stochastic_depth_drop_rate = flat_stochastic_depth_drop_rate
    self._regularize_depthwise = regularize_depthwise
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._finegrain_classification_mode = finegrain_classification_mode
    self._output_intermediate_endpoints = output_intermediate_endpoints

    inputs = tf_keras.Input(shape=input_specs.shape[1:])

    block_specs = SUPPORTED_SPECS_MAP.get(model_id)
    self._decoded_specs = block_spec_decoder(
        specs=block_specs,
        filter_size_scale=self._filter_size_scale,
        divisible_by=self._get_divisible_by(),
        finegrain_classification_mode=self._finegrain_classification_mode,
    )

    x, endpoints, next_endpoint_level = self._mobilenet_base(inputs=inputs)

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    # Don't include the final layer in `self._output_specs` to support decoders.
    endpoints[str(next_endpoint_level)] = x

    super(MobileNet, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def _get_divisible_by(self):
    if self._model_id == 'MobileNetV1':
      return 1
    else:
      return self._divisible_by

  def _mobilenet_base(
      self, inputs: tf.Tensor
  ) -> tuple[tf.Tensor, dict[str, tf.Tensor], int]:
    """Builds the base MobileNet architecture.

    Args:
      inputs: A `tf.Tensor` of shape `[batch_size, height, width, channels]`.

    Returns:
      A tuple of output Tensor and dictionary that collects endpoints.
    """

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
      raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    # Used to calulate stochastic depth drop rate. Some blocks do not use
    # stochastic depth since they do not have residuals. For simplicity, we
    # count here all the blocks in the model. If one or more of the last layers
    # do not use stochastic depth, it can be compensated with larger stochastic
    # depth drop rate.
    num_blocks = len(self._decoded_specs)

    net = inputs
    endpoints = {}
    endpoint_level = 2
    for block_idx, block_def in enumerate(self._decoded_specs):
      block_name = 'block_group_{}_{}'.format(block_def.block_fn, block_idx)
      # A small catch for gpooling block with None strides
      if not block_def.strides:
        block_def.strides = 1
      if (self._output_stride is not None and
          current_stride == self._output_stride):
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= block_def.strides
      else:
        layer_stride = block_def.strides
        layer_rate = 1
        current_stride *= block_def.strides

      if self._flat_stochastic_depth_drop_rate:
        stochastic_depth_drop_rate = self._stochastic_depth_drop_rate
      else:
        stochastic_depth_drop_rate = nn_layers.get_stochastic_depth_rate(
            self._stochastic_depth_drop_rate, block_idx + 1, num_blocks
        )
      if stochastic_depth_drop_rate is not None:
        logging.info(
            'stochastic_depth_drop_rate: %f for block = %d',
            stochastic_depth_drop_rate,
            block_idx,
        )

      intermediate_endpoints = {}
      if block_def.block_fn == 'convbn':

        net = Conv2DBNBlock(
            filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=block_def.strides,
            activation=block_def.activation,
            use_bias=block_def.use_bias,
            use_normalization=block_def.use_normalization,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon
        )(net)

      elif block_def.block_fn == 'depsepconv':
        net = nn_blocks.DepthwiseSeparableConvBlock(
            filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=layer_stride,
            activation=block_def.activation,
            dilation_rate=layer_rate,
            regularize_depthwise=self._regularize_depthwise,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
        )(net)

      elif block_def.block_fn == 'mhsa':
        block = nn_blocks.MultiHeadSelfAttentionBlock(
            input_dim=block_def.filters,
            num_heads=block_def.num_heads,
            key_dim=block_def.key_dim,
            value_dim=block_def.value_dim,
            use_multi_query=block_def.use_multi_query,
            query_h_strides=block_def.query_h_strides,
            query_w_strides=block_def.query_w_strides,
            kv_strides=block_def.kv_strides,
            downsampling_dw_kernel_size=block_def.downsampling_dw_kernel_size,
            cpe_dw_kernel_size=block_def.kernel_size,
            stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
            use_sync_bn=self._use_sync_bn,
            use_residual=block_def.use_residual,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            use_layer_scale=block_def.use_layer_scale,
            output_intermediate_endpoints=self._output_intermediate_endpoints,
        )
        if self._output_intermediate_endpoints:
          net, intermediate_endpoints = block(net)
        else:
          net = block(net)

      elif block_def.block_fn in (
          'invertedbottleneck',
          'fused_ib',
          'uib',
      ):
        use_rate = rate
        if layer_rate > 1 and block_def.kernel_size != 1:
          # We will apply atrous rate in the following cases:
          # 1) When kernel_size is not in params, the operation then uses
          #   default kernel size 3x3.
          # 2) When kernel_size is in params, and if the kernel_size is not
          #   equal to (1, 1) (there is no need to apply atrous convolution to
          #   any 1x1 convolution).
          use_rate = layer_rate
        in_filters = net.shape.as_list()[-1]
        args = {
            'in_filters': in_filters,
            'out_filters': block_def.filters,
            'strides': layer_stride,
            'expand_ratio': block_def.expand_ratio,
            'activation': block_def.activation,
            'use_residual': block_def.use_residual,
            'dilation_rate': use_rate,
            'regularize_depthwise': self._regularize_depthwise,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon,
            'stochastic_depth_drop_rate': stochastic_depth_drop_rate,
            'divisible_by': self._get_divisible_by(),
            'output_intermediate_endpoints': (
                self._output_intermediate_endpoints
            ),
        }
        if block_def.block_fn in ('invertedbottleneck', 'fused_ib'):
          args.update({
              'kernel_size': block_def.kernel_size,
              'se_ratio': block_def.se_ratio,
              'expand_se_in_filters': True,
              'use_depthwise': (
                  block_def.use_depthwise
                  if block_def.block_fn == 'invertedbottleneck'
                  else False
              ),
              'se_gating_activation': 'hard_sigmoid',
          })
          block = nn_blocks.InvertedBottleneckBlock(**args)
        else:
          args.update({
              'middle_dw_downsample': block_def.middle_dw_downsample,
              'start_dw_kernel_size': block_def.start_dw_kernel_size,
              'middle_dw_kernel_size': block_def.middle_dw_kernel_size,
              'end_dw_kernel_size': block_def.end_dw_kernel_size,
              'use_layer_scale': block_def.use_layer_scale,
          })
          block = nn_blocks.UniversalInvertedBottleneckBlock(**args)

        if self._output_intermediate_endpoints:
          net, intermediate_endpoints = block(net)
        else:
          net = block(net)

      elif block_def.block_fn == 'gpooling':
        net = layers.GlobalAveragePooling2D(keepdims=True)(net)

      else:
        raise ValueError(
            'Unknown block type {} for layer {}'.format(
                block_def.block_fn, block_idx
            )
        )

      net = tf_keras.layers.Activation('linear', name=block_name)(net)

      if block_def.is_output:
        endpoints[str(endpoint_level)] = net
        for key, tensor in intermediate_endpoints.items():
          endpoints[str(endpoint_level) + '/' + key] = tensor
        if current_stride != self._output_stride:
          endpoint_level += 1

    if str(endpoint_level) in endpoints:
      endpoint_level += 1
    return net, endpoints, endpoint_level

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'filter_size_scale': self._filter_size_scale,
        'min_depth': self._min_depth,
        'output_stride': self._output_stride,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'flat_stochastic_depth_drop_rate': (
            self._flat_stochastic_depth_drop_rate
        ),
        'regularize_depthwise': self._regularize_depthwise,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'finegrain_classification_mode': self._finegrain_classification_mode,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('mobilenet')
def build_mobilenet(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer | None = None,
) -> tf_keras.Model:
  """Builds MobileNet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'mobilenet', (f'Inconsistent backbone type '
                                        f'{backbone_type}')

  return MobileNet(
      model_id=backbone_cfg.model_id,
      filter_size_scale=backbone_cfg.filter_size_scale,
      input_specs=input_specs,
      stochastic_depth_drop_rate=backbone_cfg.stochastic_depth_drop_rate,
      flat_stochastic_depth_drop_rate=(
          backbone_cfg.flat_stochastic_depth_drop_rate
      ),
      output_stride=backbone_cfg.output_stride,
      output_intermediate_endpoints=backbone_cfg.output_intermediate_endpoints,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
  )
