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

"""Definitions of MobileDet Networks."""

import dataclasses
from typing import Any, Dict, Optional, Tuple, List

import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.vision.modeling.backbones import factory
from official.vision.modeling.backbones import mobilenet
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers


layers = tf_keras.layers


#  pylint: disable=pointless-string-statement

"""
Architecture: https://arxiv.org/abs/2004.14525.

"MobileDets: Searching for Object Detection Architectures for
Mobile Accelerators" Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin,
Gabriel Bender, Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh,
Bo Chen

Note that `round_down_protection` flag should be set to false for scaling
of the network.
"""

MD_CPU_BLOCK_SPECS = {
    'spec_name': 'MobileDetCPU',
    # [expand_ratio] is set to 1 and [use_residual] is set to false
    # for inverted_bottleneck_no_expansion
    # [se_ratio] is set to 0.25 for all inverted_bottleneck layers
    # [activation] is set to 'hard_swish' for all applicable layers
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_residual', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, None, False),
        # inverted_bottleneck_no_expansion
        ('invertedbottleneck', 3, 1, 8, 'hard_swish', 0.25, 1., False, True),
        ('invertedbottleneck', 3, 2, 16, 'hard_swish', 0.25, 4., False, True),
        ('invertedbottleneck', 3, 2, 32, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 3, 1, 32, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 32, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 32, 'hard_swish', 0.25, 4., True, True),
        ('invertedbottleneck', 5, 2, 72, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, False),
        ('invertedbottleneck', 5, 1, 72, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, False),
        ('invertedbottleneck', 3, 1, 72, 'hard_swish', 0.25, 8., True, True),
        ('invertedbottleneck', 5, 2, 104, 'hard_swish', 0.25, 8., False, False),
        ('invertedbottleneck', 5, 1, 104, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 5, 1, 104, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 104, 'hard_swish', 0.25, 4., True, False),
        ('invertedbottleneck', 3, 1, 144, 'hard_swish', 0.25, 8., False, True),
    ]
}

MD_DSP_BLOCK_SPECS = {
    'spec_name': 'MobileDetDSP',
    # [expand_ratio] is set to 1 and [use_residual] is set to false
    # for inverted_bottleneck_no_expansion
    # [use_depthwise] is set to False for fused_conv
    # [se_ratio] is set to None for all inverted_bottleneck layers
    # [activation] is set to 'relu6' for all applicable layers
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'input_compression_ratio', 'output_compression_ratio',
                          'use_depthwise', 'use_residual', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu6',
         None, None, None, None, None, None, False),
        # inverted_bottleneck_no_expansion
        ('invertedbottleneck', 3, 1, 24, 'relu6',
         None, 1., None, None, True, False, True),
        ('invertedbottleneck', 3, 2, 32, 'relu6',
         None, 4., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 32, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 32, 'relu6',
         None, 4., None, None, True, True, False),
        ('tucker', 3, 1, 32, 'relu6',
         None, None, 0.25, 0.75, None, True, True),
        ('invertedbottleneck', 3, 2, 64, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 64, 'relu6',
         None, 4., None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 64, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 64, 'relu6',
         None, 4., None, None, False, True, True),  # fused_conv
        ('invertedbottleneck', 3, 2, 120, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 120, 'relu6',
         None, 4., None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 120, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 120, 'relu6',
         None, 8., None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 144, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 144, 'relu6',
         None, 8., None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 144, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 144, 'relu6',
         None, 8., None, None, True, True, True),
        ('invertedbottleneck', 3, 2, 160, 'relu6',
         None, 4, None, None, True, False, False),
        ('invertedbottleneck', 3, 1, 160, 'relu6',
         None, 4, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 160, 'relu6',
         None, 4., None, None, False, False, False),  # fused_conv
        ('tucker', 3, 1, 160, 'relu6',
         None, None, 0.75, 0.75, None, True, False),
        ('invertedbottleneck', 3, 1, 240, 'relu6',
         None, 8, None, None, True, False, True),
    ]
}

MD_EdgeTPU_BLOCK_SPECS = {
    'spec_name': 'MobileDetEdgeTPU',
    # [use_depthwise] is set to False for fused_conv
    # [se_ratio] is set to None for all inverted_bottleneck layers
    # [activation] is set to 'relu6' for all applicable layers
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'input_compression_ratio', 'output_compression_ratio',
                          'use_depthwise', 'use_residual', 'is_output'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu6',
         None, None, None, None, None, None, False),
        ('tucker', 3, 1, 16, 'relu6',
         None, None, 0.25, 0.75, None, False, True),
        ('invertedbottleneck', 3, 2, 16, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 16, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 16, 'relu6',
         None, 8., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 16, 'relu6',
         None, 4., None, None, False, True, True),  # fused_conv
        ('invertedbottleneck', 5, 2, 40, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 40, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 40, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 40, 'relu6',
         None, 4., None, None, False, True, True),  # fused_conv
        ('invertedbottleneck', 3, 2, 72, 'relu6',
         None, 8, None, None, True, False, False),
        ('invertedbottleneck', 3, 1, 72, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 72, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 72, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 5, 1, 96, 'relu6',
         None, 8, None, None, True, False, False),
        ('invertedbottleneck', 5, 1, 96, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 96, 'relu6',
         None, 8, None, None, True, True, True),
        ('invertedbottleneck', 5, 2, 120, 'relu6',
         None, 8, None, None, True, False, False),
        ('invertedbottleneck', 3, 1, 120, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 5, 1, 120, 'relu6',
         None, 4, None, None, True, True, False),
        ('invertedbottleneck', 3, 1, 120, 'relu6',
         None, 8, None, None, True, True, False),
        ('invertedbottleneck', 5, 1, 384, 'relu6',
         None, 8, None, None, True, False, True),
    ]
}

MD_GPU_BLOCK_SPECS = {
    'spec_name': 'MobileDetGPU',
    # [use_depthwise] is set to False for fused_conv
    # [se_ratio] is set to None for all inverted_bottleneck layers
    # [activation] is set to 'relu6' for all applicable layers
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'input_compression_ratio', 'output_compression_ratio',
                          'use_depthwise', 'use_residual', 'is_output'],
    'block_specs': [
        # block 0
        ('convbn', 3, 2, 32, 'relu6',
         None, None, None, None, None, None, False),
        # block 1
        ('tucker', 3, 1, 16, 'relu6',
         None, None, 0.25, 0.25, None, False, True),
        # block 2
        ('invertedbottleneck', 3, 2, 32, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('tucker', 3, 1, 32, 'relu6',
         None, None, 0.25, 0.25, None, True, False),
        ('tucker', 3, 1, 32, 'relu6',
         None, None, 0.25, 0.25, None, True, False),
        ('tucker', 3, 1, 32, 'relu6',
         None, None, 0.25, 0.25, None, True, True),
        # block 3
        ('invertedbottleneck', 3, 2, 64, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 64, 'relu6',
         None, 8., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 64, 'relu6',
         None, 8., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 64, 'relu6',
         None, 4., None, None, False, True, True),  # fused_conv
        # block 4
        ('invertedbottleneck', 3, 2, 128, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        # block 5
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 8., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 8., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 8., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 8., None, None, False, True, True),  # fused_conv
        # block 6
        ('invertedbottleneck', 3, 2, 128, 'relu6',
         None, 4., None, None, False, False, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        ('invertedbottleneck', 3, 1, 128, 'relu6',
         None, 4., None, None, False, True, False),  # fused_conv
        # block 7
        ('invertedbottleneck', 3, 1, 384, 'relu6',
         None, 8, None, None, True, False, True),
    ]
}

SUPPORTED_SPECS_MAP = {
    'MobileDetCPU': MD_CPU_BLOCK_SPECS,
    'MobileDetDSP': MD_DSP_BLOCK_SPECS,
    'MobileDetEdgeTPU': MD_EdgeTPU_BLOCK_SPECS,
    'MobileDetGPU': MD_GPU_BLOCK_SPECS,
}


@dataclasses.dataclass
class BlockSpec(hyperparams.Config):
  """A container class that specifies the block configuration for MobileDet."""

  block_fn: str = 'convbn'
  kernel_size: int = 3
  strides: int = 1
  filters: int = 32
  use_bias: bool = False
  use_normalization: bool = True
  activation: str = 'relu6'
  is_output: bool = True
  # Used for block type InvertedResConv and TuckerConvBlock.
  use_residual: bool = True
  # Used for block type InvertedResConv only.
  use_depthwise: bool = True
  expand_ratio: Optional[float] = 8.
  se_ratio: Optional[float] = None
  # Used for block type TuckerConvBlock only.
  input_compression_ratio: Optional[float] = None
  output_compression_ratio: Optional[float] = None


def block_spec_decoder(
    specs: Dict[Any, Any],
    filter_size_scale: float,
    divisible_by: int = 8) -> List[BlockSpec]:
  """Decodes specs for a block.

  Args:
    specs: A `dict` specification of block specs of a mobiledet version.
    filter_size_scale: A `float` multiplier for the filter size for all
      convolution ops. The value must be greater than zero. Typical usage will
      be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
    divisible_by: An `int` that ensures all inner dimensions are divisible by
      this number.

  Returns:
    A list of `BlockSpec` that defines structure of the base network.
  """

  spec_name = specs['spec_name']
  block_spec_schema = specs['block_spec_schema']
  block_specs = specs['block_specs']

  if not block_specs:
    raise ValueError(
        'The block spec cannot be empty for {} !'.format(spec_name))

  if len(block_specs[0]) != len(block_spec_schema):
    raise ValueError('The block spec values {} do not match with '
                     'the schema {}'.format(block_specs[0], block_spec_schema))

  decoded_specs = []

  for s in block_specs:
    kw_s = dict(zip(block_spec_schema, s))
    decoded_specs.append(BlockSpec(**kw_s))

  for ds in decoded_specs:
    if ds.filters:
      ds.filters = nn_layers.round_filters(filters=ds.filters,
                                           multiplier=filter_size_scale,
                                           divisor=divisible_by,
                                           round_down_protect=False,
                                           min_depth=8)

  return decoded_specs


@tf_keras.utils.register_keras_serializable(package='Vision')
class MobileDet(tf_keras.Model):
  """Creates a MobileDet family model."""

  def __init__(
      self,
      model_id: str = 'MobileDetCPU',
      filter_size_scale: float = 1.0,
      input_specs: tf_keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]),
      # The followings are for hyper-parameter tuning.
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      # The followings should be kept the same most of the times.
      min_depth: int = 8,
      divisible_by: int = 8,
      regularize_depthwise: bool = False,
      use_sync_bn: bool = False,
      **kwargs):
    """Initializes a MobileDet model.

    Args:
      model_id: A `str` of MobileDet version. The supported values are
        `MobileDetCPU`, `MobileDetDSP`, `MobileDetEdgeTPU`, `MobileDetGPU`.
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
      min_depth: An `int` of minimum depth (number of channels) for all
        convolution ops. Enforced when filter_size_scale < 1, and not an active
        constraint when filter_size_scale >= 1.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      regularize_depthwise: If Ture, apply regularization on depthwise.
      use_sync_bn: If True, use synchronized batch normalization.
      **kwargs: Additional keyword arguments to be passed.
    """
    if model_id not in SUPPORTED_SPECS_MAP:
      raise ValueError('The MobileDet version {} '
                       'is not supported'.format(model_id))

    if filter_size_scale <= 0:
      raise ValueError('filter_size_scale is not greater than zero.')

    self._model_id = model_id
    self._input_specs = input_specs
    self._filter_size_scale = filter_size_scale
    self._min_depth = min_depth
    self._divisible_by = divisible_by
    self._regularize_depthwise = regularize_depthwise
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    inputs = tf_keras.Input(shape=input_specs.shape[1:])

    block_specs = SUPPORTED_SPECS_MAP.get(model_id)
    self._decoded_specs = block_spec_decoder(
        specs=block_specs,
        filter_size_scale=self._filter_size_scale,
        divisible_by=self._get_divisible_by())

    x, endpoints, next_endpoint_level = self._mobiledet_base(inputs=inputs)

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(MobileDet, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def _get_divisible_by(self):
    return self._divisible_by

  def _mobiledet_base(self,
                      inputs: tf.Tensor
                      ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], int]:
    """Builds the base MobileDet architecture.

    Args:
      inputs: A `tf.Tensor` of shape `[batch_size, height, width, channels]`.

    Returns:
      A tuple of output Tensor and dictionary that collects endpoints.
    """

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
      raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

    net = inputs
    endpoints = {}
    endpoint_level = 1
    for i, block_def in enumerate(self._decoded_specs):
      block_name = 'block_group_{}_{}'.format(block_def.block_fn, i)

      if block_def.block_fn == 'convbn':

        net = mobilenet.Conv2DBNBlock(
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

      elif block_def.block_fn == 'invertedbottleneck':

        in_filters = net.shape.as_list()[-1]
        net = nn_blocks.InvertedBottleneckBlock(
            in_filters=in_filters,
            out_filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=block_def.strides,
            expand_ratio=block_def.expand_ratio,
            se_ratio=block_def.se_ratio,
            se_inner_activation=block_def.activation,
            se_gating_activation='sigmoid',
            se_round_down_protect=False,
            expand_se_in_filters=True,
            activation=block_def.activation,
            use_depthwise=block_def.use_depthwise,
            use_residual=block_def.use_residual,
            regularize_depthwise=self._regularize_depthwise,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            divisible_by=self._get_divisible_by()
        )(net)

      elif block_def.block_fn == 'tucker':

        in_filters = net.shape.as_list()[-1]
        net = nn_blocks.TuckerConvBlock(
            in_filters=in_filters,
            out_filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=block_def.strides,
            input_compression_ratio=block_def.input_compression_ratio,
            output_compression_ratio=block_def.output_compression_ratio,
            activation=block_def.activation,
            use_residual=block_def.use_residual,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            divisible_by=self._get_divisible_by()
        )(net)

      else:
        raise ValueError('Unknown block type {} for layer {}'.format(
            block_def.block_fn, i))

      net = tf_keras.layers.Activation('linear', name=block_name)(net)

      if block_def.is_output:
        endpoints[str(endpoint_level)] = net
        endpoint_level += 1

    return net, endpoints, endpoint_level

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'filter_size_scale': self._filter_size_scale,
        'min_depth': self._min_depth,
        'divisible_by': self._divisible_by,
        'regularize_depthwise': self._regularize_depthwise,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('mobiledet')
def build_mobiledet(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> tf_keras.Model:
  """Builds MobileDet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'mobiledet', (f'Inconsistent backbone type '
                                        f'{backbone_type}')

  return MobileDet(
      model_id=backbone_cfg.model_id,
      filter_size_scale=backbone_cfg.filter_size_scale,
      input_specs=input_specs,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
