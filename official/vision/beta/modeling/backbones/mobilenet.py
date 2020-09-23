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
"""Contains definitions of EfficientNet Networks."""

import math
from typing import Text, Optional, List, Dict

# Import libraries
from absl import logging
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers


class Conv2DBNBlock(tf.keras.layers.Layer):
  """An convolution block with batch normalization."""

  def __init__(self,
               filters: int,
               kernel_size: int = 3,
               strides: int = 1,
               use_biase: bool = False,
               activation: Text = 'relu6',
               kernel_initializer: Text = 'VarianceScaling',
               kernel_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               use_normalization: bool = True,
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               **kwargs):
    """An convolution block with batch normalization.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      kernel_size: `int` an integer specifying the height and width of the
      2D convolution window.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_biase: if True, use biase in the convolution layer.
      activation: `str` name of the activation function.
      kernel_size: `int` kernel_size of the conv layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      use_normalization: if True, use batch normalization.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(Conv2DBNBlock, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._use_biase = use_biase
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_normalization = use_normalization
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_biase': self._use_biase,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
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

    self._conv0 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=1,
        strides=self._strides,
        padding='same',
        use_bias=self._use_biase,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    if self._use_normalization:
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    super(Conv2DBNBlock, self).build(input_shape)

  def call(self, inputs, training=None):
    x = self._conv0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    return self._activation_fn(x)


class GlobalPoolingBlock(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GlobalPoolingBlock, self).__init__(**kwargs)

  def call(self, inputs, training=None):
    x = layers.GlobalAveragePooling2D()(inputs)
    outputs = layers.Reshape((1, 1, x.shape[1]))(x)
    return outputs


MNV1_BLOCK_SPECS = {
    'spec_name': 'MobileNetV1',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters'],
    'block_specs': [
        ('convbn', 3, 2, 32),
        ('depsepconv', 3, 1, 64),
        ('depsepconv', 3, 2, 128),
        ('depsepconv', 3, 1, 128),
        ('depsepconv', 3, 2, 256),
        ('depsepconv', 3, 1, 256),
        ('depsepconv', 3, 2, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 2, 1024),
        ('depsepconv', 3, 1, 1024),
    ]
}

MNV2_BLOCK_SPECS = {
    'spec_name': 'MobileNetV2',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides',
                          'filters', 'expand_ratio'],
    'block_specs': [
        ('convbn', 3, 2, 32, None),
        ('mbconv', 3, 1, 16, 1),

        ('mbconv', 3, 2, 24, 6),
        ('mbconv', 3, 1, 24, 6),

        ('mbconv', 3, 2, 32, 6),
        ('mbconv', 3, 1, 32, 6),
        ('mbconv', 3, 1, 32, 6),

        ('mbconv', 3, 2, 64, 6),
        ('mbconv', 3, 1, 64, 6),
        ('mbconv', 3, 1, 64, 6),
        ('mbconv', 3, 1, 64, 6),

        ('mbconv', 3, 1, 96, 6),
        ('mbconv', 3, 1, 96, 6),
        ('mbconv', 3, 1, 96, 6),

        ('mbconv', 3, 2, 160, 6),
        ('mbconv', 3, 1, 160, 6),
        ('mbconv', 3, 1, 160, 6),

        ('mbconv', 3, 1, 320, 6),

        ('convbn', 1, 2, 1280, None),
    ]
}

MNV3Large_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Large',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides',
                          'filters', 'activation',
                          'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_biase'],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, True, False),

        ('mbconv', 3, 1, 16, 'relu', None, 1., True, False),

        ('mbconv', 3, 2, 24, 'relu', None, 4., True, False),
        ('mbconv', 3, 1, 24, 'relu', None, 3., True, False),

        ('mbconv', 5, 2, 40, 'relu', 1. / 4, 3., True, False),
        ('mbconv', 5, 1, 40, 'relu', 1. / 4, 3., True, False),
        ('mbconv', 5, 1, 40, 'relu', 1. / 4, 3., True, False),

        ('mbconv', 3, 2, 80, 'hard_swish', None, 6., True, False),
        ('mbconv', 3, 1, 80, 'hard_swish', None, 2.5, True, False),
        ('mbconv', 3, 1, 80, 'hard_swish', None, 2.3, True, False),
        ('mbconv', 3, 1, 80, 'hard_swish', None, 2.3, True, False),

        ('mbconv', 3, 1, 112, 'hard_swish', 1. / 4, 6., True, False),
        ('mbconv', 3, 1, 112, 'hard_swish', 1. / 4, 6., True, False),

        ('mbconv', 5, 2, 160, 'hard_swish', 1. / 4, 6, True, False),
        ('mbconv', 5, 1, 160, 'hard_swish', 1. / 4, 6, True, False),
        ('mbconv', 5, 1, 160, 'hard_swish', 1. / 4, 6, True, False),

        ('convbn', 1, 1, 960, 'hard_swish', None, None, True, False),
        ('gpooling', None, None, None, None, None, None, None, None),
        ('convbn', 1, 1, 1280, 'hard_swish', None, None, False, True),
    ]
}

MNV3Small_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Small',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides',
                          'filters', 'activation',
                          'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_biase'],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, True, False),

        ('mbconv', 3, 2, 16, 'relu', None, 1, True, False),

        ('mbconv', 3, 2, 24, 'relu', None, 72. / 16, True, False),
        ('mbconv', 3, 1, 24, 'relu', None, 88. / 24, True, False),

        ('mbconv', 5, 2, 40, 'hard_swish', 1. / 4, 4., True, False),
        ('mbconv', 5, 1, 40, 'hard_swish', 1. / 4, 6., True, False),
        ('mbconv', 5, 1, 40, 'hard_swish', 1. / 4, 6., True, False),

        ('mbconv', 5, 1, 48, 'hard_swish', 1. / 4, 3., True, False),
        ('mbconv', 5, 1, 48, 'hard_swish', 1. / 4, 3., True, False),

        ('mbconv', 5, 2, 96, 'hard_swish', 1. / 4, 6., True, False),
        ('mbconv', 5, 1, 96, 'hard_swish', 1. / 4, 6., True, False),
        ('mbconv', 5, 1, 96, 'hard_swish', 1. / 4, 6., True, False),

        ('convbn', 1, 1, 576, 'hard_swish', None, None, True, False),
        ('gpooling', None, None, None, None, None, None, None, None),
        ('convbn', 1, 1, 1024, 'hard_swish', None, None, False, True),
    ]
}

MNV3EdgeTPU_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3EdgeTPU',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides',
                          'filters', 'activation',
                          'se_ratio', 'expand_ratio',
                          'use_residual', 'use_depthwise'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, None, None, None),

        ('mbconv', 3, 1, 16, 'relu', None, 1, True, False),

        ('mbconv', 3, 2, 32, 'relu', None, 8., True, False),
        ('mbconv', 3, 1, 32, 'relu', None, 4., True, False),
        ('mbconv', 3, 2, 32, 'relu', None, 4., True, False),
        ('mbconv', 3, 1, 32, 'relu', None, 4., True, False),

        ('mbconv', 3, 2, 48, 'relu', None, 8., True, False),
        ('mbconv', 3, 1, 48, 'relu', None, 4., True, False),
        ('mbconv', 3, 2, 48, 'relu', None, 4., True, False),
        ('mbconv', 3, 1, 48, 'relu', None, 4., True, False),

        ('mbconv', 3, 2, 96, 'relu', None, 8., True, True),
        ('mbconv', 3, 1, 96, 'relu', None, 4., True, True),
        ('mbconv', 3, 2, 96, 'relu', None, 4., True, True),
        ('mbconv', 3, 1, 96, 'relu', None, 4., True, True),

        ('mbconv', 3, 1, 96, 'relu', None, 8., False, True),
        ('mbconv', 3, 1, 96, 'relu', None, 4., True, True),
        ('mbconv', 3, 2, 96, 'relu', None, 4., True, True),
        ('mbconv', 3, 1, 96, 'relu', None, 4., True, True),

        ('mbconv', 5, 2, 160, 'relu', None, 8., True, True),
        ('mbconv', 5, 1, 160, 'relu', None, 4., True, True),
        ('mbconv', 5, 2, 160, 'relu', None, 4., True, True),
        ('mbconv', 5, 1, 160, 'relu', None, 4., True, True),

        ('mbconv', 3, 1, 192, 'relu', None, 8., True, False),

        ('convbn', 1, 1, 1280, 'relu', None, None, None, None),
    ]
}

SUPPORTED_SPECS_MAP = {
    'MobileNetV1': MNV1_BLOCK_SPECS,
    'MobileNetV2': MNV2_BLOCK_SPECS,
    'MobileNetV3Large': MNV3Large_BLOCK_SPECS,
    'MobileNetV3Small': MNV3Small_BLOCK_SPECS,
    'MobileNetV3EdgeTPU': MNV3EdgeTPU_BLOCK_SPECS
}

BLOCK_FN_MAP = {
    'convbn': Conv2DBNBlock,
    'gpooling': GlobalPoolingBlock,
    'depsepconv': nn_blocks.DepthwiseSeparableConvBlock,
    'mbconv': nn_blocks.InvertedBottleneckBlock,

}


class BlockSpec(object):
  """A container class that specifies the block configuration for MnasNet."""

  def __init__(self,
               block_fn: Text = 'convbn',
               kernel_size: int = 3,
               strides: int = 1,
               filters: int = 32,
               use_biase: bool = False,
               use_normalization: bool = True,
               activation: Text = 'relu6',
               # used for block type InvertedResConv
               expand_ratio: Optional[float] = 6.,
               # used for block type InvertedResConv with SE
               se_ratio: Optional[float] = None,
               use_depthwise: bool = True,
               use_residual: bool = True, ):
    self.block_fn = block_fn
    self.kernel_size = kernel_size
    self.strides = strides
    self.filters = filters
    self.use_biase = use_biase
    self.use_normalization = use_normalization
    self.activation = activation
    self.expand_ratio = expand_ratio
    self.se_ratio = se_ratio
    self.use_depthwise = use_depthwise
    self.use_residual = use_residual


def block_spec_decoder(specs,
                       width_multiplier,
                       # set to 1 for mobilenetv1
                       divisible_by: int = 8,
                       finegrain_classification_mode: bool = True):
  """Decode specs for a block."""

  spec_name = specs['spec_name']
  block_spec_schema = specs['block_spec_schema']
  block_specs = specs['block_specs']

  if spec_name not in SUPPORTED_SPECS_MAP:
    raise ValueError('Model spec: {} is not supported !'.format(spec_name))

  if len(block_specs) == 0:
    raise ValueError('The block spec cannot be empty for {} !'.format(spec_name))

  if len(block_specs[0]) != len(block_spec_schema):
    raise ValueError('The block spec values {} do not match with '
                     'the schema {}'.format(block_specs[0], block_spec_schema))

  decoded_specs = []

  for s in block_specs:
    kw_s = dict(zip(block_spec_schema, s))
    decoded_specs.append(BlockSpec(**kw_s))

  # This adjustment applies to V2 and V3
  if (spec_name != 'MobileNetV1'
      and finegrain_classification_mode
      and width_multiplier < 1.0):
    decoded_specs[-1].filters /= width_multiplier

  for ds in decoded_specs:
    ds.filters = nn_layers.round_filters(filters=ds.filters,
                                         multiplier=width_multiplier,
                                         divisor=divisible_by,
                                         min_depth=8)

    return decoded_specs


def mobilenet_base(inputs: tf.Tensor,
                   spec_blocks: List[BlockSpec],
                   divisible_by: int = 8,
                   output_stride: int = None,
                   stochastic_depth_drop_rate=0.0,
                   regularize_depthwise=False,
                   kernel_initializer='VarianceScaling',
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   use_sync_bn=False,
                   norm_momentum=0.99,
                   norm_epsilon=0.001,
                   ) -> (tf.Tensor, Dict[tf.Tensor]):
  """Build the base MobileNet architecture.

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels].
    spec_blocks: `List[BlockSpec]` defines structure of the base network.
    divisible_by: `int` ensures all inner dimensions are divisible by
      this number.
    output_stride: `int` specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    stochastic_depth_drop_rate: `float` drop rate for drop connect layer.
    regularize_depthwise: if Ture, apply regularization on depthwise.
    kernel_initializer: `str` kernel_initializer for convolutional layers.
    kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      Default to None.
    bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      Default to None.
    use_sync_bn: if True, use synchronized batch normalization.
    norm_momentum: `float` normalization omentum for the moving average.
    norm_epsilon: `float` small float added to variance to avoid dividing by
      zero.

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

  net = inputs
  endpoints = {}
  endpoint_level = 1
  for i, block_def in enumerate(spec_blocks):
    name = 'block_group_{}_{}'.format(block_def.block_fn, i)
    if output_stride is not None and current_stride == output_stride:
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

    if block_def.block_fn == 'convbn':

      net = Conv2DBNBlock(
          filters=block_def.filters,
          kernel_size=block_def.kernel_size,
          strides=block_def.strides,
          activation=block_def.activation,
          use_biase=block_def.use_biase,
          use_normalization=block_def.use_normalization,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          use_sync_bn=use_sync_bn,
          norm_momentum=norm_momentum,
          norm_epsilon=norm_epsilon
      )(net)

    elif block_def.block_fn == 'depsepconv':
      net = nn_blocks.DepthwiseSeparableConvBlock(
          filters=block_def.filters,
          kernel_size=block_def.kernel_size,
          strides=block_def.strides,
          activation=block_def.activation,
          use_normalization=block_def.use_normalization,
          dilation_rate=layer_rate,
          regularize_depthwise=regularize_depthwise,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          use_sync_bn=use_sync_bn,
          norm_momentum=norm_momentum,
          norm_epsilon=norm_epsilon,
      )(net)

    elif block_def.block_fn == 'mbconv':
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
      net = nn_blocks.InvertedBottleneckBlock(
          in_filters=in_filters,
          out_filters=block_def.filters,
          kernel_size=block_def.kernel_size,
          strides=layer_stride,
          expand_ratio=block_def.expand_ratio,
          se_ratio=block_def.se_ratio,
          activation=block_def.activation,
          use_biase=block_def.use_biase,
          use_residual=block_def.use_residual,
          use_normalization=block_def.use_normalization,
          dilation_rate=use_rate,
          regularize_depthwise=regularize_depthwise,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          use_sync_bn=use_sync_bn,
          norm_momentum=norm_momentum,
          norm_epsilon=norm_epsilon,
          stochastic_depth_drop_rate=stochastic_depth_drop_rate,
          divisible_by=divisible_by,
      )(net)

    elif block_def.block_fn == 'gpooling':
      net = GlobalPoolingBlock()(net)
    else:
      raise ValueError('Unknown block type {} for layer {}'.format(
          block_def.block_fn, i))

    endpoints[endpoint_level] = net
    endpoint_level += 1
    net = tf.identity(net, name=name)
  return net, endpoints


@tf.keras.utils.register_keras_serializable(package='Vision')
class MobileNet(tf.keras.Model):
  def __init__(self,
               version: Text = 'MobileNetV2',
               width_multiplier: float = 1.0,
               input_specs: layers.InputSpec = layers.InputSpec(
                   shape=[None, None, None, 3]),
               # The followings are for hyper-parameter tuning
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               dropout_keep_prob: float = 0.8,
               kernel_initializer: Text = 'VarianceScaling',
               kernel_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               # The followings should be kept the same most of the times
               output_stride: int = None,
               min_depth: int = 8,
               # divisible is not used in MobileNetV1
               divisible_by: int = 8,
               stochastic_depth_drop_rate: float = 0.0,
               regularize_depthwise: bool = False,
               use_sync_bn: bool = False,
               # finegrain is not used in MobileNetV1
               finegrain_classification_mode: bool = True,
               **kwargs):
    """

    Args:
      version: `str` version of MobileNet. The supported values are MobileNetV2',
      'MobileNetV3Large', 'MobileNetV3Small', and 'MobileNetV3EdgeTPU'.
      width_multiplier: `float` multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      dropout_keep_prob: `float` the percentage of activation values that are
        retained.
      kernel_initializer: `str` kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      output_stride: `int` specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      min_depth: `int` minimum depth (number of channels) for all conv ops.
        Enforced when width_multiplier < 1, and not an active constraint when
        width_multiplier >= 1.
      divisible_by: `int` ensures all inner dimensions are divisible by
        this number.
      stochastic_depth_drop_rate: `float` drop rate for drop connect layer.
      regularize_depthwise: if Ture, apply regularization on depthwise.
      use_sync_bn: if True, use synchronized batch normalization.
      finegrain_classification_mode: if True, the model
        will keep the last layer large even for small multipliers. Following
        https://arxiv.org/abs/1801.04381
      **kwargs: keyword arguments to be passed.
    """
    if version not in SUPPORTED_SPECS_MAP:
      raise ValueError('The MobileNet version {} '
                       'is not supported'.format(version))

    if width_multiplier <= 0:
      raise ValueError('depth_multiplier is not greater than zero.')

    if output_stride is not None:
      if version == 'MobileNetV1':
        if output_stride not in [8, 16, 32]:
          raise ValueError('Only allowed output_stride values are 8, 16, 32.')
      else:
        if output_stride == 0 or (output_stride > 1 and output_stride % 2):
          raise ValueError('Output stride must be None, 1 or a multiple of 2.')

    if version == 'MobileNetV1':
      divisible_by = 1

    self._version = version
    self._input_specs = input_specs
    self._width_multiplier = width_multiplier
    self._min_depth = min_depth
    self._output_stride = output_stride
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._regularize_depthwise = regularize_depthwise
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._dropout_keep_prob = dropout_keep_prob
    self._finegrain_classification_mode = finegrain_classification_mode
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization

    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    block_specs = SUPPORTED_SPECS_MAP.get(version)
    decoded_specs = block_spec_decoder(
        specs=block_specs,
        width_multiplier=self._width_multiplier,
        # set to 1 for mobilenetv1
        divisible_by=self._divisible_by,
        finegrain_classification_mode=self._finegrain_classification_mode)

    x, endpoints = mobilenet_base(
        inputs=inputs,
        spec_blocks=decoded_specs,
        divisible_by=self._divisible_by,
        output_stride=self._output_stride,
        stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
        regularize_depthwise=self._regularize_depthwise,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)

    endpoints[max(endpoints.keys()) + 1] = x
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(MobileNet, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def get_config(self):
    config_dict = {
        'version': self._version,
        'width_multiplier': self._width_multiplier,
        'min_depth': self._min_depth,
        'output_stride': self._output_stride,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'regularize_depthwise': self._regularize_depthwise,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'dropout_keep_prob': self._dropout_keep_prob,
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
