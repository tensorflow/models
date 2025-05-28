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

"""Contains quantized neural blocks for the QAT."""
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official.modeling import tf_utils
from official.projects.qat.vision.modeling.layers import nn_layers as qat_nn_layers
from official.projects.qat.vision.quantization import configs
from official.projects.qat.vision.quantization import helper
from official.vision.modeling.layers import nn_layers


# This class is copied from modeling.layers.nn_blocks.BottleneckBlock and apply
# QAT.
@tf_keras.utils.register_keras_serializable(package='Vision')
class BottleneckBlockQuantized(tf_keras.layers.Layer):
  """A quantized standard bottleneck block."""

  def __init__(self,
               filters: int,
               strides: int,
               dilation_rate: int = 1,
               use_projection: bool = False,
               se_ratio: Optional[float] = None,
               resnetd_shortcut: bool = False,
               stochastic_depth_drop_rate: Optional[float] = None,
               kernel_initializer: str = 'VarianceScaling',
               kernel_regularizer: tf_keras.regularizers.Regularizer = None,
               bias_regularizer: tf_keras.regularizers.Regularizer = None,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               bn_trainable: bool = True,  # pytype: disable=annotation-type-mismatch  # typed-keras
               **kwargs):
    """Initializes a standard bottleneck block with BN after convolutions.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      strides: An `int` block stride. If greater than 1, this block will
        ultimately downsample the input.
      dilation_rate: An `int` dilation_rate of convolutions. Default to 1.
      use_projection: A `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      resnetd_shortcut: A `bool`. If True, apply the resnetd style modification
        to the shortcut connection.
      stochastic_depth_drop_rate: A `float` or None. If not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      bn_trainable: A `bool` that indicates whether batch norm layers should be
        trainable. Default to True.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(BottleneckBlockQuantized, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._resnetd_shortcut = resnetd_shortcut
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    norm_layer = (
        tf_keras.layers.experimental.SyncBatchNormalization
        if use_sync_bn else tf_keras.layers.BatchNormalization)
    self._norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    self._norm = helper.BatchNormalizationNoQuantized(norm_layer)

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._bn_trainable = bn_trainable

  def build(self, input_shape: Optional[Union[Sequence[int], tf.Tensor]]):
    """Build variables and child layers to prepare for calling."""
    if self._use_projection:
      if self._resnetd_shortcut:
        self._shortcut0 = tf_keras.layers.AveragePooling2D(
            pool_size=2, strides=self._strides, padding='same')
        self._shortcut1 = helper.Conv2DQuantized(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=helper.NoOpActivation())
      else:
        self._shortcut = helper.Conv2DQuantized(
            filters=self._filters * 4,
            kernel_size=1,
            strides=self._strides,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=helper.NoOpActivation())

      self._norm0 = self._norm_with_quantize(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          trainable=self._bn_trainable)

    self._conv1 = helper.Conv2DQuantized(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation1 = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())

    self._conv2 = helper.Conv2DQuantized(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        dilation_rate=self._dilation_rate,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation2 = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())

    self._conv3 = helper.Conv2DQuantized(
        filters=self._filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())
    self._norm3 = self._norm_with_quantize(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation3 = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = qat_nn_layers.SqueezeExcitationQuantized(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_keras.layers.Add(),
        configs.Default8BitQuantizeConfig([], [], True))

    super(BottleneckBlockQuantized, self).build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    """Get a config of this layer."""
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'dilation_rate': self._dilation_rate,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'resnetd_shortcut': self._resnetd_shortcut,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'bn_trainable': self._bn_trainable
    }
    base_config = super(BottleneckBlockQuantized, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[Union[bool, tf.Tensor]] = None) -> tf.Tensor:
    """Run the BottleneckBlockQuantized logics."""
    shortcut = inputs
    if self._use_projection:
      if self._resnetd_shortcut:
        shortcut = self._shortcut0(shortcut)
        shortcut = self._shortcut1(shortcut)
      else:
        shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation1(x)

    x = self._conv2(x)
    x = self._norm2(x)
    x = self._activation2(x)

    x = self._conv3(x)
    x = self._norm3(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    x = self._add([x, shortcut])
    return self._activation3(x)


# This class is copied from modeling.backbones.mobilenet.Conv2DBNBlock and apply
# QAT.
@tf_keras.utils.register_keras_serializable(package='Vision')
class Conv2DBNBlockQuantized(tf_keras.layers.Layer):
  """A quantized convolution block with batch normalization."""

  def __init__(
      self,
      filters: int,
      kernel_size: int = 3,
      strides: int = 1,
      use_bias: bool = False,
      use_explicit_padding: bool = False,
      activation: str = 'relu6',
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      use_normalization: bool = True,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs):
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
    super(Conv2DBNBlockQuantized, self).__init__(**kwargs)
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

    if use_explicit_padding and kernel_size > 1:
      self._padding = 'valid'
    else:
      self._padding = 'same'

    norm_layer = (
        tf_keras.layers.experimental.SyncBatchNormalization
        if use_sync_bn else tf_keras.layers.BatchNormalization)
    self._norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    self._norm = helper.BatchNormalizationNoQuantized(norm_layer)

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

  def get_config(self) -> Dict[str, Any]:
    """Get a config of this layer."""
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
    base_config = super(Conv2DBNBlockQuantized, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape: Optional[Union[Sequence[int], tf.Tensor]]):
    """Build variables and child layers to prepare for calling."""
    if self._use_explicit_padding and self._kernel_size > 1:
      padding_size = nn_layers.get_padding_for_kernel_size(self._kernel_size)
      self._pad = tf_keras.layers.ZeroPadding2D(padding_size)
    conv2d_quantized = (
        helper.Conv2DQuantized
        if self._use_normalization else helper.Conv2DOutputQuantized)

    self._conv0 = conv2d_quantized(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding=self._padding,
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())
    if self._use_normalization:
      self._norm0 = helper.norm_by_activation(self._activation,
                                              self._norm_with_quantize,
                                              self._norm)(
                                                  axis=self._bn_axis,
                                                  momentum=self._norm_momentum,
                                                  epsilon=self._norm_epsilon)
    self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    super(Conv2DBNBlockQuantized, self).build(input_shape)

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[Union[bool, tf.Tensor]] = None) -> tf.Tensor:
    """Run the Conv2DBNBlockQuantized logics."""
    if self._use_explicit_padding and self._kernel_size > 1:
      inputs = self._pad(inputs)
    x = self._conv0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    return self._activation_layer(x)


@tf_keras.utils.register_keras_serializable(package='Vision')
class InvertedBottleneckBlockQuantized(tf_keras.layers.Layer):
  """A quantized inverted bottleneck block."""

  def __init__(self,
               in_filters,
               out_filters,
               expand_ratio,
               strides,
               kernel_size=3,
               se_ratio=None,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               se_inner_activation='relu',
               se_gating_activation='sigmoid',
               se_round_down_protect=True,
               expand_se_in_filters=False,
               depthwise_activation=None,
               use_sync_bn=False,
               dilation_rate=1,
               divisible_by=1,
               regularize_depthwise=False,
               use_depthwise=True,
               use_residual=True,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               output_intermediate_endpoints=False,
               **kwargs):
    """Initializes an inverted bottleneck block with BN after convolutions.

    Args:
      in_filters: An `int` number of filters of the input tensor.
      out_filters: An `int` number of filters of the output tensor.
      expand_ratio: An `int` of expand_ratio for an inverted bottleneck block.
      strides: An `int` block stride. If greater than 1, this block will
        ultimately downsample the input.
      kernel_size: An `int` kernel_size of the depthwise conv layer.
      se_ratio: A `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      se_inner_activation: A `str` name of squeeze-excitation inner activation.
      se_gating_activation: A `str` name of squeeze-excitation gating
        activation.
      se_round_down_protect: A `bool` of whether round down more than 10% will
        be allowed in SE layer.
      expand_se_in_filters: A `bool` of whether or not to expand in_filter in
        squeeze and excitation layer.
      depthwise_activation: A `str` name of the activation function for
        depthwise only.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      dilation_rate: An `int` that specifies the dilation rate to use for.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      dilated convolution: An `int` to specify the same value for all spatial
        dimensions.
      regularize_depthwise: A `bool` of whether or not apply regularization on
        depthwise.
      use_depthwise: A `bool` of whether to uses fused convolutions instead of
        depthwise.
      use_residual: A `bool` of whether to include residual connection between
        input and output.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      output_intermediate_endpoints: A `bool` of whether or not output the
        intermediate endpoints.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(InvertedBottleneckBlockQuantized, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._expand_ratio = expand_ratio
    self._strides = strides
    self._kernel_size = kernel_size
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._dilation_rate = dilation_rate
    self._use_sync_bn = use_sync_bn
    self._regularize_depthwise = regularize_depthwise
    self._use_depthwise = use_depthwise
    self._use_residual = use_residual
    self._activation = activation
    self._se_inner_activation = se_inner_activation
    self._se_gating_activation = se_gating_activation
    self._se_round_down_protect = se_round_down_protect
    self._depthwise_activation = depthwise_activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._expand_se_in_filters = expand_se_in_filters
    self._output_intermediate_endpoints = output_intermediate_endpoints

    norm_layer = (
        tf_keras.layers.experimental.SyncBatchNormalization
        if use_sync_bn else tf_keras.layers.BatchNormalization)
    self._norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    self._norm = helper.BatchNormalizationNoQuantized(norm_layer)

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if not depthwise_activation:
      self._depthwise_activation = activation
    if regularize_depthwise:
      self._depthwise_regularizer = kernel_regularizer
    else:
      self._depthwise_regularizer = None

  def build(self, input_shape: Optional[Union[Sequence[int], tf.Tensor]]):
    """Build variables and child layers to prepare for calling."""
    expand_filters = self._in_filters
    if self._expand_ratio > 1:
      # First 1x1 conv for channel expansion.
      expand_filters = nn_layers.make_divisible(
          self._in_filters * self._expand_ratio, self._divisible_by)

      expand_kernel = 1 if self._use_depthwise else self._kernel_size
      expand_stride = 1 if self._use_depthwise else self._strides

      self._conv0 = helper.Conv2DQuantized(
          filters=expand_filters,
          kernel_size=expand_kernel,
          strides=expand_stride,
          padding='same',
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=helper.NoOpActivation())
      self._norm0 = helper.norm_by_activation(self._activation,
                                              self._norm_with_quantize,
                                              self._norm)(
                                                  axis=self._bn_axis,
                                                  momentum=self._norm_momentum,
                                                  epsilon=self._norm_epsilon)
      self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
          tf_utils.get_activation(self._activation, use_keras_layer=True),
          configs.Default8BitActivationQuantizeConfig())
    if self._use_depthwise:
      # Depthwise conv.
      self._conv1 = helper.DepthwiseConv2DQuantized(
          kernel_size=(self._kernel_size, self._kernel_size),
          strides=self._strides,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=self._kernel_initializer,
          depthwise_regularizer=self._depthwise_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=helper.NoOpActivation(),
      )
      self._norm1 = helper.norm_by_activation(self._depthwise_activation,
                                              self._norm_with_quantize,
                                              self._norm)(
                                                  axis=self._bn_axis,
                                                  momentum=self._norm_momentum,
                                                  epsilon=self._norm_epsilon)
      self._depthwise_activation_layer = (
          tfmot.quantization.keras.QuantizeWrapperV2(
              tf_utils.get_activation(self._depthwise_activation,
                                      use_keras_layer=True),
              configs.Default8BitActivationQuantizeConfig()))

    # Squeeze and excitation.
    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      logging.info('Use Squeeze and excitation.')
      in_filters = self._in_filters
      if self._expand_se_in_filters:
        in_filters = expand_filters
      self._squeeze_excitation = qat_nn_layers.SqueezeExcitationQuantized(
          in_filters=in_filters,
          out_filters=expand_filters,
          se_ratio=self._se_ratio,
          divisible_by=self._divisible_by,
          round_down_protect=self._se_round_down_protect,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._se_inner_activation,
          gating_activation=self._se_gating_activation)
    else:
      self._squeeze_excitation = None

    # Last 1x1 conv.
    self._conv2 = helper.Conv2DQuantized(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())
    self._norm2 = self._norm_with_quantize(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_keras.layers.Add(),
        configs.Default8BitQuantizeConfig([], [], True))

    super(InvertedBottleneckBlockQuantized, self).build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    """Get a config of this layer."""
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'expand_ratio': self._expand_ratio,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'se_ratio': self._se_ratio,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'se_inner_activation': self._se_inner_activation,
        'se_gating_activation': self._se_gating_activation,
        'se_round_down_protect': self._se_round_down_protect,
        'expand_se_in_filters': self._expand_se_in_filters,
        'depthwise_activation': self._depthwise_activation,
        'dilation_rate': self._dilation_rate,
        'use_sync_bn': self._use_sync_bn,
        'regularize_depthwise': self._regularize_depthwise,
        'use_depthwise': self._use_depthwise,
        'use_residual': self._use_residual,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'output_intermediate_endpoints': self._output_intermediate_endpoints
    }
    base_config = super(InvertedBottleneckBlockQuantized, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[Union[bool, tf.Tensor]] = None
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
    """Run the InvertedBottleneckBlockQuantized logics."""
    endpoints = {}
    shortcut = inputs
    if self._expand_ratio > 1:
      x = self._conv0(inputs)
      x = self._norm0(x)
      x = self._activation_layer(x)
    else:
      x = inputs

    if self._use_depthwise:
      x = self._conv1(x)
      x = self._norm1(x)
      x = self._depthwise_activation_layer(x)
      if self._output_intermediate_endpoints:
        endpoints['depthwise'] = x

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if (self._use_residual and self._in_filters == self._out_filters and
        self._strides == 1):
      if self._stochastic_depth:
        x = self._stochastic_depth(x, training=training)
      x = self._add([x, shortcut])

    if self._output_intermediate_endpoints:
      return x, endpoints
    return x


@tf_keras.utils.register_keras_serializable(package='Vision')
class UniversalInvertedBottleneckBlockQuantized(tf_keras.layers.Layer):
  """A quantized inverted bottleneck block with optional depthwise convs."""

  def __init__(
      self,
      in_filters: int,
      out_filters: int,
      expand_ratio: float,
      strides: int,
      middle_dw_downsample: bool = True,
      start_dw_kernel_size: int = 0,
      middle_dw_kernel_size: int = 3,
      end_dw_kernel_size: int = 0,
      stochastic_depth_drop_rate: float | None = None,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: tf_keras.regularizers.Regularizer | None = None,
      bias_regularizer: tf_keras.regularizers.Regularizer | None = None,
      activation: str = 'relu',
      depthwise_activation: str | None = None,
      use_sync_bn: bool = False,
      dilation_rate: int = 1,
      divisible_by: int = 1,
      regularize_depthwise: bool = False,
      use_residual: bool = True,
      use_layer_scale: bool = False,
      layer_scale_init_value: float = 1e-5,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      output_intermediate_endpoints: bool = False,
      **kwargs,
  ):
    """Initializes a UniversalInvertedBottleneckBlockQuantized.

    This is an extension of IB with optional depthwise convs before expansion (
    "starting" conv) and after projection ("ending" conv). Both of these convs
    are executed without activation. The standard depthwise conv of IB ("middle"
    conv) is optional too. This last one is followed by an activation, as in
    standard IBs. Squeeze-and-Excite or fused types of IBs are not supported.

    Args:
      in_filters: The number of filters of the input tensor.
      out_filters: The number of filters of the output tensor.
      expand_ratio: The filter multiplier for the first inverted bottleneck
        stage.
      strides: The block stride. If greater than 1, this block will ultimately
        downsample the input.
      middle_dw_downsample: If True, downsample in the middle depthwise
        otherwise downsample in the starting one.
      start_dw_kernel_size: The kernel size of the starting depthwise. A value
        of zero means that no starting depthwise will be added.
      middle_dw_kernel_size: The kernel size of the middle depthwise. A value of
        zero means that no middle depthwise will be added.
      end_dw_kernel_size: The kernel size of the ending depthwise. A value of
        zero means that no ending depthwise will be added.
      stochastic_depth_drop_rate: If not None, drop rate for the stochastic
        depth layer.
      kernel_initializer: The name of the convolutional layer
        kernel_initializer.
      kernel_regularizer: An optional kernel regularizer for the Conv2ds.
      bias_regularizer: An optional bias regularizer for the Conv2ds.
      activation: The name of the activation function.
      depthwise_activation: The name of the depthwise-only activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      dilation_rate: The dilation rate to use for convolutions.
      divisible_by: Ensures all inner dimensions are divisible by this number.
      regularize_depthwise: If True, apply regularization on depthwise.
      use_residual: If True, include residual connection between input and
        output.
      use_layer_scale: If True, use layer scale.
      layer_scale_init_value: The initial layer scale value.
      norm_momentum: Momentum value for the moving average in normalization.
      norm_epsilon: Value added to variance to avoid dividing by zero in
        normalization.
      output_intermediate_endpoints: This block does not output any intermediate
        endpoint, but this argument is included for compatibility with other
        blocks.
      **kwargs: Additional keyword arguments to be passed to
        tf_keras.layers.Layer.
    """
    super().__init__(**kwargs)
    logging.info(
        'UniversalInvertedBottleneckBlockQuantized with depthwise kernel sizes '
        '{%d, %d, %d}, strides=%d, and middle downsampling: %s',
        start_dw_kernel_size,
        middle_dw_kernel_size,
        end_dw_kernel_size,
        strides,
        middle_dw_downsample,
    )

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._expand_ratio = expand_ratio
    self._strides = strides
    self._middle_dw_downsample = middle_dw_downsample
    self._start_dw_kernel_size = start_dw_kernel_size
    self._middle_dw_kernel_size = middle_dw_kernel_size
    self._end_dw_kernel_size = end_dw_kernel_size
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._dilation_rate = dilation_rate
    self._use_sync_bn = use_sync_bn
    self._regularize_depthwise = regularize_depthwise
    self._use_residual = use_residual
    self._activation = activation
    self._depthwise_activation = depthwise_activation
    self._kernel_initializer = kernel_initializer
    self._use_layer_scale = use_layer_scale
    self._layer_scale_init_value = layer_scale_init_value
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._output_intermediate_endpoints = output_intermediate_endpoints

    if strides > 1:
      if middle_dw_downsample and not middle_dw_kernel_size:
        raise ValueError(
            'Requested downsampling at a non-existing middle depthwise conv.'
        )
      if not middle_dw_downsample and not start_dw_kernel_size:
        raise ValueError(
            'Requested downsampling at a non-existing starting depthwise conv.'
        )

    if use_sync_bn:
      norm_layer = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      norm_layer = tf_keras.layers.BatchNormalization
    self._norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    self._norm = helper.BatchNormalizationNoQuantized(norm_layer)

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if not depthwise_activation:
      self._depthwise_activation = activation
    if regularize_depthwise:
      self._depthwise_regularizer = kernel_regularizer
    else:
      self._depthwise_regularizer = None

  def build(self, input_shape):
    # Starting depthwise conv.
    if self._start_dw_kernel_size:
      self._start_dw_conv = helper.DepthwiseConv2DQuantized(
          kernel_size=self._start_dw_kernel_size,
          strides=self._strides if not self._middle_dw_downsample else 1,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer
          ),
          depthwise_regularizer=self._depthwise_regularizer,
          bias_regularizer=self._bias_regularizer,
      )
      # No activation -> quantized norm should be okay.
      self._start_dw_norm = self._norm_with_quantize(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )

    # Expansion with 1x1 convs.
    expand_filters = nn_layers.make_divisible(
        self._in_filters * self._expand_ratio, self._divisible_by
    )

    self._expand_conv = helper.Conv2DQuantized(
        filters=expand_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )
    self._expand_norm = helper.norm_by_activation(
        self._activation, self._norm_with_quantize, self._norm
    )(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
    )
    self._expand_act = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig(),
    )

    # Middle depthwise conv.
    if self._middle_dw_kernel_size:
      self._middle_dw_conv = helper.DepthwiseConv2DQuantized(
          kernel_size=self._middle_dw_kernel_size,
          strides=self._strides if self._middle_dw_downsample else 1,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer
          ),
          depthwise_regularizer=self._depthwise_regularizer,
          bias_regularizer=self._bias_regularizer,
      )
      self._middle_dw_norm = helper.norm_by_activation(
          self._activation, self._norm_with_quantize, self._norm
      )(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )
      self._middle_dw_act = tfmot.quantization.keras.QuantizeWrapperV2(
          tf_utils.get_activation(
              self._depthwise_activation, use_keras_layer=True
          ),
          configs.Default8BitActivationQuantizeConfig(),
      )

    # Projection with 1x1 convs.
    self._proj_conv = helper.Conv2DQuantized(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )
    # No activation -> quantized norm should be okay.
    self._proj_norm = self._norm_with_quantize(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
    )

    # Ending depthwise conv.
    if self._end_dw_kernel_size:
      self._end_dw_conv = helper.DepthwiseConv2DQuantized(
          kernel_size=self._end_dw_kernel_size,
          strides=1,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer
          ),
          depthwise_regularizer=self._depthwise_regularizer,
          bias_regularizer=self._bias_regularizer,
      )
      self._end_dw_norm = self._norm_with_quantize(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )

    if self._use_layer_scale:
      raise NotImplementedError

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate
      )
    else:
      self._stochastic_depth = None

    super().build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'expand_ratio': self._expand_ratio,
        'strides': self._strides,
        'middle_dw_downsample': self._middle_dw_downsample,
        'start_dw_kernel_size': self._start_dw_kernel_size,
        'middle_dw_kernel_size': self._middle_dw_kernel_size,
        'end_dw_kernel_size': self._end_dw_kernel_size,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'depthwise_activation': self._depthwise_activation,
        'dilation_rate': self._dilation_rate,
        'use_sync_bn': self._use_sync_bn,
        'regularize_depthwise': self._regularize_depthwise,
        'use_residual': self._use_residual,
        'use_layer_scale': self._use_layer_scale,
        'layer_scale_init_value': self._layer_scale_init_value,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'output_intermediate_endpoints': self._output_intermediate_endpoints,
    }
    base_config = super().get_config()
    return {**base_config, **config}

  def call(self, inputs, training=None):
    shortcut = inputs
    x = inputs
    if self._start_dw_kernel_size:
      x = self._start_dw_conv(x)
      x = self._start_dw_norm(x)

    x = self._expand_conv(x)
    x = self._expand_norm(x)
    x = self._expand_act(x)

    if self._middle_dw_kernel_size:
      x = self._middle_dw_conv(x)
      x = self._middle_dw_norm(x)
      x = self._middle_dw_act(x)

    x = self._proj_conv(x)
    x = self._proj_norm(x)

    if self._end_dw_kernel_size:
      x = self._end_dw_conv(x)
      x = self._end_dw_norm(x)

    if self._use_layer_scale:
      x = self._layer_scale(x)

    if (
        self._use_residual
        and self._in_filters == self._out_filters
        and self._strides == 1
    ):
      if self._stochastic_depth:
        x = self._stochastic_depth(x, training=training)
      x = x + shortcut

    # Return empty intermediate endpoints to be compatible with other blocks.
    if self._output_intermediate_endpoints:
      return x, {}
    return x


MaybeDwInvertedBottleneckBlockQuantized = (
    UniversalInvertedBottleneckBlockQuantized
)
