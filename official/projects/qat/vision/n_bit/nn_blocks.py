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

"""Contains quantized neural blocks for the QAT."""
from typing import Any, Dict, Optional, Sequence, Union

# Import libraries

from absl import logging
import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.modeling import tf_utils
from official.projects.qat.vision.n_bit import configs
from official.projects.qat.vision.n_bit import nn_layers as qat_nn_layers
from official.vision.modeling.layers import nn_layers


class NoOpActivation:
  """No-op activation which simply returns the incoming tensor.

  This activation is required to distinguish between `keras.activations.linear`
  which does the same thing. The main difference is that NoOpActivation should
  not have any quantize operation applied to it.
  """

  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    return x

  def get_config(self) -> Dict[str, Any]:
    """Get a config of this object."""
    return {}

  def __eq__(self, other: Any) -> bool:
    if not other or not isinstance(other, NoOpActivation):
      return False

    return True

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)


def _quantize_wrapped_layer(cls, quantize_config):
  def constructor(*arg, **kwargs):
    return tfmot.quantization.keras.QuantizeWrapperV2(
        cls(*arg, **kwargs),
        quantize_config)
  return constructor


# This class is copied from modeling.layers.nn_blocks.BottleneckBlock and apply
# QAT.
@tf.keras.utils.register_keras_serializable(package='Vision')
class BottleneckBlockNBitQuantized(tf.keras.layers.Layer):
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
               kernel_regularizer: tf.keras.regularizers.Regularizer = None,
               bias_regularizer: tf.keras.regularizers.Regularizer = None,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               bn_trainable: bool = True,
               num_bits_weight: int = 8,
               num_bits_activation: int = 8,  # pytype: disable=annotation-type-mismatch  # typed-keras
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
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      bn_trainable: A `bool` that indicates whether batch norm layers should be
        trainable. Default to True.
      num_bits_weight: An `int` number of bits for the weight. Default to 8.
      num_bits_activation: An `int` number of bits for the weight. Default to 8.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

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
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation
    if use_sync_bn:
      self._norm = _quantize_wrapped_layer(
          tf.keras.layers.experimental.SyncBatchNormalization,
          configs.NoOpQuantizeConfig())
      self._norm_with_quantize = _quantize_wrapped_layer(
          tf.keras.layers.experimental.SyncBatchNormalization,
          configs.DefaultNBitOutputQuantizeConfig(
              num_bits_weight=self._num_bits_weight,
              num_bits_activation=self._num_bits_activation))
    else:
      self._norm = _quantize_wrapped_layer(
          tf.keras.layers.BatchNormalization,
          configs.NoOpQuantizeConfig())
      self._norm_with_quantize = _quantize_wrapped_layer(
          tf.keras.layers.BatchNormalization,
          configs.DefaultNBitOutputQuantizeConfig(
              num_bits_weight=self._num_bits_weight,
              num_bits_activation=self._num_bits_activation))
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._bn_trainable = bn_trainable

  def build(self, input_shape: Optional[Union[Sequence[int], tf.Tensor]]):
    """Build variables and child layers to prepare for calling."""
    conv2d_quantized = _quantize_wrapped_layer(
        tf.keras.layers.Conv2D,
        configs.DefaultNBitConvQuantizeConfig(
            ['kernel'], ['activation'], False,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    if self._use_projection:
      if self._resnetd_shortcut:
        self._shortcut0 = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=self._strides, padding='same')
        self._shortcut1 = conv2d_quantized(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=NoOpActivation())
      else:
        self._shortcut = conv2d_quantized(
            filters=self._filters * 4,
            kernel_size=1,
            strides=self._strides,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=NoOpActivation())

      self._norm0 = self._norm_with_quantize(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          trainable=self._bn_trainable)

    self._conv1 = conv2d_quantized(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation1 = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.DefaultNBitActivationQuantizeConfig(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))

    self._conv2 = conv2d_quantized(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        dilation_rate=self._dilation_rate,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation2 = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.DefaultNBitActivationQuantizeConfig(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))

    self._conv3 = conv2d_quantized(
        filters=self._filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())
    self._norm3 = self._norm_with_quantize(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation3 = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.DefaultNBitActivationQuantizeConfig(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = qat_nn_layers.SqueezeExcitationNBitQuantized(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          num_bits_weight=self._num_bits_weight,
          num_bits_activation=self._num_bits_activation)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tfmot.quantization.keras.QuantizeWrapperV2(
        tf.keras.layers.Add(),
        configs.DefaultNBitQuantizeConfig(
            [], [], True,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))

    super().build(input_shape)

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
        'bn_trainable': self._bn_trainable,
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation
    }
    base_config = super().get_config()
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
@tf.keras.utils.register_keras_serializable(package='Vision')
class Conv2DBNBlockNBitQuantized(tf.keras.layers.Layer):
  """A quantized convolution block with batch normalization."""

  def __init__(
      self,
      filters: int,
      kernel_size: int = 3,
      strides: int = 1,
      use_bias: bool = False,
      activation: str = 'relu6',
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_normalization: bool = True,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      num_bits_weight: int = 8,
      num_bits_activation: int = 8,
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
      activation: A `str` name of the activation function.
      kernel_initializer: A `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      use_normalization: If True, use batch normalization.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      num_bits_weight: An `int` number of bits for the weight. Default to 8.
      num_bits_activation: An `int` number of bits for the weight. Default to 8.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_normalization = use_normalization
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

    if use_sync_bn:
      self._norm = _quantize_wrapped_layer(
          tf.keras.layers.experimental.SyncBatchNormalization,
          configs.NoOpQuantizeConfig())
    else:
      self._norm = _quantize_wrapped_layer(
          tf.keras.layers.BatchNormalization,
          configs.NoOpQuantizeConfig())
    if tf.keras.backend.image_data_format() == 'channels_last':
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
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'use_normalization': self._use_normalization,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape: Optional[Union[Sequence[int], tf.Tensor]]):
    """Build variables and child layers to prepare for calling."""
    conv2d_quantized = _quantize_wrapped_layer(
        tf.keras.layers.Conv2D,
        configs.DefaultNBitConvQuantizeConfig(
            ['kernel'], ['activation'], False,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    self._conv0 = conv2d_quantized(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding='same',
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())
    if self._use_normalization:
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
    self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.DefaultNBitActivationQuantizeConfig(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))

    super(Conv2DBNBlockNBitQuantized, self).build(input_shape)

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[Union[bool, tf.Tensor]] = None) -> tf.Tensor:
    """Run the Conv2DBNBlockNBitQuantized logics."""
    x = self._conv0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    return self._activation_layer(x)


@tf.keras.utils.register_keras_serializable(package='Vision')
class InvertedBottleneckBlockNBitQuantized(tf.keras.layers.Layer):
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
               num_bits_weight: int = 8,
               num_bits_activation: int = 8,
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
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      se_inner_activation: A `str` name of squeeze-excitation inner activation.
      se_gating_activation: A `str` name of squeeze-excitation gating
        activation.
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
      num_bits_weight: An `int` number of bits for the weight. Default to 8.
      num_bits_activation: An `int` number of bits for the weight. Default to 8.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

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
    self._depthwise_activation = depthwise_activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._expand_se_in_filters = expand_se_in_filters
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

    if use_sync_bn:
      self._norm = _quantize_wrapped_layer(
          tf.keras.layers.experimental.SyncBatchNormalization,
          configs.NoOpQuantizeConfig())
      self._norm_with_quantize = _quantize_wrapped_layer(
          tf.keras.layers.experimental.SyncBatchNormalization,
          configs.DefaultNBitOutputQuantizeConfig(
              num_bits_weight=self._num_bits_weight,
              num_bits_activation=self._num_bits_activation))
    else:
      self._norm = _quantize_wrapped_layer(
          tf.keras.layers.BatchNormalization,
          configs.NoOpQuantizeConfig())
      self._norm_with_quantize = _quantize_wrapped_layer(
          tf.keras.layers.BatchNormalization,
          configs.DefaultNBitOutputQuantizeConfig(
              num_bits_weight=self._num_bits_weight,
              num_bits_activation=self._num_bits_activation))
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if not depthwise_activation:
      self._depthwise_activation = activation
    if regularize_depthwise:
      self._depthsize_regularizer = kernel_regularizer
    else:
      self._depthsize_regularizer = None

  def build(self, input_shape: Optional[Union[Sequence[int], tf.Tensor]]):
    """Build variables and child layers to prepare for calling."""
    conv2d_quantized = _quantize_wrapped_layer(
        tf.keras.layers.Conv2D,
        configs.DefaultNBitConvQuantizeConfig(
            ['kernel'], ['activation'], False,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    depthwise_conv2d_quantized = _quantize_wrapped_layer(
        tf.keras.layers.DepthwiseConv2D,
        configs.DefaultNBitConvQuantizeConfig(
            ['depthwise_kernel'], ['activation'], False,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    expand_filters = self._in_filters
    if self._expand_ratio > 1:
      # First 1x1 conv for channel expansion.
      expand_filters = nn_layers.make_divisible(
          self._in_filters * self._expand_ratio, self._divisible_by)

      expand_kernel = 1 if self._use_depthwise else self._kernel_size
      expand_stride = 1 if self._use_depthwise else self._strides

      self._conv0 = conv2d_quantized(
          filters=expand_filters,
          kernel_size=expand_kernel,
          strides=expand_stride,
          padding='same',
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=NoOpActivation())
      self._norm0 = self._norm_with_quantize(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
      self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
          tf_utils.get_activation(self._activation, use_keras_layer=True),
          configs.DefaultNBitActivationQuantizeConfig(
              num_bits_weight=self._num_bits_weight,
              num_bits_activation=self._num_bits_activation))

    if self._use_depthwise:
      # Depthwise conv.
      self._conv1 = depthwise_conv2d_quantized(
          kernel_size=(self._kernel_size, self._kernel_size),
          strides=self._strides,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=self._kernel_initializer,
          depthwise_regularizer=self._depthsize_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=NoOpActivation())
      self._norm1 = self._norm_with_quantize(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
      self._depthwise_activation_layer = (
          tfmot.quantization.keras.QuantizeWrapperV2(
              tf_utils.get_activation(self._depthwise_activation,
                                      use_keras_layer=True),
              configs.DefaultNBitActivationQuantizeConfig(
                  num_bits_weight=self._num_bits_weight,
                  num_bits_activation=self._num_bits_activation)))

    # Squeeze and excitation.
    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      logging.info('Use Squeeze and excitation.')
      in_filters = self._in_filters
      if self._expand_se_in_filters:
        in_filters = expand_filters
      self._squeeze_excitation = qat_nn_layers.SqueezeExcitationNBitQuantized(
          in_filters=in_filters,
          out_filters=expand_filters,
          se_ratio=self._se_ratio,
          divisible_by=self._divisible_by,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._se_inner_activation,
          gating_activation=self._se_gating_activation,
          num_bits_weight=self._num_bits_weight,
          num_bits_activation=self._num_bits_activation)
    else:
      self._squeeze_excitation = None

    # Last 1x1 conv.
    self._conv2 = conv2d_quantized(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())
    self._norm2 = self._norm_with_quantize(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tf.keras.layers.Add()

    super().build(input_shape)

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
        'expand_se_in_filters': self._expand_se_in_filters,
        'depthwise_activation': self._depthwise_activation,
        'dilation_rate': self._dilation_rate,
        'use_sync_bn': self._use_sync_bn,
        'regularize_depthwise': self._regularize_depthwise,
        'use_depthwise': self._use_depthwise,
        'use_residual': self._use_residual,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[Union[bool, tf.Tensor]] = None) -> tf.Tensor:
    """Run the InvertedBottleneckBlockNBitQuantized logics."""
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

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if (self._use_residual and
        self._in_filters == self._out_filters and
        self._strides == 1):
      if self._stochastic_depth:
        x = self._stochastic_depth(x, training=training)
      x = self._add([x, shortcut])

    return x
