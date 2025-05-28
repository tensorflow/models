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

"""Contains common building blocks for neural networks."""

from typing import Sequence, Union

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.vision.modeling.layers import nn_layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class BasicBlock3DVolume(tf_keras.layers.Layer):
  """A basic 3d convolution block."""

  def __init__(self,
               filters: Union[int, Sequence[int]],
               strides: Union[int, Sequence[int]],
               kernel_size: Union[int, Sequence[int]],
               kernel_initializer: str = 'VarianceScaling',
               kernel_regularizer: tf_keras.regularizers.Regularizer = None,
               bias_regularizer: tf_keras.regularizers.Regularizer = None,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               use_batch_normalization: bool = False,  # pytype: disable=annotation-type-mismatch  # typed-keras
               **kwargs):
    """Creates a basic 3d convolution block applying one or more convolutions.

    Args:
      filters: A list of `int` numbers or an `int` number of filters. Given an
        `int` input, a single convolution is applied; otherwise a series of
        convolutions are applied.
      strides: An integer or tuple/list of 3 integers, specifying the strides of
        the convolution along each spatial dimension. Can be a single integer to
        specify the same value for all spatial dimensions.
      kernel_size: An integer or tuple/list of 3 integers, specifying the depth,
        height and width of the 3D convolution window. Can be a single integer
        to specify the same value for all spatial dimensions.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      use_batch_normalization: Wheher to use batch normalizaion or not.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)

    if isinstance(filters, int):
      self._filters = [filters]
    else:
      self._filters = filters
    self._strides = strides
    self._kernel_size = kernel_size
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._use_batch_normalization = use_batch_normalization

    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape: tf.TensorShape):
    """Builds the basic 3d convolution block."""
    self._convs = []
    self._norms = []
    for filters in self._filters:
      self._convs.append(
          tf_keras.layers.Conv3D(
              filters=filters,
              kernel_size=self._kernel_size,
              strides=self._strides,
              padding='same',
              data_format=tf_keras.backend.image_data_format(),
              activation=None))
      self._norms.append(
          self._norm(
              axis=self._bn_axis,
              momentum=self._norm_momentum,
              epsilon=self._norm_epsilon))

    super(BasicBlock3DVolume, self).build(input_shape)

  def get_config(self):
    """Returns the config of the basic 3d convolution block."""
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'use_batch_normalization': self._use_batch_normalization
    }
    base_config = super(BasicBlock3DVolume, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:  # pytype: disable=annotation-type-mismatch
    """Runs forward pass on the input tensor."""
    x = inputs
    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      if self._use_batch_normalization:
        x = norm(x)
      x = self._activation_fn(x)
    return x


@tf_keras.utils.register_keras_serializable(package='Vision')
class ResidualBlock3DVolume(tf_keras.layers.Layer):
  """A residual 3d block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               se_ratio=None,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A residual 3d block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: `float` or None. Ratio of the Squeeze-and-Excitation layer.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf_keras.layers.Conv3D(
          filters=self._filters,
          kernel_size=1,
          strides=self._strides,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    self._conv1 = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._conv2 = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters,
          out_filters=self._filters,
          se_ratio=self._se_ratio,
          use_3d_input=True,
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

    super(ResidualBlock3DVolume, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(ResidualBlock3DVolume, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    return self._activation_fn(x + shortcut)


@tf_keras.utils.register_keras_serializable(package='Vision')
class BottleneckBlock3DVolume(tf_keras.layers.Layer):
  """A standard bottleneck block."""

  def __init__(self,
               filters,
               strides,
               dilation_rate=1,
               use_projection=False,
               se_ratio=None,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A standard bottleneck 3d block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      dilation_rate: `int` dilation_rate of convolutions. Default to 1.
      use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: `float` or None. Ratio of the Squeeze-and-Excitation layer.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf_keras.layers.Conv3D(
          filters=self._filters * 4,
          kernel_size=1,
          strides=self._strides,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    self._conv1 = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._conv2 = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        dilation_rate=self._dilation_rate,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._conv3 = tf_keras.layers.Conv3D(
        filters=self._filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm3 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          use_3d_input=True,
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

    super(BottleneckBlock3DVolume, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'dilation_rate': self._dilation_rate,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(BottleneckBlock3DVolume, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._conv2(x)
    x = self._norm2(x)
    x = self._activation_fn(x)

    x = self._conv3(x)
    x = self._norm3(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    return self._activation_fn(x + shortcut)
