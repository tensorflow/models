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
"""Contains common building blocks for neural networks."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import libraries

import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers


def _pad_strides(strides: int, axis: int) -> Tuple[int, int, int, int]:
  """Converts int to len 4 strides (`tf.nn.avg_pool` uses length 4)."""
  if axis == 1:
    return (1, 1, strides, strides)
  else:
    return (1, strides, strides, 1)


def _maybe_downsample(x: tf.Tensor,
                      out_filter: int,
                      strides: int,
                      axis: int) -> tf.Tensor:
  """Downsamples feature map and 0-pads tensor if in_filter != out_filter."""
  data_format = 'NCHW' if axis == 1 else 'NHWC'
  strides = _pad_strides(strides, axis=axis)

  x = tf.nn.avg_pool(x, strides, strides, 'VALID', data_format=data_format)

  in_filter = x.shape[axis]
  if in_filter < out_filter:
    # Pad on channel dimension with 0s: half on top half on bottom.
    pad_size = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
    if axis == 1:
      x = tf.pad(x, [[0, 0], pad_size, [0, 0], [0, 0]])
    else:
      x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_size])

  return x + 0.


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResidualBlock(tf.keras.layers.Layer):
  """A residual block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A residual block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(ResidualBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf.keras.layers.Conv2D(
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

    self._conv1 = tf.keras.layers.Conv2D(
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

    self._conv2 = tf.keras.layers.Conv2D(
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

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    super(ResidualBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(ResidualBlock, self).get_config()
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

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    return self._activation_fn(x + shortcut)


@tf.keras.utils.register_keras_serializable(package='Vision')
class BottleneckBlock(tf.keras.layers.Layer):
  """A standard bottleneck block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A standard bottleneck block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(BottleneckBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf.keras.layers.Conv2D(
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

    self._conv1 = tf.keras.layers.Conv2D(
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

    self._conv2 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._conv3 = tf.keras.layers.Conv2D(
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

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    super(BottleneckBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(BottleneckBlock, self).get_config()
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

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    return self._activation_fn(x + shortcut)


@tf.keras.utils.register_keras_serializable(package='Vision')
class InvertedBottleneckBlock(tf.keras.layers.Layer):
  """An inverted bottleneck block."""

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
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """An inverted bottleneck block with BN after convolutions.

    Args:
      in_filters: `int` number of filters of the input tensor.
      out_filters: `int` number of filters of the output tensor.
      expand_ratio: `int` expand_ratio for an inverted bottleneck block.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      kernel_size: `int` kernel_size of the depthwise conv layer.
      se_ratio: `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(InvertedBottleneckBlock, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._expand_ratio = expand_ratio
    self._strides = strides
    self._kernel_size = kernel_size
    self._se_ratio = se_ratio
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    if self._expand_ratio != 1:
      # First 1x1 conv for channel expansion.
      self._conv0 = tf.keras.layers.Conv2D(
          filters=self._in_filters * self._expand_ratio,
          kernel_size=1,
          strides=1,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    # Depthwise conv.
    self._conv1 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(self._kernel_size, self._kernel_size),
        strides=self._strides,
        padding='same',
        use_bias=False,
        depthwise_initializer=self._kernel_initializer,
        depthwise_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    # Squeeze and excitation.
    if self._se_ratio is not None and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._in_filters,
          se_ratio=self._se_ratio,
          expand_ratio=self._expand_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    # Last 1x1 conv.
    self._conv2 = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    super(InvertedBottleneckBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'expand_ratio': self._expand_ratio,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
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
    base_config = super(InvertedBottleneckBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._expand_ratio != 1:
      x = self._conv0(inputs)
      x = self._norm0(x)
      x = self._activation_fn(x)
    else:
      x = inputs

    x = self._conv1(x)
    x = self._norm1(x)
    x = self._activation_fn(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if self._in_filters == self._out_filters and self._strides == 1:
      if self._stochastic_depth:
        x = self._stochastic_depth(x, training=training)
      x = tf.add(x, shortcut)

    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResidualInner(tf.keras.layers.Layer):
  """Single inner block of a residual.

  This corresponds to `F`/`G` functions in the RevNet paper:
  https://arxiv.org/pdf/1707.04585.pdf
  """

  def __init__(
      self,
      filters: int,
      strides: int,
      kernel_initializer: Union[
          str, Callable[..., tf.keras.initializers.Initializer]]
      = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activation: Union[str, Callable[..., tf.Tensor]] = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      batch_norm_first: bool = True,
      **kwargs):
    """ResidualInner Initialization.

    Args:
      filters: `int` output filter size.
      strides: `int` stride size for convolution for the residual block.
      kernel_initializer: `str` or `tf.keras.initializers.Initializer` instance
        for convolutional layers.
      kernel_regularizer: `tf.keras.regularizers.Regularizer` for Conv2D.
      activation: `str` or `callable` instance of the activation function.
      use_sync_bn: `bool` if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      batch_norm_first: `bool` whether to apply activation and batch norm
        before conv.
      **kwargs: additional keyword arguments to be passed.
    """
    super(ResidualInner, self).__init__(**kwargs)

    self.strides = strides
    self.filters = filters
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._kernel_regularizer = kernel_regularizer
    self._activation = tf.keras.activations.get(activation)
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._batch_norm_first = batch_norm_first

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape: tf.TensorShape):
    if self._batch_norm_first:
      self._batch_norm_0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    self._conv2d_1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=3,
        strides=self.strides,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)

    self._batch_norm_1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._conv2d_2 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)

    super(ResidualInner, self).build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    config = {
        'filters': self.filters,
        'strides': self.strides,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'batch_norm_first': self._batch_norm_first,
    }
    base_config = super(ResidualInner, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
    x = inputs
    if self._batch_norm_first:
      x = self._batch_norm_0(x, training=training)
      x = self._activation_fn(x)
    x = self._conv2d_1(x)

    x = self._batch_norm_1(x, training=training)
    x = self._activation_fn(x)
    x = self._conv2d_2(x)
    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class BottleneckResidualInner(tf.keras.layers.Layer):
  """Single inner block of a bottleneck residual.

  This corresponds to `F`/`G` functions in the RevNet paper:
  https://arxiv.org/pdf/1707.04585.pdf
  """

  def __init__(
      self,
      filters: int,
      strides: int,
      kernel_initializer: Union[
          str, Callable[..., tf.keras.initializers.Initializer]]
      = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activation: Union[str, Callable[..., tf.Tensor]] = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      batch_norm_first: bool = True,
      **kwargs):
    """BottleneckResidualInner Initialization.

    Args:
      filters: `int` number of filters for first 2 convolutions. Last
        Last, and thus the number of output channels from the bottlneck
        block is `4*filters`
      strides: `int` stride size for convolution for the residual block.
      kernel_initializer: `str` or `tf.keras.initializers.Initializer` instance
        for convolutional layers.
      kernel_regularizer: `tf.keras.regularizers.Regularizer` for Conv2D.
      activation: `str` or `callable` instance of the activation function.
      use_sync_bn: `bool` if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      batch_norm_first: `bool` whether to apply activation and batch norm
        before conv.
      **kwargs: additional keyword arguments to be passed.
    """
    super(BottleneckResidualInner, self).__init__(**kwargs)

    self.strides = strides
    self.filters = filters
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._kernel_regularizer = kernel_regularizer
    self._activation = tf.keras.activations.get(activation)
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._batch_norm_first = batch_norm_first

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape: tf.TensorShape):
    if self._batch_norm_first:
      self._batch_norm_0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
    self._conv2d_1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=self.strides,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)
    self._batch_norm_1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)
    self._conv2d_2 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)
    self._batch_norm_2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)
    self._conv2d_3 = tf.keras.layers.Conv2D(
        filters=self.filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)

    super(BottleneckResidualInner, self).build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    config = {
        'filters': self.filters,
        'strides': self.strides,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'batch_norm_first': self._batch_norm_first,
    }
    base_config = super(BottleneckResidualInner, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
    x = inputs
    if self._batch_norm_first:
      x = self._batch_norm_0(x, training=training)
      x = self._activation_fn(x)
    x = self._conv2d_1(x)

    x = self._batch_norm_1(x, training=training)
    x = self._activation_fn(x)
    x = self._conv2d_2(x)

    x = self._batch_norm_2(x, training=training)
    x = self._activation_fn(x)
    x = self._conv2d_3(x)

    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class ReversibleLayer(tf.keras.layers.Layer):
  """A reversible layer.

  Computes y1 = x1 + f(x2), y2 = x2 + g(y1), where f and g can be arbitrary
  layers that are stateless, which in this case are `ResidualInner` layers.
  """

  def __init__(self,
               f: tf.keras.layers.Layer,
               g: tf.keras.layers.Layer,
               manual_grads: bool = True,
               **kwargs):
    """ReversibleLayer Initialization.

    Args:
      f: `tf.keras.layers.Layer` f inner block referred to in paper. Each
        reversible layer consists of two inner functions. For example, in RevNet
        the reversible residual consists of two f/g inner (bottleneck) residual
        functions. Where the input to the reversible layer is x, the input gets
        partitioned in the channel dimension and the forward pass follows (eq8):
        x = [x1; x2], z1 = x1 + f(x2), y2 = x2 + g(z1), y1 = stop_gradient(z1).
      g: `tf.keras.layers.Layer` g inner block referred to in paper. Detailed
        explanation same as above as `f` arg.
      manual_grads: `bool` [Testing Only] whether to manually take gradients
        as in Algorithm 1 or defer to autograd.
      **kwargs: additional keyword arguments to be passed.
    """
    super(ReversibleLayer, self).__init__(**kwargs)

    self._f = f
    self._g = g
    self._manual_grads = manual_grads

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._axis = -1
    else:
      self._axis = 1

  def get_config(self) -> Dict[str, Any]:
    config = {
        'f': self._f,
        'g': self._g,
        'manual_grads': self._manual_grads,
    }
    base_config = super(ReversibleLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _ckpt_non_trainable_vars(self):
    self._f_non_trainable_vars = [
        v.read_value() for v in self._f.non_trainable_variables]
    self._g_non_trainable_vars = [
        v.read_value() for v in self._g.non_trainable_variables]

  def _load_ckpt_non_trainable_vars(self):
    for v, v_chkpt in zip(
        self._f.non_trainable_variables, self._f_non_trainable_vars):
      v.assign(v_chkpt)
    for v, v_chkpt in zip(
        self._g.non_trainable_variables, self._g_non_trainable_vars):
      v.assign(v_chkpt)

  def call(
      self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:

    @tf.custom_gradient
    def reversible(x: tf.Tensor) -> Tuple[
        tf.Tensor, Callable[[Any], Tuple[List[tf.Tensor], List[tf.Tensor]]]]:
      """Implements Algorithm 1 in RevNet paper.

      Paper: https://arxiv.org/pdf/1707.04585.pdf

      Args:
        x: input tensor.

      Returns:
        y: the output [y1; y2] in algorithm 1.
        grad_fn: callable function that computes the gradients.
      """
      with tf.GradientTape() as fwdtape:
        fwdtape.watch(x)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self._axis)
        f_x2 = self._f(x2, training=training)
        x1_down = _maybe_downsample(
            x1, f_x2.shape[self._axis], self._f.strides, self._axis)
        z1 = f_x2 + x1_down
        g_z1 = self._g(z1, training=training)
        x2_down = _maybe_downsample(
            x2, g_z1.shape[self._axis], self._f.strides, self._axis)
        y2 = x2_down + g_z1

        # Equation 8: https://arxiv.org/pdf/1707.04585.pdf
        # Decouple y1 and z1 so that their derivatives are different.
        y1 = tf.identity(z1)
        y = tf.concat([y1, y2], axis=self._axis)

        irreversible = (
            (self._f.strides != 1 or self._g.strides != 1)
            or (y.shape[self._axis] != inputs.shape[self._axis]))

        # Checkpointing moving mean/variance for batch normalization layers
        # as they shouldn't be updated during the custom gradient pass of f/g.
        self._ckpt_non_trainable_vars()

      def grad_fn(dy: tf.Tensor,
                  variables: Optional[List[tf.Variable]] = None,
                 ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """Given dy calculate (dy/dx)|_{x_{input}} using f/g."""
        if irreversible or not self._manual_grads:
          grads_combined = fwdtape.gradient(
              y, [x] + variables, output_gradients=dy)
          dx = grads_combined[0]
          grad_vars = grads_combined[1:]
        else:
          y1_nograd = tf.stop_gradient(y1)
          y2_nograd = tf.stop_gradient(y2)
          dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self._axis)

          # Index mapping from self.f/g.trainable_variables to grad_fn
          # input `variables` kwarg so that we can reorder dwf + dwg
          # variable gradient list to match `variables` order.
          f_var_refs = [v.ref() for v in self._f.trainable_variables]
          g_var_refs = [v.ref() for v in self._g.trainable_variables]
          fg_var_refs = f_var_refs + g_var_refs
          self_to_var_index = [fg_var_refs.index(v.ref()) for v in variables]

          # Algorithm 1 in paper (line # documented in-line)
          z1 = y1_nograd                                         # line 2
          with tf.GradientTape() as gtape:
            gtape.watch(z1)
            g_z1 = self._g(z1, training=training)
          x2 = y2_nograd - g_z1                                  # line 3

          with tf.GradientTape() as ftape:
            ftape.watch(x2)
            f_x2 = self._f(x2, training=training)
          x1 = z1 - f_x2  # pylint: disable=unused-variable      # line 4

          # Compute gradients
          g_grads_combined = gtape.gradient(
              g_z1,
              [z1] + self._g.trainable_variables,
              output_gradients=dy2)
          dz1 = dy1 + g_grads_combined[0]                        # line 5
          dwg = g_grads_combined[1:]                             # line 9

          f_grads_combined = ftape.gradient(
              f_x2,
              [x2] + self._f.trainable_variables,
              output_gradients=dz1)
          dx2 = dy2 + f_grads_combined[0]                        # line 6
          dwf = f_grads_combined[1:]                             # line 8
          dx1 = dz1                                              # line 7

          # Pack the input and variable gradients.
          dx = tf.concat([dx1, dx2], axis=self._axis)
          grad_vars = dwf + dwg
          # Reorder gradients (trainable_variables to variables kwarg order)
          grad_vars = [grad_vars[i] for i in self_to_var_index]

          # Restore batch normalization moving mean/variance for correctness.
          self._load_ckpt_non_trainable_vars()

        return dx, grad_vars  # grad_fn end

      return y, grad_fn  # reversible end

    activations = reversible(inputs)
    return activations
