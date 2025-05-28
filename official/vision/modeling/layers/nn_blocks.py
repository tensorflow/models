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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Text

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.nlp import modeling as nlp_modeling
from official.vision.modeling.layers import nn_layers


def _pad_strides(strides: int, axis: int) -> Tuple[int, int, int, int]:
  """Converts int to len 4 strides (`tf.nn.avg_pool` uses length 4)."""
  if axis == 1:
    return (1, 1, strides, strides)
  else:
    return (1, strides, strides, 1)


def _maybe_downsample(x: tf.Tensor, out_filter: int, strides: int,
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


@tf_keras.utils.register_keras_serializable(package='Vision')
class ResidualBlock(tf_keras.layers.Layer):
  """A residual block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               se_ratio=None,
               resnetd_shortcut=False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_explicit_padding: bool = False,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               bn_trainable=True,
               **kwargs):
    """Initializes a residual block with BN after convolutions.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      strides: An `int` block stride. If greater than 1, this block will
        ultimately downsample the input.
      use_projection: A `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      resnetd_shortcut: A `bool` if True, apply the resnetd style modification
        to the shortcut connection. Not implemented in residual blocks.
      stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      bn_trainable: A `bool` that indicates whether batch norm layers should be
        trainable. Default to True.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ResidualBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._resnetd_shortcut = resnetd_shortcut
    self._use_explicit_padding = use_explicit_padding
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)
    self._bn_trainable = bn_trainable

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf_keras.layers.Conv2D(
          filters=self._filters,
          kernel_size=1,
          strides=self._strides,
          use_bias=False,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          trainable=self._bn_trainable,
          synchronized=self._use_sync_bn,
      )

    conv1_padding = 'same'
    # explicit padding here is added for centernet
    if self._use_explicit_padding:
      self._pad = tf_keras.layers.ZeroPadding2D(padding=(1, 1))
      conv1_padding = 'valid'

    self._conv1 = tf_keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding=conv1_padding,
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )

    self._conv2 = tf_keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters,
          out_filters=self._filters,
          se_ratio=self._se_ratio,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

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
        'se_ratio': self._se_ratio,
        'resnetd_shortcut': self._resnetd_shortcut,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_explicit_padding': self._use_explicit_padding,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'bn_trainable': self._bn_trainable
    }
    base_config = super(ResidualBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    if self._use_explicit_padding:
      inputs = self._pad(inputs)
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
class BottleneckBlock(tf_keras.layers.Layer):
  """A standard bottleneck block."""

  def __init__(self,
               filters,
               strides,
               dilation_rate=1,
               use_projection=False,
               se_ratio=None,
               resnetd_shortcut=False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               bn_trainable=True,
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
    super(BottleneckBlock, self).__init__(**kwargs)

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
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._bn_trainable = bn_trainable

  def build(self, input_shape):
    if self._use_projection:
      if self._resnetd_shortcut:
        self._shortcut0 = tf_keras.layers.AveragePooling2D(
            pool_size=2, strides=self._strides, padding='same')
        self._shortcut1 = tf_keras.layers.Conv2D(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
      else:
        self._shortcut = tf_keras.layers.Conv2D(
            filters=self._filters * 4,
            kernel_size=1,
            strides=self._strides,
            use_bias=False,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)

      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          trainable=self._bn_trainable,
          synchronized=self._use_sync_bn,
      )

    self._conv1 = tf_keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )
    self._activation1 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    self._conv2 = tf_keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        dilation_rate=self._dilation_rate,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )
    self._activation2 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    self._conv3 = tf_keras.layers.Conv2D(
        filters=self._filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm3 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )
    self._activation3 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tf_keras.layers.Add()

    super(BottleneckBlock, self).build(input_shape)

  def get_config(self):
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
    base_config = super(BottleneckBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
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


@tf_keras.utils.register_keras_serializable(package='Vision')
class InvertedBottleneckBlock(tf_keras.layers.Layer):
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
        this number. dilated convolution: An `int` to specify the same value for
        all spatial dimensions.
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
    super(InvertedBottleneckBlock, self).__init__(**kwargs)

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
    self._se_round_down_protect = se_round_down_protect
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._expand_se_in_filters = expand_se_in_filters
    self._output_intermediate_endpoints = output_intermediate_endpoints
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if not depthwise_activation:
      self._depthwise_activation = activation
    if regularize_depthwise:
      self._depthsize_regularizer = kernel_regularizer
    else:
      self._depthsize_regularizer = None

  def build(self, input_shape):
    # First 1x1 conv for channel expansion.
    expand_filters = nn_layers.make_divisible(
        self._in_filters * self._expand_ratio, self._divisible_by
    )

    expand_kernel = 1 if self._use_depthwise else self._kernel_size
    expand_stride = 1 if self._use_depthwise else self._strides

    self._conv0 = tf_keras.layers.Conv2D(
        filters=expand_filters,
        kernel_size=expand_kernel,
        strides=expand_stride,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )
    self._activation_layer = tf_utils.get_activation(
        self._activation, use_keras_layer=True
    )

    if self._use_depthwise:
      # Depthwise conv.
      self._conv1 = tf_keras.layers.DepthwiseConv2D(
          kernel_size=(self._kernel_size, self._kernel_size),
          strides=self._strides,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          depthwise_regularizer=self._depthsize_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm1 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          synchronized=self._use_sync_bn,
      )
      self._depthwise_activation_layer = tf_utils.get_activation(
          self._depthwise_activation, use_keras_layer=True)

    # Squeeze and excitation.
    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      logging.info('Use Squeeze and excitation.')
      in_filters = self._in_filters
      if self._expand_se_in_filters:
        in_filters = expand_filters
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=in_filters,
          out_filters=expand_filters,
          se_ratio=self._se_ratio,
          divisible_by=self._divisible_by,
          round_down_protect=self._se_round_down_protect,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._se_inner_activation,
          gating_activation=self._se_gating_activation)
    else:
      self._squeeze_excitation = None

    # Last 1x1 conv.
    self._conv2 = tf_keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tf_keras.layers.Add()

    super(InvertedBottleneckBlock, self).build(input_shape)

  def get_config(self):
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
    base_config = super(InvertedBottleneckBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
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
class UniversalInvertedBottleneckBlock(tf_keras.layers.Layer):
  """An inverted bottleneck block with optional depthwises."""

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
    """Initializes a UniversalInvertedBottleneckBlock.

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
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    logging.info(
        'UniversalInvertedBottleneckBlock with depthwise kernel sizes '
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
            'Requested downsampling at a non-existing middle depthwise.'
        )
      if not middle_dw_downsample and not start_dw_kernel_size:
        raise ValueError(
            'Requested downsampling at a non-existing starting depthwise.'
        )

    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if not depthwise_activation:
      self._depthwise_activation = activation
    if regularize_depthwise:
      self._depthsize_regularizer = kernel_regularizer
    else:
      self._depthsize_regularizer = None

  def build(self, input_shape):
    # Starting depthwise conv.
    if self._start_dw_kernel_size:
      self._start_dw_conv = tf_keras.layers.DepthwiseConv2D(
          kernel_size=self._start_dw_kernel_size,
          strides=self._strides if not self._middle_dw_downsample else 1,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer
          ),
          depthwise_regularizer=self._depthsize_regularizer,
          bias_regularizer=self._bias_regularizer,
      )
      self._start_dw_norm = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )

    # Expansion with 1x1 convs.
    expand_filters = nn_layers.make_divisible(
        self._in_filters * self._expand_ratio, self._divisible_by
    )

    self._expand_conv = tf_keras.layers.Conv2D(
        filters=expand_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )
    self._expand_norm = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
    )
    self._expand_act = tf_utils.get_activation(
        self._activation, use_keras_layer=True
    )

    # Middle depthwise conv.
    if self._middle_dw_kernel_size:
      self._middle_dw_conv = tf_keras.layers.DepthwiseConv2D(
          kernel_size=self._middle_dw_kernel_size,
          strides=self._strides if self._middle_dw_downsample else 1,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer
          ),
          depthwise_regularizer=self._depthsize_regularizer,
          bias_regularizer=self._bias_regularizer,
      )
      self._middle_dw_norm = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )
      self._middle_dw_act = tf_utils.get_activation(
          self._depthwise_activation, use_keras_layer=True
      )

    # Projection with 1x1 convs.
    self._proj_conv = tf_keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )
    self._proj_norm = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
    )

    # Ending depthwise conv.
    if self._end_dw_kernel_size:
      self._end_dw_conv = tf_keras.layers.DepthwiseConv2D(
          kernel_size=self._end_dw_kernel_size,
          strides=1,
          padding='same',
          depth_multiplier=1,
          dilation_rate=self._dilation_rate,
          use_bias=False,
          depthwise_initializer=tf_utils.clone_initializer(
              self._kernel_initializer
          ),
          depthwise_regularizer=self._depthsize_regularizer,
          bias_regularizer=self._bias_regularizer,
      )
      self._end_dw_norm = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )

    if self._use_layer_scale:
      self._layer_scale = MNV4LayerScale(self._layer_scale_init_value)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate
      )
    else:
      self._stochastic_depth = None

    super().build(input_shape)

  def get_config(self) -> dict[str, Any]:
    """Return a Python dict containing this layer's configuration data."""
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
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    """Run layer computation."""
    endpoints = {}
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

    if self._output_intermediate_endpoints:
      return x, endpoints
    return x


class MultiQueryAttentionLayerV1(tf_keras.layers.Layer):
  """Multi Query Attention.

  Fast Transformer Decoding: One Write-Head is All You Need
  https://arxiv.org/pdf/1911.02150.pdf

  This gives 2x speed up compared to vanilla multihead attention at the cost
  of negligible precision drop.
  """

  def __init__(self, num_heads, key_dim, value_dim, dropout=0):
    """Initializer."""
    super().__init__()
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._dropout = dropout

  def build(self, input_shape):
    """Create layer state."""
    x_shape, m_shape = input_shape
    self._channel_dim = x_shape[-1]
    assert self._channel_dim == m_shape[-1], f'x={x_shape}, m={m_shape}'
    # Note: weight initializers are left to default
    self._query_proj = self.add_weight(
        'query', [self._num_heads, self._channel_dim, self._key_dim]
    )
    self._key_proj = self.add_weight('key', [self._channel_dim, self._key_dim])
    self._value_proj = self.add_weight(
        'value', [self._channel_dim, self._value_dim]
    )
    self._output_proj = self.add_weight(
        'output', [self._num_heads, self._channel_dim, self._value_dim]
    )
    self._dropout_layer = tf_keras.layers.Dropout(rate=self._dropout)

  def _reshape_input(self, t):
    """Reshapes a tensor to three dimensions, keeping the first and last."""
    s = tf.shape(t)
    # Propagate the shape statically where possible.
    static_num = t.shape[1:-1].num_elements()
    num = static_num or tf.math.reduce_prod(s[1:-1])
    return tf.ensure_shape(
        tf.reshape(t, [s[0], num, s[-1]]), [t.shape[0], static_num, t.shape[-1]]
    )

  def call(self, inputs, optimize_einsum=False):
    """Run layer computation."""
    x, m = inputs

    reshaped_x = self._reshape_input(x)
    reshaped_m = self._reshape_input(m)

    if optimize_einsum:
      logits = tf.einsum(
          'bnd,bme,hdk,ek->bhnm',
          reshaped_x,
          reshaped_m,
          self._query_proj,
          self._key_proj,
          optimize='optimal',
      )
    else:
      q = tf.einsum('bnd,hdk->bhnk', reshaped_x, self._query_proj)
      k = tf.einsum('bmd,dk->bmk', reshaped_m, self._key_proj)
      logits = tf.einsum('bhnk,bmk->bhnm', q, k)

    logits = logits / tf.math.sqrt(tf.cast(self._key_dim, x.dtype))
    attention_scores = self._dropout_layer(tf.nn.softmax(logits))

    if optimize_einsum:
      result = tf.einsum(
          'bhnm,bmd,dv,hev->bne',
          attention_scores,
          reshaped_m,
          self._value_proj,
          self._output_proj,
          optimize='optimal',
      )
    else:
      v = tf.einsum('bmd,dv->bmv', reshaped_m, self._value_proj)
      o = tf.einsum('bhnm,bmv->bhnv', attention_scores, v)
      result = tf.einsum('bhnv,hdv->bnd', o, self._output_proj)

    return tf.ensure_shape(tf.reshape(result, tf.shape(x)), x.shape)


class MultiQueryAttentionLayerV2(tf_keras.layers.Layer):
  """Multi Query Attention.

  Fast Transformer Decoding: One Write-Head is All You Need
  https://arxiv.org/pdf/1911.02150.pdf

  This is an acceletor optimized version - removing multiple unneccessary
  tensor transpose by re-arranging indices according to the following rules: 1)
  contracted indices are at the end, 2) other indices have the same order in the
  input and output tensores.

  Compared to V1, this gives 3x speed up.
  """

  def __init__(self, num_heads, key_dim, value_dim, dropout=0):
    """Initializer."""
    super().__init__()
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._dropout = dropout

  def build(self, input_shape):
    """Create layer state."""
    x_shape, m_shape = input_shape
    self._channel_dim = x_shape[-1]
    assert self._channel_dim == m_shape[-1], f'x={x_shape}, m={m_shape}'
    self._query_proj = self.add_weight(
        'query', [self._num_heads, self._key_dim, self._channel_dim]
    )
    self._key_proj = self.add_weight('key', [self._channel_dim, self._key_dim])
    self._value_proj = self.add_weight(
        'value', [self._channel_dim, self._value_dim]
    )
    self._output_proj = self.add_weight(
        'output', [self._channel_dim, self._num_heads, self._value_dim]
    )
    self._dropout_layer = tf_keras.layers.Dropout(rate=self._dropout)

  def _reshape_input(self, t):
    """Reshapes a tensor to three dimensions, keeping the first and last."""
    s = tf.shape(t)
    # Propagate the shape statically where possible.
    static_num = t.shape[1:-1].num_elements()
    num = static_num or tf.math.reduce_prod(s[1:-1])
    return tf.ensure_shape(
        tf.reshape(t, [s[0], num, s[-1]]), [t.shape[0], static_num, t.shape[-1]]
    )

  def call(self, inputs):
    """Run layer computation."""
    x, m = inputs

    reshaped_x = self._reshape_input(x)
    reshaped_m = self._reshape_input(m)

    q = tf.einsum('bnd,hkd->bnhk', reshaped_x, self._query_proj)
    k = tf.einsum('bmd,dk->bmk', reshaped_m, self._key_proj)
    logits = tf.einsum('bnhk,bmk->bnhm', q, k)

    logits = logits / tf.math.sqrt(tf.cast(self._key_dim, x.dtype))
    attention_scores = self._dropout_layer(tf.nn.softmax(logits))

    v = tf.einsum('bmd,dv->bmv', reshaped_m, self._value_proj)
    o = tf.einsum('bnhm,bmv->bnhv', attention_scores, v)
    result = tf.einsum('bnhv,dhv->bnd', o, self._output_proj)

    return tf.ensure_shape(tf.reshape(result, tf.shape(x)), x.shape)


class OptimizedMultiQueryAttentionLayerWithDownSampling(tf_keras.layers.Layer):
  """Multi Query Attention with spatial downsampling.

   3 parameters are introduced for the spatial downsampling:
   1. kv_strides: downsampling factor on Key and Values only.
   2. query_h_strides: vertical strides on Query only.
   3. query_w_strides: horizontal strides on Query only.

  This is an optimized version.
  1. Projections in Attention is explict written out as 1x1 Conv2D.
  2. Additional reshapes are introduced to bring a up to 3x speed up.
  """

  def __init__(
      self,
      num_heads: int,
      key_dim: int,
      value_dim: int,
      query_h_strides: int = 1,
      query_w_strides: int = 1,
      kv_strides: int = 1,
      dropout: float = 0,
      dw_kernel_size: int = 3,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
  ):
    """Initializer.

    Args:
      num_heads: Number of attention heads.
      key_dim: Size of the attention key dimension.
      value_dim: Size of the attention value dimension.
      query_h_strides: Vertical stride size for query only.
      query_w_strides: Horizontal stride size for query only.
      kv_strides: Key and value stride size.
      dropout: Dropout probability (between 0 and 1).
      dw_kernel_size: Spatial dimension of the depthwise kernel.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: Momentum value for use with normalization moving average.
      norm_epsilon: Small float added to norm variance to avoid dividing by
        zero.
    """
    super().__init__()
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._query_h_strides = query_h_strides
    self._query_w_strides = query_w_strides
    self._kv_strides = kv_strides
    self._dw_kernel_size = dw_kernel_size
    self._dropout = dropout
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

  def build(self, input_shape):
    """Create layer state."""
    self._channel_dim = input_shape[-1]

    if self._query_h_strides > 1 or self._query_w_strides > 1:
      self._query_downsampling = tf_keras.layers.AvgPool2D(
          pool_size=(self._query_h_strides, self._query_w_strides),
          padding='same',
      )
      self._query_downsampling_norm = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )

    self._query_proj = tf_keras.layers.Conv2D(
        filters=self._num_heads * self._key_dim,
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=False,
    )

    if self._kv_strides > 1:
      self._key_dw_conv = tf_keras.layers.DepthwiseConv2D(
          kernel_size=self._dw_kernel_size,
          strides=self._kv_strides,
          padding='same',
          depth_multiplier=1,
          use_bias=False,
      )
      self._key_dw_norm = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )
    self._key_proj = tf_keras.layers.Conv2D(
        filters=self._key_dim,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
    )

    if self._kv_strides > 1:
      self._value_dw_conv = tf_keras.layers.DepthwiseConv2D(
          kernel_size=self._dw_kernel_size,
          strides=self._kv_strides,
          padding='same',
          depth_multiplier=1,
          use_bias=False,
      )
      self._value_dw_norm = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
      )
    self._value_proj = tf_keras.layers.Conv2D(
        filters=self._value_dim,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
    )

    self._output_proj = tf_keras.layers.Conv2D(
        filters=self._channel_dim,
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=False,
    )
    if self._query_h_strides > 1 or self._query_w_strides > 1:
      self._upsampling = tf_keras.layers.UpSampling2D(
          size=(self._query_h_strides, self._query_w_strides),
          interpolation='bilinear',
      )
    self._dropout_layer = tf_keras.layers.Dropout(rate=self._dropout)

  def _reshape_input(self, t):
    """Reshapes a tensor to three dimensions, keeping the first and last."""
    s = tf.shape(t)
    # Propagate the shape statically where possible.
    static_num = t.shape[1:-1].num_elements()
    num = static_num or tf.math.reduce_prod(s[1:-1])
    return tf.ensure_shape(
        tf.reshape(t, [s[0], num, s[-1]]), [t.shape[0], static_num, t.shape[-1]]
    )

  def _reshape_projected_query(self, t, num_heads, h_px, w_px, key_dim):
    """Reshapes projected query: [b, n, n, h x k] -> [b, n x n, h, k]."""
    s = tf.shape(t)
    return tf.reshape(t, [s[0], h_px * w_px, num_heads, key_dim])

  def _get_pixels(self, t):
    s = tf.shape(t)
    static_num = t.shape[1]
    px = static_num or s[1]
    return px

  def _reshape_output(self, t, num_heads, h_px, w_px):
    """Reshape output:[b, n x n x h, k] -> [b, n, n, hk]."""
    s = tf.shape(t)
    # Propagate the shape statically where possible.
    static_last_dim = t.shape[-1]
    last_dim = (static_last_dim or s[-1]) * num_heads
    return tf.reshape(t, [t.shape[0] or s[0], h_px, w_px, last_dim])

  def call(self, inputs):
    """Run layer computation."""
    x = inputs
    px = self._get_pixels(x)

    if self._query_h_strides > 1 or self._query_w_strides > 1:
      q = self._query_downsampling(x)
      q = self._query_downsampling_norm(q)
      q = self._query_proj(q)
    else:
      q = self._query_proj(x)

    # desired q shape: [b, n x n, h, k] - [b, l, h, k]
    q = self._reshape_projected_query(
        q,
        self._num_heads,
        px // self._query_h_strides,
        px // self._query_w_strides,
        self._key_dim,
    )

    if self._kv_strides > 1:
      k = self._key_dw_conv(x)
      k = self._key_dw_norm(k)
      k = self._key_proj(k)
    else:
      k = self._key_proj(x)
    # output shape of k: [b, k, p], p = m x m
    k = self._reshape_input(k)

    # desired q shape: [b, n x n, h, k]
    # desired k shape: [b, m x m, k]
    # desired logits shape: [b, n x n, h, m x m]
    logits = tf.einsum('blhk,bpk->blhp', q, k)

    logits = logits / tf.math.sqrt(tf.cast(self._key_dim, x.dtype))

    attention_scores = self._dropout_layer(tf.nn.softmax(logits))

    if self._kv_strides > 1:
      v = self._value_dw_conv(x)
      v = self._value_dw_norm(v)
      v = self._value_proj(v)
    else:
      v = self._value_proj(x)

    # output shape of v: [ b, p, k], p = m x m
    v = self._reshape_input(v)
    o = tf.einsum('blhp,bpk->blhk', attention_scores, v)
    # reshape o into [b, n, n, hk]
    o = self._reshape_output(
        o,
        self._num_heads,
        px // self._query_h_strides,
        px // self._query_w_strides,
    )
    if self._query_h_strides > 1 or self._query_w_strides > 1:
      o = self._upsampling(o)

    result = self._output_proj(o)

    return tf.ensure_shape(tf.reshape(result, tf.shape(x)), x.shape)


@tf_keras.utils.register_keras_serializable(package='Vision')
class MultiHeadSelfAttentionBlock(tf_keras.layers.Layer):
  """A Multi Head Self Attention block."""

  def __init__(
      self,
      input_dim,
      num_heads=8,
      key_dim=64,
      value_dim=64,
      use_multi_query=False,
      query_h_strides=1,
      query_w_strides=1,
      kv_strides=1,
      downsampling_dw_kernel_size=3,
      dropout=0.0,
      use_bias=False,
      use_cpe=False,
      cpe_dw_kernel_size=7,
      stochastic_depth_drop_rate=None,
      use_residual=True,
      use_sync_bn=False,
      use_layer_scale=True,
      layer_scale_init_value=1e-5,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      output_intermediate_endpoints=False,
      **kwargs,
  ):
    """Initializes a MultiHeadSelfAttentionBlock.

    A Self-Attention block mixing tokens spatially and globally.

    Args:
      input_dim: dimension of the channels of the input feature.
      num_heads: number of heads. Default is 8. If None, num_heads are computed
        automatically as input_dim // key_dim.
      key_dim: Number of projected key and query dimension per head. Default is
        64.
      value_dim: Number of projected value dimension per head. Default is 64.
      use_multi_query: If true, use MultiQueryAttention.
      query_h_strides: Spatial downsampling strides on vertical axis on query.
      query_w_strides: Spatial downsampling strides on horizontal axis on query.
      kv_strides: Spatial downsampling strides on key and values.
      downsampling_dw_kernel_size: The sise of DW kernel in the downsampling
        layer.
      dropout: Dropout rate for the attention score layer and projection layer.
      use_bias: whether to use bias.
      use_cpe: A 'bool'. If True, add Conditional Position Encoding.
      cpe_dw_kernel_size: An `int` kernel size of the CPE depthwise.
      stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
        the stochastic depth layer.
      use_residual: A `bool` of whether to include residual connection between
        input and output.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      use_layer_scale: A 'bool'. If True, scale the output of MHSA.
      layer_scale_init_value: A 'float' of initial value of layer scale.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      output_intermediate_endpoints: A `bool` of whether or not output the
        intermediate endpoints. For the moment, this block does not output any
        intermediate endpoint.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._input_dim = input_dim
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._use_multi_query = use_multi_query
    self._query_h_strides = query_h_strides
    self._query_w_strides = query_w_strides
    self._kv_strides = kv_strides
    self._downsampling_dw_kernel_size = downsampling_dw_kernel_size
    self._dropout = dropout
    self._use_bias = use_bias
    self._use_cpe = use_cpe
    self._cpe_dw_kernel_size = cpe_dw_kernel_size
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._use_residual = use_residual
    self._use_sync_bn = use_sync_bn
    self._use_layer_scale = use_layer_scale
    self._layer_scale_init_value = layer_scale_init_value
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._output_intermediate_endpoints = output_intermediate_endpoints

    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

  def build(self, input_shape):
    """Create layer state."""
    self._input_norm = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
    )

    # This CPE is different than the one suggested in the original paper.
    # https://arxiv.org/abs/2102.10882
    # 1. Rather than adding one CPE before the attention blocks, we add a CPE
    #    into every attention block.
    # 2. We replace the expensive Conv2D by a Seperable DW Conv.
    if self._use_cpe:
      self._cpe_dw_conv = tf_keras.layers.DepthwiseConv2D(
          kernel_size=self._cpe_dw_kernel_size,
          strides=1,
          padding='same',
          depth_multiplier=1,
          use_bias=True,
      )

    # TODO(qind): assert feature dim dividable by 32
    if self._num_heads is None:
      num_heads = self._input_dim // self._key_dim
    else:
      num_heads = self._num_heads
    if self._use_multi_query:
      if (
          self._query_h_strides > 1
          or self._query_w_strides > 1
          or self._kv_strides > 1
      ):
        self._multi_query_attention = (
            OptimizedMultiQueryAttentionLayerWithDownSampling(
                num_heads=num_heads,
                key_dim=self._key_dim,
                value_dim=self._value_dim,
                query_h_strides=self._query_h_strides,
                query_w_strides=self._query_w_strides,
                kv_strides=self._kv_strides,
                dw_kernel_size=self._downsampling_dw_kernel_size,
                dropout=self._dropout,
            )
        )
      else:
        self._multi_query_attention = MultiQueryAttentionLayerV2(
            num_heads=num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
            dropout=self._dropout,
        )
    else:
      self._multi_head_attention = tf_keras.layers.MultiHeadAttention(
          num_heads=num_heads,
          key_dim=self._key_dim,
          dropout=self._dropout,
          use_bias=self._use_bias,
      )

    if self._use_layer_scale:
      self._layer_scale = MNV4LayerScale(self._layer_scale_init_value)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate
      )
    else:
      self._stochastic_depth = None

    super().build(input_shape)

  def get_config(self) -> dict[str, Any]:
    """Return a Python dict containing this layer's configuration data."""
    config = {
        'input_dim': self._input_dim,
        'num_heads': self._num_heads,
        'key_dim': self._key_dim,
        'value_dim': self._value_dim,
        'use_multi_query': self._use_multi_query,
        'kv_strides': self._kv_strides,
        'query_h_strides': self._query_h_strides,
        'query_w_strides': self._query_w_strides,
        'downsampling_dw_kernel_size': self._downsampling_dw_kernel_size,
        'dropout': self._dropout,
        'use_bias': self._use_bias,
        'cpe_dw_kernel_size': self._cpe_dw_kernel_size,
        'use_cpe': self._use_cpe,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'use_sync_bn': self._use_sync_bn,
        'use_residual': self._use_residual,
        'use_layer_scale': self._use_layer_scale,
        'layer_scale_init_value': self._layer_scale_init_value,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'output_intermediate_endpoints': self._output_intermediate_endpoints,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Run layer computation."""
    if self._use_cpe:
      x = self._cpe_dw_conv(inputs)
      x = x + inputs
      cpe_outputs = x
    else:
      cpe_outputs = inputs

    shortcut = cpe_outputs
    x = self._input_norm(cpe_outputs)

    if self._use_multi_query:
      if (
          self._query_h_strides > 1
          or self._query_w_strides > 1
          or self._kv_strides > 1
      ):
        x = self._multi_query_attention(x)
      else:
        x = self._multi_query_attention((x, x))
    else:
      x = self._multi_head_attention(x, x)

    if self._use_layer_scale:
      x = self._layer_scale(x)

    if self._use_residual:
      if self._stochastic_depth:
        x = self._stochastic_depth(x)
      x = x + shortcut

    # Return empty intermediate endpoints to be compatible with other blocks.
    if self._output_intermediate_endpoints:
      return x, {}
    return x


@tf_keras.utils.register_keras_serializable(package='Vision')
class ResidualInner(tf_keras.layers.Layer):
  """Creates a single inner block of a residual.

  This corresponds to `F`/`G` functions in the RevNet paper:
  Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse.
  The Reversible Residual Network: Backpropagation Without Storing Activations.
  (https://arxiv.org/pdf/1707.04585.pdf)
  """

  def __init__(
      self,
      filters: int,
      strides: int,
      kernel_initializer: Union[str, Callable[
          ..., tf_keras.initializers.Initializer]] = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      activation: Union[str, Callable[..., tf.Tensor]] = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      batch_norm_first: bool = True,
      **kwargs):
    """Initializes a ResidualInner.

    Args:
      filters: An `int` of output filter size.
      strides: An `int` of stride size for convolution for the residual block.
      kernel_initializer: A `str` or `tf_keras.initializers.Initializer`
        instance for convolutional layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` for Conv2D.
      activation: A `str` or `callable` instance of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      batch_norm_first: A `bool` of whether to apply activation and batch norm
        before conv.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ResidualInner, self).__init__(**kwargs)

    self.strides = strides
    self.filters = filters
    self._kernel_initializer = tf_keras.initializers.get(kernel_initializer)
    self._kernel_regularizer = kernel_regularizer
    self._activation = tf_keras.activations.get(activation)
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._batch_norm_first = batch_norm_first
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape: tf.TensorShape):
    if self._batch_norm_first:
      self._batch_norm_0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          synchronized=self._use_sync_bn,
      )

    self._conv2d_1 = tf_keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=3,
        strides=self.strides,
        use_bias=False,
        padding='same',
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer)

    self._batch_norm_1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )

    self._conv2d_2 = tf_keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
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

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    x = inputs
    if self._batch_norm_first:
      x = self._batch_norm_0(x, training=training)
      x = self._activation_fn(x)
    x = self._conv2d_1(x)

    x = self._batch_norm_1(x, training=training)
    x = self._activation_fn(x)
    x = self._conv2d_2(x)
    return x


@tf_keras.utils.register_keras_serializable(package='Vision')
class BottleneckResidualInner(tf_keras.layers.Layer):
  """Creates a single inner block of a bottleneck.

  This corresponds to `F`/`G` functions in the RevNet paper:
  Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse.
  The Reversible Residual Network: Backpropagation Without Storing Activations.
  (https://arxiv.org/pdf/1707.04585.pdf)
  """

  def __init__(
      self,
      filters: int,
      strides: int,
      kernel_initializer: Union[str, Callable[
          ..., tf_keras.initializers.Initializer]] = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      activation: Union[str, Callable[..., tf.Tensor]] = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      batch_norm_first: bool = True,
      **kwargs):
    """Initializes a BottleneckResidualInner.

    Args:
      filters: An `int` number of filters for first 2 convolutions. Last Last,
        and thus the number of output channels from the bottlneck block is
        `4*filters`
      strides: An `int` of stride size for convolution for the residual block.
      kernel_initializer: A `str` or `tf_keras.initializers.Initializer`
        instance for convolutional layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` for Conv2D.
      activation: A `str` or `callable` instance of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      batch_norm_first: A `bool` of whether to apply activation and batch norm
        before conv.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(BottleneckResidualInner, self).__init__(**kwargs)

    self.strides = strides
    self.filters = filters
    self._kernel_initializer = tf_keras.initializers.get(kernel_initializer)
    self._kernel_regularizer = kernel_regularizer
    self._activation = tf_keras.activations.get(activation)
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._batch_norm_first = batch_norm_first
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape: tf.TensorShape):
    if self._batch_norm_first:
      self._batch_norm_0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          synchronized=self._use_sync_bn,
      )
    self._conv2d_1 = tf_keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=self.strides,
        use_bias=False,
        padding='same',
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer)
    self._batch_norm_1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )
    self._conv2d_2 = tf_keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=3,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer)
    self._batch_norm_2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )
    self._conv2d_3 = tf_keras.layers.Conv2D(
        filters=self.filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
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

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
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


@tf_keras.utils.register_keras_serializable(package='Vision')
class ReversibleLayer(tf_keras.layers.Layer):
  """Creates a reversible layer.

  Computes y1 = x1 + f(x2), y2 = x2 + g(y1), where f and g can be arbitrary
  layers that are stateless, which in this case are `ResidualInner` layers.
  """

  def __init__(self,
               f: tf_keras.layers.Layer,
               g: tf_keras.layers.Layer,
               manual_grads: bool = True,
               **kwargs):
    """Initializes a ReversibleLayer.

    Args:
      f: A `tf_keras.layers.Layer` instance of `f` inner block referred to in
        paper. Each reversible layer consists of two inner functions. For
        example, in RevNet the reversible residual consists of two f/g inner
        (bottleneck) residual functions. Where the input to the reversible layer
        is x, the input gets partitioned in the channel dimension and the
        forward pass follows (eq8): x = [x1; x2], z1 = x1 + f(x2), y2 = x2 +
        g(z1), y1 = stop_gradient(z1).
      g: A `tf_keras.layers.Layer` instance of `g` inner block referred to in
        paper. Detailed explanation same as above as `f` arg.
      manual_grads: A `bool` [Testing Only] of whether to manually take
        gradients as in Algorithm 1 or defer to autograd.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ReversibleLayer, self).__init__(**kwargs)

    self._f = f
    self._g = g
    self._manual_grads = manual_grads

    if tf_keras.backend.image_data_format() == 'channels_last':
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
        v.read_value() for v in self._f.non_trainable_variables
    ]
    self._g_non_trainable_vars = [
        v.read_value() for v in self._g.non_trainable_variables
    ]

  def _load_ckpt_non_trainable_vars(self):
    for v, v_chkpt in zip(self._f.non_trainable_variables,
                          self._f_non_trainable_vars):
      v.assign(v_chkpt)
    for v, v_chkpt in zip(self._g.non_trainable_variables,
                          self._g_non_trainable_vars):
      v.assign(v_chkpt)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:

    @tf.custom_gradient
    def reversible(
        x: tf.Tensor
    ) -> Tuple[tf.Tensor, Callable[[Any], Tuple[List[tf.Tensor],
                                                List[tf.Tensor]]]]:
      """Implements Algorithm 1 in the RevNet paper.

         Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse.
         The Reversible Residual Network: Backpropagation Without Storing
         Activations.
         (https://arxiv.org/pdf/1707.04585.pdf)

      Args:
        x: An input `tf.Tensor.

      Returns:
        y: The output [y1; y2] in Algorithm 1.
        grad_fn: A callable function that computes the gradients.
      """
      with tf.GradientTape() as fwdtape:
        fwdtape.watch(x)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self._axis)
        f_x2 = self._f(x2, training=training)
        x1_down = _maybe_downsample(x1, f_x2.shape[self._axis], self._f.strides,
                                    self._axis)
        z1 = f_x2 + x1_down
        g_z1 = self._g(z1, training=training)
        x2_down = _maybe_downsample(x2, g_z1.shape[self._axis], self._f.strides,
                                    self._axis)
        y2 = x2_down + g_z1

        # Equation 8: https://arxiv.org/pdf/1707.04585.pdf
        # Decouple y1 and z1 so that their derivatives are different.
        y1 = tf.identity(z1)
        y = tf.concat([y1, y2], axis=self._axis)

        irreversible = ((self._f.strides != 1 or self._g.strides != 1) or
                        (y.shape[self._axis] != inputs.shape[self._axis]))

        # Checkpointing moving mean/variance for batch normalization layers
        # as they shouldn't be updated during the custom gradient pass of f/g.
        self._ckpt_non_trainable_vars()

      def grad_fn(
          dy: tf.Tensor,
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
          z1 = y1_nograd  # line 2
          with tf.GradientTape() as gtape:
            gtape.watch(z1)
            g_z1 = self._g(z1, training=training)
          x2 = y2_nograd - g_z1  # line 3

          with tf.GradientTape() as ftape:
            ftape.watch(x2)
            f_x2 = self._f(x2, training=training)
          x1 = z1 - f_x2  # pylint: disable=unused-variable      # line 4

          # Compute gradients
          g_grads_combined = gtape.gradient(
              g_z1, [z1] + self._g.trainable_variables, output_gradients=dy2)
          dz1 = dy1 + g_grads_combined[0]  # line 5
          dwg = g_grads_combined[1:]  # line 9

          f_grads_combined = ftape.gradient(
              f_x2, [x2] + self._f.trainable_variables, output_gradients=dz1)
          dx2 = dy2 + f_grads_combined[0]  # line 6
          dwf = f_grads_combined[1:]  # line 8
          dx1 = dz1  # line 7

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


@tf_keras.utils.register_keras_serializable(package='Vision')
class DepthwiseSeparableConvBlock(tf_keras.layers.Layer):
  """Creates a depthwise separable convolution block with batch normalization.
  """

  def __init__(
      self,
      filters: int,
      kernel_size: int = 3,
      strides: int = 1,
      regularize_depthwise=False,
      activation: Text = 'relu6',
      kernel_initializer: Text = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      dilation_rate: int = 1,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs):
    """Initializes a convolution block with batch normalization.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      kernel_size: An `int` that specifies the height and width of the 2D
        convolution window.
      strides: An `int` of block stride. If greater than 1, this block will
        ultimately downsample the input.
      regularize_depthwise: A `bool`. If Ture, apply regularization on
        depthwise.
      activation: A `str` name of the activation function.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      dilation_rate: An `int` or tuple/list of 2 `int`, specifying the dilation
        rate to use for dilated convolution. Can be a single integer to specify
        the same value for all spatial dimensions.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(DepthwiseSeparableConvBlock, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._regularize_depthwise = regularize_depthwise
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._dilation_rate = dilation_rate
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)
    if regularize_depthwise:
      self._depthsize_regularizer = kernel_regularizer
    else:
      self._depthsize_regularizer = None

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'regularize_depthwise': self._regularize_depthwise,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(DepthwiseSeparableConvBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):

    self._dwconv0 = tf_keras.layers.DepthwiseConv2D(
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding='same',
        depth_multiplier=1,
        dilation_rate=self._dilation_rate,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._depthsize_regularizer,
        use_bias=False)
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )

    self._conv1 = tf_keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )

    super(DepthwiseSeparableConvBlock, self).build(input_shape)

  def call(self, inputs, training=None):
    x = self._dwconv0(inputs)
    x = self._norm0(x)
    x = self._activation_fn(x)

    x = self._conv1(x)
    x = self._norm1(x)
    return self._activation_fn(x)


@tf_keras.utils.register_keras_serializable(package='Vision')
class TuckerConvBlock(tf_keras.layers.Layer):
  """An Tucker block (generalized bottleneck)."""

  def __init__(self,
               in_filters,
               out_filters,
               input_compression_ratio,
               output_compression_ratio,
               strides,
               kernel_size=3,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               divisible_by=1,
               use_residual=True,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Initializes an inverted bottleneck block with BN after convolutions.

    Args:
      in_filters: An `int` number of filters of the input tensor.
      out_filters: An `int` number of filters of the output tensor.
      input_compression_ratio: An `float` of compression ratio for input
        filters.
      output_compression_ratio: An `float` of compression ratio for output
        filters.
      strides: An `int` block stride. If greater than 1, this block will
        ultimately downsample the input.
      kernel_size: An `int` kernel_size of the depthwise conv layer.
      stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      use_residual: A `bool` of whether to include residual connection between
        input and output.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(TuckerConvBlock, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._input_compression_ratio = input_compression_ratio
    self._output_compression_ratio = output_compression_ratio
    self._strides = strides
    self._kernel_size = kernel_size
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._use_sync_bn = use_sync_bn
    self._use_residual = use_residual
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

  def build(self, input_shape):
    input_compressed_filters = nn_layers.make_divisible(
        value=self._in_filters * self._input_compression_ratio,
        divisor=self._divisible_by,
        round_down_protect=False)

    self._conv0 = tf_keras.layers.Conv2D(
        filters=input_compressed_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )
    self._activation_layer0 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    output_compressed_filters = nn_layers.make_divisible(
        value=self._out_filters * self._output_compression_ratio,
        divisor=self._divisible_by,
        round_down_protect=False)

    self._conv1 = tf_keras.layers.Conv2D(
        filters=output_compressed_filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )
    self._activation_layer1 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    # Last 1x1 conv.
    self._conv2 = tf_keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn,
    )

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tf_keras.layers.Add()

    super(TuckerConvBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'input_compression_ratio': self._input_compression_ratio,
        'output_compression_ratio': self._output_compression_ratio,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'use_residual': self._use_residual,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(TuckerConvBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs

    x = self._conv0(inputs)
    x = self._norm0(x)
    x = self._activation_layer0(x)

    x = self._conv1(x)
    x = self._norm1(x)
    x = self._activation_layer1(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if (self._use_residual and self._in_filters == self._out_filters and
        self._strides == 1):
      if self._stochastic_depth:
        x = self._stochastic_depth(x, training=training)
      x = self._add([x, shortcut])

    return x


@tf_keras.utils.register_keras_serializable(package='Vision')
class LayerScale(tf_keras.layers.Layer):
  """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239.

  Attributes:
      init_values (float): value to initialize the diagonal matrix of
        LayerScale.
  """

  def __init__(self, init_values: float, **kwargs):
    """Initializes LayerScale."""
    super().__init__(**kwargs)
    self.gamma_init_value = init_values

  def build(self, inputs_shape):
    gamma_shape = (1, 1, inputs_shape[2])
    self.gamma = self.add_weight(
        name='layerscale_gamma',
        shape=gamma_shape,
        initializer=tf_keras.initializers.Constant(self.gamma_init_value),
        trainable=True,
        dtype=tf.float32,
    )

  def call(self, inputs, inputs_positions=None):
    del inputs_positions
    return tf.cast(self.gamma, inputs.dtype) * inputs


@tf_keras.utils.register_keras_serializable(package='Vision')
class MNV4LayerScale(tf_keras.layers.Layer):
  """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239.

  As used in MobileNetV4.

  Attributes:
      init_value (float): value to initialize the diagonal matrix of LayerScale.
  """

  def __init__(self, init_value: float, **kwargs):
    super().__init__(**kwargs)
    self._init_value = init_value

  def build(self, inputs_shape):
    embedding_dim = inputs_shape[-1]
    self._gamma = tf.Variable(self._init_value * tf.ones((embedding_dim,)))

  def call(self, x, training=None):
    return x * tf.cast(self._gamma, x.dtype)


@tf_keras.utils.register_keras_serializable(package='Vision')
class TransformerEncoderBlock(nlp_modeling.layers.TransformerEncoderBlock):
  """TransformerEncoderBlock layer with stochastic depth and layerscale."""

  def __init__(
      self,
      *args,
      stochastic_depth_drop_rate=0.0,
      layer_scale_init_value=0.0,
      transformer_partition_dims=None,
      max_attention_inference_parallelism=None,
      **kwargs
  ):
    """Initializes TransformerEncoderBlock.

    Args:
      *args: positional arguments passed to super().__init__.
      stochastic_depth_drop_rate: the drop rate for the stochastic depth layer.
      layer_scale_init_value:
      transformer_partition_dims: transformer spatial partition dimenstions.
      max_attention_inference_parallelism: the number of examples to run in
        parallel in the attention blocks during inference. Set this limit to
        reduce the peak memory usage. If None, use vectorized operations to run
        the whole batch in parallel.
      **kwargs: keyword arguments passed to super().__init__.
    """
    super().__init__(*args, **kwargs)
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._layer_scale_init_value = layer_scale_init_value
    self._transformer_partition_dims = transformer_partition_dims
    self._max_attention_inference_parallelism = (
        max_attention_inference_parallelism
    )

  def build(self, input_shape):
    super().build(input_shape)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = lambda x, *args, **kwargs: tf.identity(x)

    if self._layer_scale_init_value:
      self._layer_scale_attn = LayerScale(
          init_values=self._layer_scale_init_value, name='layer_scale_attn')
      self._layer_scale_mlp = LayerScale(
          init_values=self._layer_scale_init_value, name='layer_scale_mlp')
    else:
      self._layer_scale_attn = lambda x, *args, **kwargs: tf.identity(x)
      self._layer_scale_mlp = lambda x, *args, **kwargs: tf.identity(x)

    self._attention_layer = nn_layers.MultiHeadAttention(
        num_heads=self._num_heads,
        key_dim=self._key_dim,
        value_dim=self._value_dim,
        dropout=self._attention_dropout_rate,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        attention_axes=self._attention_axes,
        output_shape=self._output_last_dim,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        max_inference_parallelism=self._max_attention_inference_parallelism,
        partition_dims=self._transformer_partition_dims,
        name='self_attention',
    )

  def get_config(self):
    config = super().get_config()
    config.update({
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'layer_scale_init_value': self._layer_scale_init_value,
        'transformer_partition_dims': self._transformer_partition_dims,
        'max_attention_inference_parallelism': (
            self._max_attention_inference_parallelism
        ),
    })
    return config

  def call(self, inputs, output_range=None, training=None):
    """Transformer self-attention encoder block call."""
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError('Unexpected inputs to %s with length at %d' %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    if output_range is None:
      output_range = self._output_range
    if output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor[:, 0:output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor

    attention_output, attention_scores = self._attention_layer(
        query=target_tensor,
        value=key_value,
        attention_mask=attention_mask,
        return_attention_scores=True)
    attention_output = self._attention_dropout(attention_output)

    attention_output = self._layer_scale_attn(attention_output)

    if self._norm_first:
      # Important to not combine `self._norm_first` and
      # `self._use_query_residual` into one if clause because else is only for
      # `_norm_first == False`.
      if self._use_query_residual:
        attention_output = source_tensor + self._stochastic_depth(
            attention_output, training=training)
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    else:
      if self._use_query_residual:
        attention_output = target_tensor + self._stochastic_depth(
            attention_output, training=training)
      attention_output = self._attention_layer_norm(attention_output)

    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    # Layerscale after MLP.
    layer_output = self._layer_scale_mlp(layer_output)

    if self._norm_first:
      layer_output = source_attention_output + self._stochastic_depth(
          layer_output, training=training)
    else:
      # During mixed precision training, layer norm output is always fp32 for
      # now. Casts fp32 for the subsequent add.
      layer_output = tf.cast(layer_output, tf.float32)
      layer_output = self._output_layer_norm(
          layer_output
          + self._stochastic_depth(attention_output, training=training))

    if self._return_attention_scores:
      return layer_output, attention_scores
    else:
      return layer_output


@tf_keras.utils.register_keras_serializable(package='Vision')
class TransformerScaffold(nlp_modeling.layers.TransformerScaffold):
  """TransformerScaffold layer for vision applications."""

  def __init__(
      self,
      *args,
      stochastic_depth_drop_rate: float = 0.0,
      return_attention_scores: bool = False,
      ffn_has_residual_connection: bool = False,
      max_attention_inference_parallelism: Optional[int] = None,
      **kwargs
  ):
    """Initializes TransformerEncoderBlock.

    Args:
      *args: positional arguments passed to super().__init__.
      stochastic_depth_drop_rate: the drop rate for the stochastic depth layer.
      return_attention_scores: whether to return the attention output.
      ffn_has_residual_connection: whether the feedforward network has internal
        residual connection and layer norm. If False, the residual connection
        and the layer norm op are called inside TransformerScaffold.
      max_attention_inference_parallelism: the number of examples to run in
        parallel in the attention blocks during inference. Set this limit to
        reduce the peak memory usage. If None, use vectorized operations to run
        the whole batch in parallel.
      **kwargs: keyword arguments passed to super().__init__.
    """
    super().__init__(*args, **kwargs)
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._return_attention_scores = return_attention_scores
    self._ffn_has_residual_connection = ffn_has_residual_connection
    self._max_attention_inference_parallelism = (
        max_attention_inference_parallelism
    )

  def build(self, input_shape: Union[tf.TensorShape, List[int]]):
    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = lambda x, *args, **kwargs: tf.identity(x)

    super().build(input_shape)

    if self._max_attention_inference_parallelism is not None:
      attention_layer_config = self._attention_layer.get_config()
      self._attention_layer = self._attention_cls.from_config({
          **attention_layer_config,
          'max_inference_parallelism': (
              self._max_attention_inference_parallelism
          ),
      })

  def get_config(self):
    config = super().get_config()
    config.update({
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'return_attention_scores': self._return_attention_scores,
        'ffn_has_residual_connection': self._ffn_has_residual_connection,
        'max_attention_inference_parallelism': (
            self._max_attention_inference_parallelism
        ),
    })
    return config

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[bool] = None
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Transformer self-attention encoder block call."""
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError('Unexpected inputs to %s with length at %d' %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    if self._norm_first:
      source_tensor = input_tensor
      input_tensor = self._attention_layer_norm(input_tensor)

    if key_value is None:
      key_value = input_tensor

    attention_output, attention_scores = self._attention_layer(
        query=input_tensor,
        value=key_value,
        attention_mask=attention_mask,
        training=training,
        return_attention_scores=True)
    attention_output = self._attention_dropout(
        attention_output, training=training)

    if self._norm_first:
      source_attention_output = source_tensor + self._stochastic_depth(
          attention_output, training=training)
      attention_output = self._output_layer_norm(
          source_attention_output)
    else:
      attention_output = self._attention_layer_norm(
          input_tensor +
          self._stochastic_depth(attention_output, training=training))

    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      intermediate_output = self._intermediate_activation_layer(
          intermediate_output)
      layer_output = self._output_dense(intermediate_output)
      layer_output = self._output_dropout(layer_output, training=training)
    else:
      layer_output = self._feedforward_block(
          attention_output, training=training)

    if self._norm_first:
      if self._ffn_has_residual_connection:
        raise ValueError(
            'In the case of `norm_first`, the residual connection should be'
            "done in the TransformerScaffold call function, not FFN's"
            'call function.')
      output = source_attention_output + self._stochastic_depth(
          layer_output, training=training)
    else:
      # During mixed precision training, layer norm output is always fp32 for
      # now. Casts fp32 for the subsequent add.
      layer_output = tf.cast(layer_output, tf.float32)
      if self._ffn_has_residual_connection:
        output = self._stochastic_depth(layer_output, training=training)
      else:
        output = self._output_layer_norm(
            attention_output +
            self._stochastic_depth(layer_output, training=training))

    if self._return_attention_scores:
      return output, attention_scores
    else:
      return output
