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

"""Contains common building blocks for neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, tf_keras

from official.modeling import tf_utils


class ResidualBlock(tf_keras.layers.Layer):
  """A residual block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
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
    super(ResidualBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._use_sync_bn = use_sync_bn
    self._activation = activation
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
      self._shortcut = tf_keras.layers.Conv2D(
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

    self._conv1 = tf_keras.layers.Conv2D(
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

    self._conv2 = tf_keras.layers.Conv2D(
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

    super(ResidualBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
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

  def call(self, inputs):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._conv2(x)
    x = self._norm2(x)

    return self._activation_fn(x + shortcut)


class BottleneckBlock(tf_keras.layers.Layer):
  """A standard bottleneck block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
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
    super(BottleneckBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._use_sync_bn = use_sync_bn
    self._activation = activation
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
      self._shortcut = tf_keras.layers.Conv2D(
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

    self._conv1 = tf_keras.layers.Conv2D(
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

    self._conv2 = tf_keras.layers.Conv2D(
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

    self._conv3 = tf_keras.layers.Conv2D(
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

    super(BottleneckBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
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

  def call(self, inputs):
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

    return self._activation_fn(x + shortcut)
