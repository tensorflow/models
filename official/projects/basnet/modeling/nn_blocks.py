# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Contains common building blocks for BasNet model."""

import tensorflow as tf

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class ConvBlock(tf.keras.layers.Layer):
  """A (Conv+BN+Activation) block."""

  def __init__(self,
               filters,
               strides,
               dilation_rate=1,
               kernel_size=3,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_bias=False,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A vgg block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      dilation_rate: `int`, dilation rate for conv layers.
      kernel_size: `int`, kernel size of conv layers.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      activation: `str` name of the activation function.
      use_bias: `bool`, whether or not use bias in conv layers.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(ConvBlock, self).__init__(**kwargs)
    self._config_dict = {
        'filters': filters,
        'kernel_size': kernel_size,
        'strides': strides,
        'dilation_rate': dilation_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'use_bias': use_bias,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon
    }
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
    conv_kwargs = {
        'padding': 'same',
        'use_bias': self._config_dict['use_bias'],
        'kernel_initializer': self._config_dict['kernel_initializer'],
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }

    self._conv0 = tf.keras.layers.Conv2D(
        filters=self._config_dict['filters'],
        kernel_size=self._config_dict['kernel_size'],
        strides=self._config_dict['strides'],
        dilation_rate=self._config_dict['dilation_rate'],
        **conv_kwargs)
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._config_dict['norm_momentum'],
        epsilon=self._config_dict['norm_epsilon'])

    super(ConvBlock, self).build(input_shape)

  def get_config(self):
    return self._config_dict

  def call(self, inputs, training=None):
    x = self._conv0(inputs)
    x = self._norm0(x)
    x = self._activation_fn(x)

    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResBlock(tf.keras.layers.Layer):
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
               use_bias=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
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
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      use_bias: A `bool`. If True, use bias in conv2d.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ResBlock, self).__init__(**kwargs)
    self._config_dict = {
        'filters': filters,
        'strides': strides,
        'use_projection': use_projection,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'use_bias': use_bias,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon
    }
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
    conv_kwargs = {
        'filters': self._config_dict['filters'],
        'padding': 'same',
        'use_bias': self._config_dict['use_bias'],
        'kernel_initializer': self._config_dict['kernel_initializer'],
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }

    if self._config_dict['use_projection']:
      self._shortcut = tf.keras.layers.Conv2D(
          filters=self._config_dict['filters'],
          kernel_size=1,
          strides=self._config_dict['strides'],
          use_bias=self._config_dict['use_bias'],
          kernel_initializer=self._config_dict['kernel_initializer'],
          kernel_regularizer=self._config_dict['kernel_regularizer'],
          bias_regularizer=self._config_dict['bias_regularizer'])
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._config_dict['norm_momentum'],
          epsilon=self._config_dict['norm_epsilon'])

    self._conv1 = tf.keras.layers.Conv2D(
        kernel_size=3,
        strides=self._config_dict['strides'],
        **conv_kwargs)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._config_dict['norm_momentum'],
        epsilon=self._config_dict['norm_epsilon'])

    self._conv2 = tf.keras.layers.Conv2D(
        kernel_size=3,
        strides=1,
        **conv_kwargs)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._config_dict['norm_momentum'],
        epsilon=self._config_dict['norm_epsilon'])

    super(ResBlock, self).build(input_shape)

  def get_config(self):
    return self._config_dict

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._config_dict['use_projection']:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._conv2(x)
    x = self._norm2(x)

    return self._activation_fn(x + shortcut)
