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
from typing import Text, Optional

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
