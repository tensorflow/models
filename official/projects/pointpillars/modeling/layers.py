# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Featurizer layers for Pointpillars."""

from typing import Any, Mapping, Optional

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.pointpillars.utils import utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class ConvBlock(tf.keras.layers.Layer):
  """A conv2d followed by a norm then an activation."""

  def __init__(
      self,
      filters: int,
      kernel_size: int,
      strides: int,
      use_transpose_conv: bool = False,
      kernel_initializer: Optional[tf.keras.initializers.Initializer] = tf.keras
      .initializers.VarianceScaling(),
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_bias: bool = False,
      bias_initializer: Optional[tf.keras.initializers.Initializer] = tf.keras
      .initializers.Zeros(),
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_sync_bn: bool = True,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      bn_trainable: bool = True,
      activation: str = 'relu',
      **kwargs):
    """Initialize a block with conv, bn and activation.

    Args:
      filters: An int number of filters of the conv layer.
      kernel_size: An int number of kernel size of the conv layer.
      strides: An int number of strides of the conv layer.
      use_transpose_conv: A bool for wether to use transpose conv or not.
      kernel_initializer: A tf Initializer object for the conv layer.
      kernel_regularizer: A tf Regularizer object for the conv layer.
      use_bias: A bool for whether to use bias for the conv layer.
      bias_initializer: A tf Initializer object for the conv layer bias.
      bias_regularizer: A tf Regularizer object for the conv layer bias.
      use_sync_bn: A bool for wether to use synchronized batch normalization.
      norm_momentum: A float of normalization momentum for the moving average.
      norm_epsilon: A float added to variance to avoid dividing by zero.
      bn_trainable:  A bool that indicates whether batch norm layers should be
        trainable. Default to True.
      activation: A str name of the activation function.
      **kwargs: Additional keyword arguments to be passed.
    """

    super(ConvBlock, self).__init__(**kwargs)

    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._use_transpose_conv = use_transpose_conv
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._use_bias = use_bias
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._bn_trainable = bn_trainable
    self._activation = activation
    self._activation_fn = tf_utils.get_activation(activation)

    utils.assert_channels_last()

  def build(self, input_shape: tf.TensorShape):
    """Creates variables for the block."""
    # Config conv
    if self._use_transpose_conv:
      conv_op = tf.keras.layers.Conv2DTranspose
    else:
      conv_op = tf.keras.layers.Conv2D
    conv_kwargs = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'padding': 'same',
        'use_bias': self._use_bias,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    self._conv = conv_op(**conv_kwargs)

    # Config norm
    if self._use_sync_bn:
      bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_op = tf.keras.layers.BatchNormalization
    bn_kwargs = {
        'axis': -1,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon,
        'trainable': self._bn_trainable,
    }
    self._norm = bn_op(**bn_kwargs)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of the block."""
    x = inputs
    x = self._conv(x)
    x = self._norm(x)
    outputs = self._activation_fn(x)
    return outputs

  def get_config(self) -> Mapping[str, Any]:
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'use_transpose_conv': self._use_transpose_conv,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_bias': self._use_bias,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'bn_trainable': self._bn_trainable,
        'activation': self._activation,
    }
    return config

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> tf.keras.Model:
    return cls(**config)
