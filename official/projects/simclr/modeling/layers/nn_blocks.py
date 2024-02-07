# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Contains common building blocks for simclr neural networks."""
from typing import Text, Optional

import tensorflow as tf, tf_keras

from official.modeling import tf_utils

regularizers = tf_keras.regularizers


class DenseBN(tf_keras.layers.Layer):
  """Modified Dense layer to help build simclr system.

  The layer is a standards combination of Dense, BatchNorm and Activation.
  """

  def __init__(
      self,
      output_dim: int,
      use_bias: bool = True,
      use_normalization: bool = False,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      activation: Optional[Text] = 'relu',
      kernel_initializer: Text = 'VarianceScaling',
      kernel_regularizer: Optional[regularizers.Regularizer] = None,
      bias_regularizer: Optional[regularizers.Regularizer] = None,
      name='linear_layer',
      **kwargs):
    """Customized Dense layer.

    Args:
      output_dim: `int` size of output dimension.
      use_bias: if True, use biase in the dense layer.
      use_normalization: if True, use batch normalization.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization momentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      activation: `str` name of the activation function.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      name: `str`, name of the layer.
      **kwargs: keyword arguments to be passed.
    """
    # Note: use_bias is ignored for the dense layer when use_bn=True.
    # However, it is still used for batch norm.
    super(DenseBN, self).__init__(**kwargs)
    self._output_dim = output_dim
    self._use_bias = use_bias
    self._use_normalization = use_normalization
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._name = name

    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if activation:
      self._activation_fn = tf_utils.get_activation(activation)
    else:
      self._activation_fn = None

  def get_config(self):
    config = {
        'output_dim': self._output_dim,
        'use_bias': self._use_bias,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'use_normalization': self._use_normalization,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    base_config = super(DenseBN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._dense0 = tf_keras.layers.Dense(
        self._output_dim,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_bias=self._use_bias and not self._use_normalization)

    if self._use_normalization:
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          center=self._use_bias,
          scale=True)

    super(DenseBN, self).build(input_shape)

  def call(self, inputs, training=None):
    assert inputs.shape.ndims == 2, inputs.shape
    x = self._dense0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    if self._activation:
      x = self._activation_fn(x)
    return x
