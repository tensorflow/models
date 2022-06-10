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

"""Commonly used TensorFlow 2 network blocks."""
from typing import Any, Text, Sequence, Union

import tensorflow as tf
from official.modeling import tf_utils

WEIGHT_INITIALIZER = {
    'Xavier': tf.keras.initializers.GlorotUniform,
    'Gaussian': lambda: tf.keras.initializers.RandomNormal(stddev=0.01),
}

initializers = tf.keras.initializers
regularizers = tf.keras.regularizers


def make_set_from_start_endpoint(start_endpoint: Text,
                                 endpoints: Sequence[Text]):
  """Makes a subset of endpoints from the given starting position."""
  if start_endpoint not in endpoints:
    return set()
  start_index = endpoints.index(start_endpoint)
  return set(endpoints[start_index:])


def apply_depth_multiplier(d: Union[int, Sequence[Any]],
                           depth_multiplier: float):
  """Applies depth_multiplier recursively to ints."""
  if isinstance(d, int):
    return int(d * depth_multiplier)
  else:
    return [apply_depth_multiplier(x, depth_multiplier) for x in d]


class ParameterizedConvLayer(tf.keras.layers.Layer):
  """Convolution layer based on the input conv_type."""

  def __init__(
      self,
      conv_type: Text,
      kernel_size: int,
      filters: int,
      strides: Sequence[int],
      rates: Sequence[int],
      use_sync_bn: bool = False,
      norm_momentum: float = 0.999,
      norm_epsilon: float = 0.001,
      temporal_conv_initializer: Union[
          Text, initializers.Initializer] = 'glorot_uniform',
      kernel_initializer: Union[Text,
                                initializers.Initializer] = 'truncated_normal',
      kernel_regularizer: Union[Text, regularizers.Regularizer] = 'l2',
      **kwargs):
    super(ParameterizedConvLayer, self).__init__(**kwargs)
    self._conv_type = conv_type
    self._kernel_size = kernel_size
    self._filters = filters
    self._strides = strides
    self._rates = rates
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1
    self._temporal_conv_initializer = temporal_conv_initializer
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer

  def _build_conv_layer_params(self, input_shape):
    """Builds params for conv layers."""
    conv_layer_params = []
    if self._conv_type == '3d':
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[self._kernel_size] * 3,
              strides=self._strides,
              dilation_rate=self._rates,
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
    elif self._conv_type == '2d':
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[1, self._kernel_size, self._kernel_size],
              strides=[1, self._strides[1], self._strides[2]],
              dilation_rate=[1, self._rates[1], self._rates[2]],
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
    elif self._conv_type == '1+2d':
      channels_in = input_shape[self._channel_axis]
      conv_layer_params.append(
          dict(
              filters=channels_in,
              kernel_size=[self._kernel_size, 1, 1],
              strides=[self._strides[0], 1, 1],
              dilation_rate=[self._rates[0], 1, 1],
              kernel_initializer=tf_utils.clone_initializer(
                  self._temporal_conv_initializer),
          ))
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[1, self._kernel_size, self._kernel_size],
              strides=[1, self._strides[1], self._strides[2]],
              dilation_rate=[1, self._rates[1], self._rates[2]],
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
    elif self._conv_type == '2+1d':
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[1, self._kernel_size, self._kernel_size],
              strides=[1, self._strides[1], self._strides[2]],
              dilation_rate=[1, self._rates[1], self._rates[2]],
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[self._kernel_size, 1, 1],
              strides=[self._strides[0], 1, 1],
              dilation_rate=[self._rates[0], 1, 1],
              kernel_initializer=tf_utils.clone_initializer(
                  self._temporal_conv_initializer),
          ))
    elif self._conv_type == '1+1+1d':
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[1, 1, self._kernel_size],
              strides=[1, 1, self._strides[2]],
              dilation_rate=[1, 1, self._rates[2]],
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[1, self._kernel_size, 1],
              strides=[1, self._strides[1], 1],
              dilation_rate=[1, self._rates[1], 1],
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
      conv_layer_params.append(
          dict(
              filters=self._filters,
              kernel_size=[self._kernel_size, 1, 1],
              strides=[self._strides[0], 1, 1],
              dilation_rate=[self._rates[0], 1, 1],
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
          ))
    else:
      raise ValueError('Unsupported conv_type: {}'.format(self._conv_type))
    return conv_layer_params

  def _build_norm_layer_params(self, conv_param):
    """Builds params for the norm layer after one conv layer."""
    return dict(
        axis=self._channel_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        scale=False,
        gamma_initializer='ones')

  def _build_activation_layer_params(self, conv_param):
    """Builds params for the activation layer after one conv layer."""
    return {}

  def _append_conv_layer(self, param):
    """Appends conv, normalization and activation layers."""
    self._parameterized_conv_layers.append(
        tf.keras.layers.Conv3D(
            padding='same',
            use_bias=False,
            kernel_regularizer=self._kernel_regularizer,
            **param,
        ))
    norm_layer_params = self._build_norm_layer_params(param)
    self._parameterized_conv_layers.append(self._norm(**norm_layer_params))

    relu_layer_params = self._build_activation_layer_params(param)
    self._parameterized_conv_layers.append(
        tf.keras.layers.Activation('relu', **relu_layer_params))

  def build(self, input_shape):
    self._parameterized_conv_layers = []
    for conv_layer_param in self._build_conv_layer_params(input_shape):
      self._append_conv_layer(conv_layer_param)
    super(ParameterizedConvLayer, self).build(input_shape)

  def call(self, inputs):
    x = inputs
    for layer in self._parameterized_conv_layers:
      x = layer(x)
    return x
