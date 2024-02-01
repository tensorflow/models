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

"""Constructs simple heads."""

from typing import Any, Mapping, Optional

import tensorflow as tf, tf_keras
from official.modeling import tf_utils


class MLP(tf_keras.layers.Layer):
  """Constructs the Multi-Layer Perceptron head."""

  def __init__(self,
               num_hidden_layers: int,
               num_hidden_channels: int,
               num_output_channels: int,
               use_sync_bn: bool,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 1e-5,
               activation: Optional[str] = None,
               normalize_inputs: bool = False,
               **kwargs):
    """Multi-Layer Perceptron initialization.

    Args:
      num_hidden_layers: the number of hidden layers in the MLP.
      num_hidden_channels: the number of hidden nodes.
      num_output_channels: the number of final output nodes.
      use_sync_bn: whether to use sync batch norm.
      norm_momentum: the batch norm momentum.
      norm_epsilon: the batch norm epsilon.
      activation: the activation function.
      normalize_inputs: whether to normalize inputs.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._num_hidden_layers = num_hidden_layers
    self._num_hidden_channels = num_hidden_channels
    self._num_output_channels = num_output_channels
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._activation = activation
    self._normalize_inputs = normalize_inputs

    self._layers = []
    # MLP hidden layers
    for _ in range(num_hidden_layers):
      self._layers.append(
          tf_keras.layers.Dense(num_hidden_channels, use_bias=False))
      if use_sync_bn:
        self._layers.append(
            tf_keras.layers.experimental.SyncBatchNormalization(
                momentum=norm_momentum,
                epsilon=norm_epsilon))
      else:
        self._layers.append(
            tf_keras.layers.BatchNormalization(
                momentum=norm_momentum,
                epsilon=norm_epsilon))
      if activation is not None:
        self._layers.append(tf_utils.get_activation(activation))

    # Projection head
    self._layers.append(tf_keras.layers.Dense(num_output_channels))

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Forward calls with N-D inputs tensor."""
    if self._normalize_inputs:
      inputs = tf.nn.l2_normalize(inputs, axis=-1)

    for layer in self._layers:
      if isinstance(layer, tf_keras.layers.Layer):
        inputs = layer(inputs, training=training)
      else:  # activation
        inputs = layer(inputs)
    return inputs

  def get_config(self) -> Mapping[str, Any]:
    """Gets class config parameters."""
    config_dict = {
        'num_hidden_layer': self._num_hidden_layer,
        'num_hidden_channels': self._num_hidden_channels,
        'num_output_channels': self._num_output_channels,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'normalize_inputs': self._normalize_inputs}
    return config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]):
    """Factory constructor from config."""
    return cls(**config)
