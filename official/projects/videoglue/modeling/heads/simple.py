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

"""Constructs simple task head layers."""

from typing import Any, Mapping, Optional, Union

import tensorflow as tf, tf_keras
from official.modeling import tf_utils
from official.vision.modeling.backbones import vit


class AddTemporalPositionEmbs(tf_keras.layers.Layer):
  """Adds learned temporal positional embeddings to the video features."""

  def __init__(self,
               posemb_init: Optional[tf_keras.initializers.Initializer] = None,
               **kwargs):
    """Constructs Postional Embedding module.

    Args:
      posemb_init: The positional embedding initializer.
      **kwargs: other args.
    """
    super().__init__(**kwargs)
    self.posemb_init = posemb_init

  def build(self, inputs_shape: Union[tf.TensorShape, list[int]]) -> None:
    pos_emb_length = inputs_shape[1]
    pos_emb_shape = (1, pos_emb_length, inputs_shape[-1])
    self.pos_embedding = self.add_weight(
        'pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    pos_embedding = self.pos_embedding
    # inputs.shape is (batch_size, temporal_len, spatial_len, emb_dim).
    pos_embedding = tf.cast(pos_embedding, inputs.dtype)
    _, t, _, c = inputs.shape
    inputs = tf.reshape(pos_embedding, [-1, t, 1, c]) + inputs
    return inputs


class MLP(tf_keras.layers.Layer):
  """Constructs the Multi-Layer Perceptron head."""

  def __init__(
      self,
      num_hidden_layers: int,
      num_hidden_channels: int,
      num_output_channels: int,
      use_sync_bn: bool,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 1e-5,
      activation: Optional[str] = None,
      normalize_inputs: bool = False,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
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
      kernel_regularizer: tf_keras.regularizers.Regularizer object.
      bias_regularizer: tf_keras.regularizers.Regularizer object.
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
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._layers = []
    # MLP hidden layers
    for _ in range(num_hidden_layers):
      self._layers.append(
          tf_keras.layers.Dense(
              num_hidden_channels,
              use_bias=False,
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer))
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
        'normalize_inputs': self._normalize_inputs,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]):
    """Factory constructor from config."""
    return cls(**config)


class AttentionPoolerClassificationHead(tf_keras.layers.Layer):
  """Head layer for attention pooling classification network.

  Applies pooling attention, dropout, and classifier projection. Expects input
  to be vector with shape [batch_size, n, num_channels].
  """

  def __init__(
      self,
      num_heads: int,
      hidden_size: int,
      num_classes: int,
      attention_dropout_rate: float = 0.,
      dropout_rate: float = 0.,
      kernel_initializer: str = 'HeNormal',
      kernel_regularizer: Optional[
          tf_keras.regularizers.Regularizer] = tf_keras.regularizers.L2(1.5e-5),
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      add_temporal_pos_embed: bool = False,
      **kwargs):
    """Implementation for video model classifier head.

    Args:
      num_heads: number of heads in attention layer.
      hidden_size: hidden size in attention layer.
      num_classes: number of output classes for the final logits.
      attention_dropout_rate: the dropout rate applied to the attention map.
      dropout_rate: the dropout rate applied to the head projection.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv operations.
      bias_regularizer: bias regularizer for the conv operations.
      add_temporal_pos_embed: whether to add temporal position embedding or not.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super().__init__(**kwargs)

    self._num_heads = num_heads
    self._num_classes = num_classes
    self._dropout_rate = dropout_rate
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._add_pooler_token = vit.TokenLayer(name='pooler_token')
    self._add_temporal_pos_embed = add_temporal_pos_embed
    if self._add_temporal_pos_embed:
      self._pos_embed = AddTemporalPositionEmbs(
          posemb_init=tf_keras.initializers.RandomNormal(stddev=0.02),
          name='posembed_final_learnt',
      )

    self._pooler_attention_layer_norm = tf_keras.layers.LayerNormalization(
        name='pooler_attention_layer_norm',
        axis=-1,
        epsilon=1e-6,
        dtype=tf.float32)

    self._pooler_attention_layer = tf_keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=(hidden_size // num_heads),
        value_dim=None,
        dropout=attention_dropout_rate,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='pooler_attention')

    self._dropout = tf_keras.layers.Dropout(dropout_rate)
    self._classifier = tf_keras.layers.Dense(
        num_classes,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the layer with the given inputs."""
    # Input Shape: [batch_size, n, input_channels]
    x = inputs
    tf.assert_rank(x, 4, message=
                   '(b, t, s, c) shaped inputs are required.')
    if self._add_temporal_pos_embed:
      x = self._pos_embed(x)
    _, s, t, c = x.shape
    x = tf.reshape(x, [-1, s * t, c])

    x = self._pooler_attention_layer_norm(x)
    x = self._add_pooler_token(x)
    pooler_token = x[:, 0:1, :]
    x = x[:, 1:, :]
    x = self._pooler_attention_layer(query=pooler_token, value=x,
                                     return_attention_scores=False)

    if self._dropout_rate and self._dropout_rate > 0:
      x = self._dropout(x)

    return self._classifier(tf.squeeze(x, axis=1))
