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

"""Definition for Transformer decoder."""

from typing import Mapping, Optional, Union, List, Sequence
from absl import logging

import tensorflow as tf, tf_keras


def _get_shape(x: tf.Tensor):
  """Helper function to return shape of a given tensor."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class DecoderUnit(tf_keras.layers.Layer):
  """Constructs the decoder MHA module used in Transformer layers."""

  def __init__(
      self,
      num_channels: int,
      use_bias: bool,
      dropout_rate: float,
      activation: str,
      layer_norm_epsilon: float,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):

    super().__init__(**kwargs)
    self._num_channels = num_channels
    self._use_bias = use_bias
    self._dropout_rate = dropout_rate
    self._activation = activation
    self._layer_norm_epsilon = layer_norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Builds the layer.

    Args:
      input_shape: the input shape for the keras tensor.
    """
    # Query, key, and value mapping.
    self.layer_q = tf_keras.layers.Dense(
        self._num_channels,
        use_bias=self._use_bias,
        activation=None,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='query')
    self.layer_k = tf_keras.layers.Dense(
        self._num_channels,
        use_bias=self._use_bias,
        activation=None,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='key')
    self.layer_v = tf_keras.layers.Dense(
        self._num_channels,
        use_bias=self._use_bias,
        activation=None,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='value')

    self.dropout = tf_keras.layers.Dropout(self._dropout_rate)
    # Note here is a different behavior for contrib_layers.layer_norm and
    # tf_keras.layers.LayerNormalization, where by default, the former
    # calculates mean/variance across all axes except the first one
    # (batch axis), while the latter one computes statistics only on the last
    # axis.
    self.layer_norm = tf_keras.layers.LayerNormalization(
        epsilon=self._layer_norm_epsilon,
        name='layer_norm')

    self.ffn1 = tf_keras.layers.Dense(
        self._num_channels,
        use_bias=self._use_bias,
        activation=self._activation,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='ffn1')
    self.ffn2 = tf_keras.layers.Dense(
        self._num_channels,
        use_bias=self._use_bias,
        activation=None,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='ffn2')

    super().build(input_shape)

  def call(self,
           query: tf.Tensor,
           memory: Optional[tf.Tensor],
           training: bool = False) -> Mapping[str, tf.Tensor]:
    """Forward pass of the Transformer decoder unit.

    Args:
      query: the input query tensor.
      memory: the input memory tensor for key/value pairs. If None,
        self-attention will be performed.
      training: whether in training mode.

    Returns:
      outputs: the output dictionary contains 'hidden_states' and
        'attention weights' matrix.
    """
    if memory is None:
      memory = query

    tensor_q = self.layer_q(query)  # (bs, qlen, inner_dim)
    tensor_k = self.layer_k(memory)  # (bs, klen, inner_dim)
    tensor_v = self.layer_v(memory)  # (bs, klen, inner_dim)

    scores = tf.matmul(tensor_q, tensor_k, transpose_b=True)
    # Scales attention_scores.
    dk = tf.cast(_get_shape(tensor_k)[-1], dtype=scores.dtype)
    scores = scores / tf.math.sqrt(dk)

    # Shape: (bs, seq_len, seq_len)
    attention_weights = tf.nn.softmax(scores, axis=-1)
    # Shape: (bs, seq_len, dim_per_head)
    attention_features = tf.matmul(attention_weights, tensor_v)
    # Shape: (bs, seq_len, seq_len)
    attention_features = self.dropout(attention_features, training=training)

    hidden_states = attention_features + tensor_q
    hidden_states = self.layer_norm(hidden_states)

    # Shape: (bs, seq_len, out_dim)
    hidden_states = self.ffn1(hidden_states)
    hidden_states = self.ffn2(hidden_states)

    outputs = {
        'hidden_states': hidden_states,
        'attention_weights': attention_weights,
    }
    return outputs


class TransformerDecoderLayer(tf_keras.layers.Layer):
  """Constructs the main Transformer decoder module which includes MHA + FFN."""

  def __init__(
      self,
      num_channels: int,
      num_heads: int,
      use_bias: bool,
      activation: str,
      dropout_rate: float,
      layer_norm_epsilon: float,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      name: str = 'decoder_layer',
      **kwargs):
    super().__init__(name=name)

    self._num_channels = num_channels
    self._num_heads = num_heads
    self._use_bias = use_bias
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._layer_norm_epsilon = layer_norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._name = name

    self._mha_units = []
    for i in range(num_heads):
      self._mha_units.append(
          DecoderUnit(
              num_channels=num_channels,
              use_bias=use_bias,
              dropout_rate=dropout_rate,
              activation=activation,
              layer_norm_epsilon=layer_norm_epsilon,
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer,
              name='mha_{}'.format(i)))

  def call(
      self,
      inputs: tf.Tensor,
      memory: Optional[tf.Tensor] = None,
      training: bool = False
  ) -> Mapping[str, Union[tf.Tensor, Sequence[tf.Tensor]]]:
    """Forward pass of the Transformer decoder layer.

    Args:
      inputs: the input query tensor.
      memory: the input memory tensor for key/value pairs. If None,
        self-attention will be performed.
      training: whether in training mode.

    Returns:
      outputs: the output dictionary contains 'hidden_states' and
        'attention weights' matrix.
    """

    if memory is None:
      logging.info('No memory tokens are provided. Performing self-attention '
                   'on input tokens in TransfomerDecoder.')

    all_head_feats = []
    all_head_attentions = []
    for i in range(self._num_heads):
      outputs = self._mha_units[i](
          query=inputs, memory=memory, training=training)
      all_head_feats.append(outputs['hidden_states'])
      all_head_attentions.append(outputs['attention_weights'])

    outputs = {
        'hidden_states': tf.concat(all_head_feats, axis=-1),
        'attention_weights': all_head_attentions,
    }
    return outputs


class TransformerDecoder(tf_keras.layers.Layer):
  """Constructs the final Transformer decoder stack."""

  def __init__(
      self,
      num_channels: int,
      num_layers: int,
      num_heads: int,
      use_bias: bool,
      activation: str,
      dropout_rate: float,
      layer_norm_epsilon: float,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      name: str = 'transformer_decoder',
      **kwargs):
    super().__init__(name=name)

    self._num_channels = num_channels
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._use_bias = use_bias
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._layer_norm_epsilon = layer_norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._layers = []
    for n in range(self._num_layers):
      self._layers.append(
          TransformerDecoderLayer(
              num_channels=num_channels,
              num_heads=num_heads,
              use_bias=use_bias,
              activation=activation,
              dropout_rate=dropout_rate,
              layer_norm_epsilon=layer_norm_epsilon,
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer,
              name='layer_{}'.format(n)))

  def call(self,
           inputs: tf.Tensor,
           memory: Optional[tf.Tensor] = None,
           training: bool = False) -> Mapping[str, Sequence[tf.Tensor]]:
    """Forward pass of the Transformer decoder.

    Args:
      inputs: the input query tensor.
      memory: the input memory tensor for key/value pairs. If None,
        self-attention will be performed.
      training: whether in training mode.

    Returns:
      outputs: the output dictionary contains 'hidden_states' and
        'attention weights' matrix.
    """

    all_hidden_states = ()
    all_attentions = ()

    memory_shape = _get_shape(memory)
    memory = tf.reshape(memory, [memory_shape[0], -1, memory_shape[-1]])
    hidden_states = inputs

    for layer in self._layers:
      layer_outputs = layer(inputs=hidden_states,
                            memory=memory,
                            training=training)

      # layer_outputs is a dictionary with the following keys:
      # hidden_states, self_attention_weights
      hidden_states = layer_outputs['hidden_states']
      all_attentions += (layer_outputs['attention_weights'],)

    # Add last layer
    all_hidden_states += (hidden_states,)

    outputs = {
        'hidden_states': all_hidden_states,
        'attention_weights': all_attentions,
    }

    return outputs
