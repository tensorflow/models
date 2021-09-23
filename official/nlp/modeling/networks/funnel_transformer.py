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

"""Funnel Transformer network."""
# pylint: disable=g-classes-have-attributes
from typing import Union, Sequence
from absl import logging
import tensorflow as tf

from official.nlp import keras_nlp


def _pool_and_concat(data, unpool_length: int, strides: Union[Sequence[int],
                                                              int],
                     axes: Union[Sequence[int], int]):
  """Pools the data along a given axis with stride.

  It also skips first unpool_length elements.

  Args:
    data: Tensor to be pooled.
    unpool_length: Leading elements to be skipped.
    strides: Strides for the given axes.
    axes: Axes to pool the Tensor.

  Returns:
    Pooled and concatenated Tensor.
  """
  # Wraps the axes as a list.
  if isinstance(axes, int):
    axes = [axes]
  if isinstance(strides, int):
    strides = [strides] * len(axes)
  else:
    if len(strides) != len(axes):
      raise ValueError('The lengths of strides and axes need to match.')

  for axis, stride in zip(axes, strides):
    # Skips first `unpool_length` tokens.
    unpool_tensor_shape = [slice(None)] * axis + [slice(None, unpool_length)]
    unpool_tensor = data[unpool_tensor_shape]
    # Pools the second half.
    pool_tensor_shape = [slice(None)] * axis + [
        slice(unpool_length, None, stride)
    ]
    pool_tensor = data[pool_tensor_shape]
    data = tf.concat((unpool_tensor, pool_tensor), axis=axis)
  return data


@tf.keras.utils.register_keras_serializable(package='Text')
class FunnelTransformerEncoder(tf.keras.layers.Layer):
  """Funnel Transformer-based encoder network.

  Funnel Transformer Implementation of https://arxiv.org/abs/2006.03236.
  This implementation utilizes the base framework with Bert
  (https://arxiv.org/abs/1810.04805).
  Its output is compatible with `BertEncoder`.

  Args:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    inner_dim: The output dimension of the first Dense layer in a two-layer
      feedforward network for each transformer.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    output_dropout: Dropout probability for the post-attention and output
      dropout.
    attention_dropout: The dropout rate to use for the attention layers within
      the transformer layers.
    pool_stride: An int or a list of ints. Pooling stride(s) to compress the
      sequence length. If set to int, each layer will have the same stride size.
      If set to list, the number of elements needs to match num_layers.
    unpool_length: Leading n tokens to be skipped from pooling.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yields the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    embedding_layer: An optional Layer instance which will be called to generate
      embeddings for the input word IDs.
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized.
  """

  def __init__(
      self,
      vocab_size,
      hidden_size=768,
      num_layers=12,
      num_attention_heads=12,
      max_sequence_length=512,
      type_vocab_size=16,
      inner_dim=3072,
      inner_activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
      output_dropout=0.1,
      attention_dropout=0.1,
      pool_stride=2,
      unpool_length=0,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      output_range=None,
      embedding_width=None,
      embedding_layer=None,
      norm_first=False,
      **kwargs):
    super().__init__(**kwargs)
    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    if embedding_width is None:
      embedding_width = hidden_size

    if embedding_layer is None:
      self._embedding_layer = keras_nlp.layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=initializer,
          name='word_embeddings')
    else:
      self._embedding_layer = embedding_layer

    self._position_embedding_layer = keras_nlp.layers.PositionEmbedding(
        initializer=initializer,
        max_length=max_sequence_length,
        name='position_embedding')

    self._type_embedding_layer = keras_nlp.layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=True,
        name='type_embeddings')

    self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    self._embedding_dropout = tf.keras.layers.Dropout(
        rate=output_dropout, name='embedding_dropout')

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    self._embedding_projection = None
    if embedding_width != hidden_size:
      self._embedding_projection = tf.keras.layers.experimental.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=initializer,
          name='embedding_projection')

    self._transformer_layers = []
    self._attention_mask_layer = keras_nlp.layers.SelfAttentionMask(
        name='self_attention_mask')
    for i in range(num_layers):
      layer = keras_nlp.layers.TransformerEncoderBlock(
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          output_range=output_range if i == num_layers - 1 else None,
          kernel_initializer=initializer,
          name='transformer/layer_%d' % i)
      self._transformer_layers.append(layer)

    self._pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')
    if isinstance(pool_stride, int):
      # TODO(b/197133196): Pooling layer can be shared.
      pool_strides = [pool_stride] * num_layers
    else:
      if len(pool_stride) != num_layers:
        raise ValueError('Lengths of pool_stride and num_layers are not equal.')
      pool_strides = pool_stride
    self._att_input_pool_layers = []
    for layer_pool_stride in pool_strides:
      att_input_pool_layer = tf.keras.layers.MaxPooling1D(
          pool_size=layer_pool_stride,
          strides=layer_pool_stride,
          padding='same',
          name='att_input_pool_layer')
      self._att_input_pool_layers.append(att_input_pool_layer)

    self._pool_strides = pool_strides  # This is a list here.
    self._unpool_length = unpool_length

    self._config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'inner_dim': inner_dim,
        'inner_activation': tf.keras.activations.serialize(activation),
        'output_dropout': output_dropout,
        'attention_dropout': attention_dropout,
        'initializer': tf.keras.initializers.serialize(initializer),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
        'norm_first': norm_first,
        'pool_stride': pool_stride,
        'unpool_length': unpool_length,
    }

  def call(self, inputs):
    # inputs are [word_ids, mask, type_ids]
    if isinstance(inputs, (list, tuple)):
      logging.warning('List inputs to  %s are discouraged.', self.__class__)
      if len(inputs) == 3:
        word_ids, mask, type_ids = inputs
      else:
        raise ValueError('Unexpected inputs to %s with length at %d.' %
                         (self.__class__, len(inputs)))
    elif isinstance(inputs, dict):
      word_ids = inputs.get('input_word_ids')
      mask = inputs.get('input_mask')
      type_ids = inputs.get('input_type_ids')
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    word_embeddings = self._embedding_layer(word_ids)
    # absolute position embeddings
    position_embeddings = self._position_embedding_layer(word_embeddings)
    type_embeddings = self._type_embedding_layer(type_ids)

    embeddings = tf.keras.layers.add(
        [word_embeddings, position_embeddings, type_embeddings])
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout(embeddings)

    if self._embedding_projection is not None:
      embeddings = self._embedding_projection(embeddings)

    attention_mask = self._attention_mask_layer(embeddings, mask)

    encoder_outputs = []
    x = embeddings
    # TODO(b/195972228): attention_mask can be co-generated with pooling.
    attention_mask = _pool_and_concat(
        attention_mask,
        unpool_length=self._unpool_length,
        strides=self._pool_strides[0],
        axes=[1])
    for i, layer in enumerate(self._transformer_layers):
      # Pools layer for compressing the query length.
      pooled_inputs = self._att_input_pool_layers[i](
          x[:, self._unpool_length:, :])
      query_inputs = tf.concat(
          values=(tf.cast(
              x[:, :self._unpool_length, :],
              dtype=pooled_inputs.dtype), pooled_inputs),
          axis=1)
      x = layer([query_inputs, x, attention_mask])
      # Pools the corresponding attention_mask.
      if i < len(self._transformer_layers) - 1:
        attention_mask = _pool_and_concat(
            attention_mask,
            unpool_length=self._unpool_length,
            strides=[self._pool_strides[i+1], self._pool_strides[i]],
            axes=[1, 2])
      encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooled_output = self._pooler_layer(first_token_tensor)

    return dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=pooled_output,
        encoder_outputs=encoder_outputs)

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return dict(self._config)

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'embedding_layer' in config and config['embedding_layer'] is not None:
      warn_string = (
          'You are reloading a model that was saved with a '
          'potentially-shared embedding layer object. If you contine to '
          'train this model, the embedding layer will no longer be shared. '
          'To work around this, load the model outside of the Keras API.')
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)
