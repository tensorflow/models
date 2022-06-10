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

"""Longformer encoder. Modified From huggingface/transformers."""

# pylint: disable=g-classes-have-attributes

from typing import Any, Callable, List, Optional, Union

from absl import logging
import tensorflow as tf

from official.modeling.tf_utils import get_shape_list
from official.nlp.modeling import layers
from official.projects.longformer.longformer_encoder_block import LongformerEncoderBlock


_Initializer = Union[str, tf.keras.initializers.Initializer]
_approx_gelu = lambda x: tf.keras.activations.gelu(x, approximate=True)


class LongformerEncoder(tf.keras.layers.Layer):
  """LongformerEncoder.

  Args:
    vocab_size: The size of the token vocabulary.
    attention_window: list of ints representing the window size for each layer.
    global_attention_size: the size of global attention used for each token.
    pad_token_id: the token id for the pad token
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
      vocab_size: int,
      attention_window: Union[List[int], int] = 512,
      global_attention_size: int = 0,
      pad_token_id: int = 1,
      hidden_size: int = 768,
      num_layers: int = 12,
      num_attention_heads: int = 12,
      max_sequence_length: int = 512,
      type_vocab_size: int = 16,
      inner_dim: int = 3072,
      inner_activation: Callable[..., Any] = _approx_gelu,
      output_dropout: float = 0.1,
      attention_dropout: float = 0.1,
      initializer: _Initializer = tf.keras.initializers.TruncatedNormal(
          stddev=0.02),
      output_range: Optional[int] = None,
      embedding_width: Optional[int] = None,
      embedding_layer: Optional[tf.keras.layers.Layer] = None,
      norm_first: bool = False,
      **kwargs):
    super().__init__(**kwargs)
    # Longformer args
    self._attention_window = attention_window
    self._global_attention_size = global_attention_size
    self._pad_token_id = pad_token_id

    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    if embedding_width is None:
      embedding_width = hidden_size

    if embedding_layer is None:
      self._embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=initializer,
          name='word_embeddings')
    else:
      self._embedding_layer = embedding_layer

    self._position_embedding_layer = layers.PositionEmbedding(
        initializer=initializer,
        max_length=max_sequence_length,
        name='position_embedding')

    self._type_embedding_layer = layers.OnDeviceEmbedding(
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
      self._embedding_projection = tf.keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=initializer,
          name='embedding_projection')

    self._transformer_layers = []
    self._attention_mask_layer = layers.SelfAttentionMask(
        name='self_attention_mask')
    for i in range(num_layers):
      layer = LongformerEncoderBlock(
          global_attention_size=global_attention_size,
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          attention_window=attention_window[i],
          layer_id=i,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          output_range=output_range if i == num_layers - 1 else None,
          kernel_initializer=initializer,
          name=f'transformer/layer_{i}')
      self._transformer_layers.append(layer)

    self._pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')

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
        'attention_window': attention_window,
        'global_attention_size': global_attention_size,
        'pad_token_id': pad_token_id,
    }
    self.inputs = dict(
        input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32))

  def call(self, inputs):
    word_embeddings = None
    if isinstance(inputs, dict):
      word_ids = inputs.get('input_word_ids')  # input_ids
      mask = inputs.get('input_mask')  # attention_mask
      type_ids = inputs.get('input_type_ids')  # token_type_ids
      word_embeddings = inputs.get('input_word_embeddings',
                                   None)  # input_embeds
    else:
      raise ValueError(f'Unexpected inputs type to {self.__class__}.')

    (
        padding_len,
        word_ids,
        mask,
        type_ids,
        word_embeddings,
    ) = self._pad_to_window_size(
        word_ids=word_ids,
        mask=mask,
        type_ids=type_ids,
        word_embeddings=word_embeddings,
        pad_token_id=self._pad_token_id)

    if word_embeddings is None:
      word_embeddings = self._embedding_layer(word_ids)
    # absolute position embeddings.
    position_embeddings = self._position_embedding_layer(word_embeddings)
    type_embeddings = self._type_embedding_layer(type_ids)

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout(embeddings)

    if self._embedding_projection is not None:
      embeddings = self._embedding_projection(embeddings)

    batch_size, seq_len = get_shape_list(mask)
    # create masks with fixed len global_attention_size
    mask = tf.transpose(
        tf.concat(
            values=[
                tf.ones(
                    (self._global_attention_size, batch_size), tf.int32) * 2,
                tf.transpose(mask)[self._global_attention_size:]
            ],
            axis=0))

    is_index_masked = tf.math.less(mask, 1)

    is_index_global_attn = tf.transpose(
        tf.concat(
            values=[
                tf.ones((self._global_attention_size, batch_size), tf.bool),
                tf.zeros((seq_len - self._global_attention_size, batch_size),
                         tf.bool)
            ],
            axis=0))

    # Longformer
    attention_mask = mask
    extended_attention_mask = tf.reshape(
        attention_mask, (tf.shape(mask)[0], tf.shape(mask)[1], 1, 1))
    attention_mask = tf.cast(
        tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0

    encoder_outputs = []
    x = embeddings
    # TFLongformerEncoder
    for layer in self._transformer_layers:
      x = layer([x, attention_mask, is_index_masked, is_index_global_attn])
      encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    if padding_len > 0:
      last_encoder_output = last_encoder_output[:, :-padding_len]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooled_output = self._pooler_layer(first_token_tensor)

    return dict(
        sequence_output=last_encoder_output,
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

  def _pad_to_window_size(
      self,
      word_ids,
      mask,
      type_ids,
      word_embeddings,
      pad_token_id,
  ):
    # padding
    attention_window = max(self._attention_window)

    assert (attention_window %
            2 == 0), ('`attention_window` should be an even value.'
                      f'Given {attention_window}')

    input_shape = get_shape_list(
        word_ids) if word_ids is not None else get_shape_list(word_embeddings)
    batch_size, seq_len = input_shape[:2]

    if seq_len is not None:
      padding_len = (attention_window -
                     seq_len % attention_window) % attention_window
    else:
      padding_len = 0

    paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])

    if word_ids is not None:
      word_ids = tf.pad(word_ids, paddings, constant_values=pad_token_id)

    if word_embeddings is not None:

      def pad_embeddings():
        word_ids_padding = tf.fill((batch_size, padding_len), self.pad_token_id)
        word_embeddings_padding = self._embedding_layer(word_ids_padding)
        return tf.concat([word_embeddings, word_embeddings_padding], axis=-2)

      word_embeddings = tf.cond(
          tf.math.greater(padding_len, 0), pad_embeddings,
          lambda: word_embeddings)

    mask = tf.pad(
        mask, paddings,
        constant_values=False)  # no attention on the padding tokens
    token_type_ids = tf.pad(
        type_ids, paddings, constant_values=0)  # pad with token_type_id = 0

    return (
        padding_len,
        word_ids,
        mask,
        token_type_ids,
        word_embeddings,
    )
