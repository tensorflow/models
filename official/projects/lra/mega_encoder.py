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

"""Mega encoder. Modified From huggingface/transformers."""

# pylint: disable=g-classes-have-attributes

from typing import Any, Callable, Optional, Union

from absl import logging
import tensorflow as tf, tf_keras
import tensorflow_models as tfm

from official.modeling import tf_utils
from official.projects.lra.moving_average_gated_attention import MovingAverageGatedAttention


layers = tfm.nlp.layers

_Initializer = Union[str, tf_keras.initializers.Initializer]
_approx_gelu = lambda x: tf_keras.activations.gelu(x, approximate=True)


@tf_keras.utils.register_keras_serializable(package='Text')
class MegaEncoder(tf_keras.layers.Layer):
  """MegaEncoder.

  Args:
    vocab_size: The size of the token vocabulary.
    embedding_width: The number of embedding dimensions.
    intermediate_size: The number of dimension for MLP layers.
    num_layers: The number of transformer layers.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    zdim: hidden dimension for gates used in MEGA Layer.
    hdim: hidden dimension used in MEGA Layer.
    ndim: number of EMA used in MEGA layer.
    activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    bidirectional: Whether to use bidirectional EMA.
    dropout: Dropout probability for the post-attention and output dropout.
    attention_dropout: The dropout rate to use for the attention layers within
      the transformer layers.
    hidden_dropout: The dropout rate to use for hidden states in MEGA.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yields the full
      output.
    embedding_layer: An optional Layer instance which will be called to generate
      embeddings for the input word IDs.
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized.
  """

  def __init__(
      self,
      vocab_size: int,
      embedding_width: int = 128,
      intermediate_size: int = 256,
      num_layers: int = 12,
      max_sequence_length: int = 512,
      type_vocab_size: int = 16,
      zdim: int = 64,
      hdim: int = 256,
      ndim: int = 16,
      activation='silu',
      bidirectional=False,
      dropout: float = 0.0,
      attention_dropout: float = 0.0,
      hidden_dropout: float = 0.0,
      inner_activation: Callable[..., Any] = _approx_gelu,
      initializer: _Initializer = tf_keras.initializers.TruncatedNormal(
          stddev=0.02
      ),
      output_range: Optional[int] = None,
      embedding_layer: Optional[tf_keras.layers.Layer] = None,
      norm_first: bool = False,
      hidden_size: Optional[int] = None,
      **kwargs
  ):
    super().__init__(**kwargs)
    # Mega args
    initializer = tf_keras.initializers.get(initializer)

    if embedding_layer is None:
      self._embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=initializer,
          name='word_embeddings',
      )
    else:
      self._embedding_layer = embedding_layer

    self._position_embedding_layer = layers.PositionEmbedding(
        initializer=initializer,
        max_length=max_sequence_length,
        name='position_embedding',
    )

    self._type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=True,
        name='type_embeddings',
    )

    self._embedding_norm_layer = tf_keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32
    )

    self._embedding_dropout = tf_keras.layers.Dropout(
        rate=dropout, name='embedding_dropout'
    )

    self._transformer_layers = []
    self._attention_mask_layer = layers.SelfAttentionMask(
        name='self_attention_mask'
    )
    for _ in range(num_layers):
      layer = MovingAverageGatedAttention(
          embed_dim=embedding_width,
          zdim=zdim,
          hdim=hdim,
          ndim=ndim,
          intermediate_size=intermediate_size,
          inner_activation=inner_activation,
          dropout=dropout,
          attention_dropout=attention_dropout,
          hidden_dropout=hidden_dropout,
          activation=activation,
          bidirectional=bidirectional,
          prenorm=norm_first,
          max_positions=max_sequence_length,
          use_bias=True,
          return_attention_scores=False,
          kernel_initializer=tf_utils.clone_initializer(initializer),
      )
      self._transformer_layers.append(layer)
    self._num_layers = num_layers
    self._pooler_layer = tf_keras.layers.Dense(
        units=embedding_width,
        activation='silu',
        kernel_initializer=initializer,
        name='pooler_transform',
    )
    self._config = {
        'vocab_size': vocab_size,
        'num_layers': num_layers,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'zdim': zdim,
        'hdim': hdim,
        'ndim': ndim,
        'activation': activation,
        'bidirectional': bidirectional,
        'dropout': dropout,
        'attention_dropout': attention_dropout,
        'hidden_dropout': hidden_dropout,
        'inner_activation': tf_keras.activations.serialize(inner_activation),
        'initializer': tf_keras.initializers.serialize(initializer),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
        'norm_first': norm_first,
    }
    self.inputs = dict(
        input_word_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf_keras.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
    )

  def call(self, inputs):
    word_embeddings = None

    if isinstance(inputs, dict):
      if 'input_word_ids' in inputs.keys():
        word_ids = inputs.get('input_word_ids')
        mask = inputs.get('input_mask')
        type_ids = inputs.get('input_type_ids', None)
        word_embeddings = inputs.get('input_word_embeddings', None)
      elif 'left_word_ids' in inputs.keys():
        word_ids = inputs.get('left_word_ids')
        mask = inputs.get('left_mask')
      elif 'right_word_ids' in inputs.keys():
        word_ids = inputs.get('right_word_ids')
        mask = inputs.get('right_mask')
      dense_inputs = inputs.get('dense_inputs', None)
      dense_mask = inputs.get('dense_mask', None)
    elif isinstance(inputs, list):
      ## Dual Encoder Tasks
      word_ids, mask = inputs
      type_ids = None
      dense_inputs, dense_mask = None, None
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    if type_ids is None:
      type_ids = tf.zeros_like(mask)

    if word_embeddings is None:
      word_embeddings = self._embedding_layer(word_ids)

    if dense_inputs is not None:
      mask = tf.concat([mask, dense_mask], axis=1)

    embeddings = self._embedding_norm_layer(word_embeddings)
    embeddings = self._embedding_dropout(embeddings)

    encoder_outputs = []
    x = embeddings

    for l in range(self._num_layers):
      if x.shape[0] is None:
        pass
      else:
        x = self._transformer_layers[l]([x, mask])
      encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    avg_token_tensor = tf.math.reduce_mean(last_encoder_output, axis=1)
    pooled_output = self._pooler_layer(avg_token_tensor)

    output = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=pooled_output,
        encoder_outputs=encoder_outputs,
    )

    return output

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
          'To work around this, load the model outside of the Keras API.'
      )
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)

  def _get_embeddings(
      self,
      word_ids: tf.Tensor,
      type_ids: tf.Tensor,
      word_embeddings: Optional[tf.Tensor],
      dense_inputs: Optional[tf.Tensor],
      dense_type_ids: Optional[tf.Tensor],
  ) -> tf.Tensor:
    if word_embeddings is None:
      word_embeddings = self._embedding_layer(word_ids)

    if dense_inputs is not None:
      # Concat the dense embeddings at sequence end.
      word_embeddings = tf.concat([word_embeddings, dense_inputs], axis=1)
      type_ids = tf.concat([type_ids, dense_type_ids], axis=1)

    type_embeddings = self._type_embedding_layer(type_ids)

    # absolute position embeddings.
    position_embeddings = self._position_embedding_layer(word_embeddings)
    return word_embeddings + position_embeddings + type_embeddings
