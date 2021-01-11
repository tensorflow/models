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
"""Bert encoder network."""
# pylint: disable=g-classes-have-attributes

import collections
from absl import logging
import tensorflow as tf

from official.nlp.keras_nlp import layers


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
class BertEncoder(tf.keras.Model):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

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
    attention_dropout: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yeilds the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    embedding_layer: An optional Layer instance which will be called to
     generate embeddings for the input word IDs.
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
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      output_range=None,
      embedding_width=None,
      embedding_layer=None,
      **kwargs):
    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')

    if embedding_width is None:
      embedding_width = hidden_size

    if embedding_layer is None:
      embedding_layer_inst = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=initializer,
          name='word_embeddings')
    else:
      embedding_layer_inst = embedding_layer
    word_embeddings = embedding_layer_inst(word_ids)

    # Always uses dynamic slicing for simplicity.
    position_embedding_layer = layers.PositionEmbedding(
        initializer=initializer,
        max_length=max_sequence_length,
        name='position_embedding')
    position_embeddings = position_embedding_layer(word_embeddings)
    type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=True,
        name='type_embeddings')
    type_embeddings = type_embedding_layer(type_ids)

    embeddings = tf.keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings])

    embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    embeddings = embedding_norm_layer(embeddings)
    embeddings = (tf.keras.layers.Dropout(rate=output_dropout)(embeddings))

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    if embedding_width != hidden_size:
      embedding_projection = tf.keras.layers.experimental.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=initializer,
          name='embedding_projection')
      embeddings = embedding_projection(embeddings)
    else:
      embedding_projection = None

    transformer_layers = []
    data = embeddings
    attention_mask = layers.SelfAttentionMask()(data, mask)
    encoder_outputs = []
    for i in range(num_layers):
      if i == num_layers - 1 and output_range is not None:
        transformer_output_range = output_range
      else:
        transformer_output_range = None
      layer = layers.TransformerEncoderBlock(
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          output_range=transformer_output_range,
          kernel_initializer=initializer,
          name='transformer/layer_%d' % i)
      transformer_layers.append(layer)
      data = layer([data, attention_mask])
      encoder_outputs.append(data)

    last_encoder_output = encoder_outputs[-1]
    # Applying a tf.slice op (through subscript notation) to a Keras tensor
    # like this will create a SliceOpLambda layer. This is better than a Lambda
    # layer with Python code, because that is fundamentally less portable.
    first_token_tensor = last_encoder_output[:, 0, :]
    pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')
    cls_output = pooler_layer(first_token_tensor)

    outputs = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=cls_output,
        encoder_outputs=encoder_outputs,
    )

    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(BertEncoder, self).__init__(
        inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)

    config_dict = {
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
    }

    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self._pooler_layer = pooler_layer
    self._transformer_layers = transformer_layers
    self._embedding_norm_layer = embedding_norm_layer
    self._embedding_layer = embedding_layer_inst
    self._position_embedding_layer = position_embedding_layer
    self._type_embedding_layer = type_embedding_layer
    if embedding_projection is not None:
      self._embedding_projection = embedding_projection

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return dict(self._config._asdict())

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
