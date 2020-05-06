# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.engine import network  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations
from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class TransformerEncoder(network.Network):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  Arguments:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    sequence_length: The sequence length that this encoder expects. If None, the
      sequence length is dynamic; if an integer, the encoder will require
      sequences padded to this length.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    intermediate_size: The intermediate size for the transformer layers.
    activation: The activation to use for the transformer layers.
    dropout_rate: The dropout rate to use for the transformer layers.
    attention_dropout_rate: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    return_all_encoder_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers.
    output_range: the sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yeilds the full
      output.
  """

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               sequence_length=512,
               max_sequence_length=None,
               type_vocab_size=16,
               intermediate_size=3072,
               activation=activations.gelu,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
               return_all_encoder_outputs=False,
               output_range=None,
               **kwargs):
    activation = tf.keras.activations.get(activation)
    initializer = tf.keras.initializers.get(initializer)

    if not max_sequence_length:
      max_sequence_length = sequence_length
    self._self_setattr_tracking = False
    self._config_dict = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'sequence_length': sequence_length,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'intermediate_size': intermediate_size,
        'activation': tf.keras.activations.serialize(activation),
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'initializer': tf.keras.initializers.serialize(initializer),
        'return_all_encoder_outputs': return_all_encoder_outputs,
        'output_range': output_range,
    }

    word_ids = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='input_type_ids')

    self._embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=hidden_size,
        initializer=initializer,
        name='word_embeddings')
    word_embeddings = self._embedding_layer(word_ids)

    # Always uses dynamic slicing for simplicity.
    self._position_embedding_layer = layers.PositionEmbedding(
        initializer=initializer,
        use_dynamic_slicing=True,
        max_sequence_length=max_sequence_length,
        name='position_embedding')
    position_embeddings = self._position_embedding_layer(word_embeddings)

    type_embeddings = (
        layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=hidden_size,
            initializer=initializer,
            use_one_hot=True,
            name='type_embeddings')(type_ids))

    embeddings = tf.keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings])
    embeddings = (
        tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)(embeddings))
    embeddings = (
        tf.keras.layers.Dropout(rate=dropout_rate)(embeddings))

    self._transformer_layers = []
    data = embeddings
    attention_mask = layers.SelfAttentionMask()([data, mask])
    encoder_outputs = []
    for i in range(num_layers):
      if i == num_layers - 1 and output_range is not None:
        transformer_output_range = output_range
      else:
        transformer_output_range = None
      layer = layers.Transformer(
          num_attention_heads=num_attention_heads,
          intermediate_size=intermediate_size,
          intermediate_activation=activation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          output_range=transformer_output_range,
          kernel_initializer=initializer,
          name='transformer/layer_%d' % i)
      self._transformer_layers.append(layer)
      data = layer([data, attention_mask])
      encoder_outputs.append(data)

    first_token_tensor = (
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
            encoder_outputs[-1]))
    cls_output = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')(
            first_token_tensor)

    if return_all_encoder_outputs:
      outputs = [encoder_outputs, cls_output]
    else:
      outputs = [encoder_outputs[-1], cls_output]

    super(TransformerEncoder, self).__init__(
        inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_config(self):
    return self._config_dict

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
