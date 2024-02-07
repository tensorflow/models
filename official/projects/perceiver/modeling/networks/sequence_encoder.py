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

"""Perceiver sequence encoder."""

from typing import Optional, Dict

import tensorflow as tf, tf_keras

from official.nlp.modeling import layers


class SequenceEncoder(tf_keras.layers.Layer):
  """Perceiver encoder for sequences.

  Assumes positional learned encoding for latent inputs and embeddings. Creates
  an embedding table with vocab size. It uses the perceiver encode processor
  to encode the input and process the latent representation. It can be
  pretrained on masked LM and reused for fine-tuning.

  Use `self.inputs` for inputs.
  """

  def __init__(self,
               encoder: tf_keras.layers.Layer,
               d_model: int,
               d_latents: int,
               z_index_dim: int,
               max_seq_len: int,
               vocab_size: int,
               z_pos_enc_init_scale: float = 0.02,
               embedding_width: Optional[int] = None,
               embedding_initializer_stddev: float = 0.02,
               input_position_encoding_intializer_stddev: float = 0.02,
               name: str = 'sequence_encoder',
               **kwargs):
    """Init.

    Args:
      encoder:
        Instance of perceiver `Encoder`.
      d_model:
        Last dimension size of the input and output tensors. e.g.
        `[batch_size, max_seq_len, d_model]`.
      d_latents:
        Last dimension size of the latent tensors. e.g.
        `[batch_size, z_index_dim, d_latents]`.
      z_index_dim:
        Second dimension size of the latent tensors. e.g.
        `[batch_size, z_index_dim, d_latents]`.
      max_seq_len:
        Second dimension size of the input and outputs tensors. e.g.
        `[batch_size, max_seq_len, d_model]`.
      vocab_size:
        Vocabulary size of the embedding table.
      z_pos_enc_init_scale:
        Latent array's positional encoding's truncated_normal initializer's
        `stddev`.
      embedding_width:
        Embedding dimension of the embedding table.
      embedding_initializer_stddev:
        `stddev` of `tf_keras.initializers.TruncatedNormal` used for the
        embedding table kernel initializer.
      input_position_encoding_intializer_stddev:
        `stddev` of `tf_keras.initializers.TruncatedNormal` used for the
        learned position embedding table kernel initializer.
      name:
        Sets the `tf_keras.layers.Layer` name.
      **kwargs:
        Any keyword arguments to pass through to `tf_keras.layers.Layer`.
    """
    super().__init__(**kwargs, name=name)

    self._embedding_width = embedding_width

    self._encoder = encoder

    self._d_model = d_model
    self._z_index_dim = z_index_dim
    self._d_latents = d_latents
    if self._embedding_width is None:
      self._embedding_width = self._d_model

    # Construct the embeddling layer for the sequence vocab.
    self._embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=self._embedding_width,
        initializer=tf_keras.initializers.TruncatedNormal(
            stddev=embedding_initializer_stddev),
        name='word_embeddings')

    # Construct the input positional encoding layer.
    self._input_pos_encoding = layers.PositionEmbedding(
        max_length=max_seq_len,
        initializer=tf_keras.initializers.TruncatedNormal(
            stddev=input_position_encoding_intializer_stddev),
        name='input_pos_encoding')

    # Construct the latent array initial state.
    self._z_pos_enc = layers.PositionEmbedding(
        max_length=z_index_dim,
        initializer=tf_keras.initializers.TruncatedNormal(
            stddev=z_pos_enc_init_scale),
        name='z_pos_enc')

    self.inputs = dict(
        input_word_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf_keras.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf_keras.Input(shape=(None,), dtype=tf.int32))

  def get_embedding_table(self) -> tf.Variable:
    """Get embedding table."""
    return self._embedding_layer.embeddings

  def call(self,
           inputs: Dict[str, tf.Tensor],
           training: Optional[bool] = None) -> Dict[str, tf.Tensor]:
    """Return encoded and processed latent output of inputs.

    Args:
      inputs:
        Expect inputs to be a dictionary of `input_word_ids` and `input_mask`.
      training:
        Flag to indicate training status.

    Returns:
      `Dict[str, tf.Tensor]` decoded output of latent vector via the query.
    """
    if not isinstance(inputs, dict):
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)
    word_ids = inputs['input_word_ids']
    input_mask = inputs.get('input_mask')

    word_embeddings = self._embedding_layer(word_ids)
    pos_encodings = self._input_pos_encoding(word_embeddings)
    embeddings = word_embeddings + pos_encodings

    tensor_for_shape = tf.ones(
        [tf.shape(embeddings)[0], self._z_index_dim, self._d_latents],
        dtype=embeddings.dtype)
    encoder_query = self._z_pos_enc(tensor_for_shape)

    z = self._encoder(
        [embeddings, encoder_query], input_mask=input_mask, training=training)
    return dict(latent_output=z)
