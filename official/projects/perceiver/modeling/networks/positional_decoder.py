# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Perceiver networks."""

import tensorflow as tf

from official.nlp.modeling import layers


class PositionalDecoder(tf.keras.layers.Layer):
  """Perceiver Positional Decoder Network.

  Creates a position encoding for queries and composes basic decoder.
  e.g. the positional decoder can be used to do MLM, classification, or
  regression.

  Currently only supports positional decoding.

  Use `self.inputs` for inputs.

  Attributes:
    inputs: A `Dict[Text, tf.keras.Input]` with `latent_output` and
      `input_mask`. The shape of `latent_output` is shape
      `(z_index_dim, d_latents)` with dtype `tf.float32` and `input_mask` is
       shape `(None)` with dtype `tf.int32`.
  """

  def __init__(self,
               decoder,
               output_index_dim,
               z_index_dim,
               d_latents,
               d_model,
               position_encoding_intializer_stddev=0.02,
               name='positional_decoder',
               **kwargs):
    """Init.

    Args:
      decoder:
        Instance of perceiver `Decoder`.
      output_index_dim:
        Sequence length for the query encoding.
      z_index_dim:
        Latent index dimension.
      d_latents:
        Latent last dimension.
      d_model:
        Model last dimension.
      position_encoding_intializer_stddev:
        `stddev` of `tf.keras.initializers.TruncatedNormal` used for the
        learned position embedding table kernel initializer.
      name:
        Sets the `tf.keras.layers.Layer` name.
      **kwargs:
        Any keyword arguments to pass through to `tf.keras.layers.Layer`.
    """
    super().__init__(**kwargs, name=name)

    self._decoder = decoder
    self._output_index_dim = output_index_dim
    self._z_index_dim = z_index_dim
    self._d_latents = d_latents
    self._d_model = d_model

    self._output_pos_enc = self._create_decoder_query(
        position_encoding_intializer_stddev)

    self.inputs = dict(
        latent_output=tf.keras.Input(
            shape=(self._z_index_dim, self._d_latents),
            dtype=tf.float32),
        input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32))

  def _create_decoder_query(self, position_encoding_intializer_stddev):
    """Create the position encoding for the output query."""
    return layers.PositionEmbedding(
        max_length=self._output_index_dim,
        name='decoder_pos_enc',
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=position_encoding_intializer_stddev))

  def call(self, inputs, training=None):
    """Return decoded output of latent vector.

    Uses the positional encoding as query for the decoder and uses the
    `latent_output` as key-value for the decoder.
    Args:
      inputs:
        A `Dict[Text, tf.keras.Input]` with `latent_output` and
        `input_mask`. The shape of `latent_output` is shape
        `(z_index_dim, d_latents)` with dtype `tf.float32` and `input_mask` is
        shape `(None)` with dtype `tf.int32`.
      training:
        Flag to indicate training status. Default is `None`. It is passed to
        the decoder as is.

    Returns:
      `Dict[Text, tf.Tensor]` decoded `sequence_output` of a latent vector.
    """
    if not isinstance(inputs, dict):
      raise ValueError(f'Unexpected inputs type to {self.__class__}.')

    latent_output = inputs['latent_output']
    query_mask = inputs.get('input_mask')
    decoder_query = self._output_pos_enc(tf.ones(
        (tf.shape(latent_output)[0], self._output_index_dim, self._d_model),
        dtype=latent_output.dtype))
    z = latent_output

    sequence_output = self._decoder(
        [decoder_query, z],
        query_mask=query_mask,
        training=training)
    return dict(sequence_output=sequence_output)
