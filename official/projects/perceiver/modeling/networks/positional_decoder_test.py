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

"""Tests for positional_decoder."""

import tensorflow as tf, tf_keras

from official.projects.perceiver.configs import perceiver as cfg
from official.projects.perceiver.modeling.layers import decoder
from official.projects.perceiver.modeling.networks import positional_decoder


class PositionalDecoderTest(tf.test.TestCase):

  def test_dict_outputs_network_creation(self):
    sequence_length = 21
    z_index_dim = 8
    d_model = 64
    d_latents = 48
    decoder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents,
        num_heads=2)
    positional_decoder_cfg = cfg.PositionalDecoder(
        decoder=decoder_cfg,
        d_model=d_model,
        d_latents=d_latents,
        output_index_dim=sequence_length,
        z_index_dim=z_index_dim)

    decoder_ = decoder.Decoder(positional_decoder_cfg.decoder.as_dict())
    mlm_decoder = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=positional_decoder_cfg.output_index_dim,
        z_index_dim=positional_decoder_cfg.z_index_dim,
        d_latents=positional_decoder_cfg.d_latents,
        d_model=positional_decoder_cfg.d_model)

    # Create the inputs (note that the first dimension is implicit).
    latent_output = tf_keras.Input(
        shape=(z_index_dim, d_latents), dtype=tf.float32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = mlm_decoder(
        dict(latent_output=latent_output, input_mask=mask))
    data = dict_outputs["sequence_output"]

    expected_data_shape = [None, sequence_length, d_model]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    sequence_length = 21
    z_index_dim = 8
    d_model = 64
    d_latents = 48
    decoder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents,
        num_heads=2)
    positional_decoder_cfg = cfg.PositionalDecoder(
        decoder=decoder_cfg,
        d_model=d_model,
        d_latents=d_latents,
        output_index_dim=sequence_length,
        z_index_dim=z_index_dim)

    decoder_ = decoder.Decoder(positional_decoder_cfg.decoder.as_dict())
    mlm_decoder = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=positional_decoder_cfg.output_index_dim,
        z_index_dim=positional_decoder_cfg.z_index_dim,
        d_latents=positional_decoder_cfg.d_latents,
        d_model=positional_decoder_cfg.d_model)

    # Create the inputs (note that the first dimension is implicit).
    latent_output = tf_keras.Input(
        shape=(z_index_dim, d_latents), dtype=tf.float32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = mlm_decoder(
        dict(latent_output=latent_output, input_mask=mask))
    data = dict_outputs["sequence_output"]

    # Create a model based off of this network:
    # model =
    _ = tf_keras.Model([latent_output, mask], [data])

    # TODO(b/222634115) make save work.
    # Tests model saving/loading.
    # model_path = self.get_temp_dir() + "/model"
    # model.save(model_path)
    # _ = tf_keras.models.load_model(model_path)

# TODO(b/222634115) add test coverage.

if __name__ == "__main__":
  tf.test.main()
