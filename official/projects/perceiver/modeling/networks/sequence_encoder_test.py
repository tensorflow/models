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

"""Tests for sequence_encoder."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.perceiver.configs import encoders
from official.projects.perceiver.configs import perceiver
from official.projects.perceiver.modeling.layers import encoder
from official.projects.perceiver.modeling.networks import sequence_encoder


class SequenceEncoderTest(tf.test.TestCase):

  def _create_small_network(
      self,
      sequence_length,
      z_index_dim,
      d_latents,
      vocab_size=100):
    d_model = 64
    num_layers = 2
    encoder_cfg = perceiver.EncoderConfig(
        v_last_dim=d_latents,
        num_self_attends_per_block=num_layers)
    sequence_encoder_cfg = perceiver.SequenceEncoderConfig(
        d_model=d_model,
        d_latents=d_latents,
        z_index_dim=z_index_dim,
        max_seq_len=sequence_length,
        vocab_size=vocab_size,
        encoder=encoder_cfg)
    return encoders.build_encoder(sequence_encoder_cfg)

  def test_dict_outputs_network_creation(self):
    sequence_length = 21
    z_index_dim = 128
    d_latents = 48
    test_network = self._create_small_network(
        sequence_length=sequence_length,
        z_index_dim=z_index_dim,
        d_latents=d_latents)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["latent_output"]

    expected_data_shape = [None, z_index_dim, d_latents]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)

  def test_dict_outputs_network_invocation(self):
    num_types = 7
    vocab_size = 57
    sequence_length = 21
    z_index_dim = 128
    d_latents = 48
    test_network = self._create_small_network(
        sequence_length=sequence_length,
        z_index_dim=z_index_dim,
        d_latents=d_latents,
        vocab_size=vocab_size)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["latent_output"]

    # Create a model based off of this network:
    model = tf_keras.Model([word_ids, mask, type_ids], [data])

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], d_latents)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    sequence_length = 21
    vocab_size = 57
    d_model = 64
    d_latents = 48
    z_index_dim = 128
    num_layers = 2
    encoder_cfg = perceiver.EncoderConfig(
        v_last_dim=d_latents,
        num_self_attends_per_block=num_layers)
    sequence_encoder_config = perceiver.SequenceEncoderConfig(
        d_model=d_model,
        d_latents=d_latents,
        z_index_dim=z_index_dim,
        max_seq_len=sequence_length,
        vocab_size=vocab_size,
        encoder=encoder_cfg)
    encoder_ = encoder.Encoder(
        **sequence_encoder_config.encoder.as_dict())
    network = sequence_encoder.SequenceEncoder(
        encoder=encoder_,
        d_model=sequence_encoder_config.d_model,
        d_latents=sequence_encoder_config.d_latents,
        z_index_dim=sequence_encoder_config.z_index_dim,
        max_seq_len=sequence_encoder_config.max_seq_len,
        vocab_size=sequence_encoder_config.vocab_size,
        z_pos_enc_init_scale=sequence_encoder_config.z_pos_enc_init_scale,
        embedding_width=sequence_encoder_config.embedding_width,
        embedding_initializer_stddev=sequence_encoder_config
        .embedding_initializer_stddev,
        input_position_encoding_intializer_stddev=sequence_encoder_config
        .input_position_encoding_intializer_stddev)

    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)

    dict_outputs = network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["latent_output"]

    # Create a model based off of this network:
    # model =
    _ = tf_keras.Model([word_ids, mask, type_ids], [data])

    # TODO(b/222634115) make save work.
    # Tests model saving/loading.
    # model_path = self.get_temp_dir() + "/model"
    # model.save(model_path)
    # _ = tf_keras.models.load_model(model_path)

# TODO(b/222634115) add test coverage.

if __name__ == "__main__":
  tf.test.main()
