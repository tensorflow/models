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

"""Tests for decoder."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.perceiver.modeling.layers import decoder


class PerceiverBasicDecoderTest(tf.test.TestCase):

  def test_layer_creation(self):
    sequence_length = 80
    embedding_width = 800
    test_layer = decoder.Decoder(
        output_last_dim=embedding_width,
        num_heads=8)
    lantent_length = 8
    latent_width = 80
    query_input = tf_keras.Input(
        shape=(sequence_length, embedding_width))
    latent_input = tf_keras.Input(
        shape=(lantent_length, latent_width))

    output_tensor = test_layer((query_input, latent_input))
    self.assertEqual(
        query_input.shape.as_list(),
        output_tensor.shape.as_list())

  def test_layer_creation_with_mask(self):
    embedding_width = 800
    sequence_length = 80
    test_layer = decoder.Decoder(
        output_last_dim=embedding_width,
        num_heads=8)
    lantent_length = 8
    latent_width = 80
    query_input = tf_keras.Input(
        shape=(sequence_length, embedding_width))
    latent_input = tf_keras.Input(
        shape=(lantent_length, latent_width))
    mask_tensor = tf_keras.Input(
        shape=(sequence_length),
        dtype=tf.int32)
    output_tensor = test_layer(
        (query_input, latent_input),
        query_mask=mask_tensor)
    self.assertEqual(
        query_input.shape.as_list(),
        output_tensor.shape.as_list())

  def test_layer_invocation(self):
    embedding_width = 800
    sequence_length = 80
    test_layer = decoder.Decoder(
        output_last_dim=embedding_width,
        num_heads=8)
    lantent_length = 8
    latent_width = 80
    query_input = tf_keras.Input(
        shape=(sequence_length, embedding_width))
    latent_input = tf_keras.Input(
        shape=(lantent_length, latent_width))
    mask_tensor = tf_keras.Input(
        shape=(sequence_length),
        dtype=tf.int32)
    output_tensor = test_layer(
        (query_input, latent_input),
        query_mask=mask_tensor)

    # Create a model from the test layer.
    model = tf_keras.Model(
        ((query_input, latent_input), mask_tensor),
        output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    latent_data = 10 * np.random.random_sample(
        (batch_size, lantent_length, latent_width))
    mask_data = tf.ones((batch_size, sequence_length), dtype=tf.int32)
    query_data = tf.ones(
        (batch_size, sequence_length, embedding_width),
        dtype=tf.float32)
    _ = model.predict(((query_data, latent_data), mask_data))

# TODO(b/222634115) Add tests to validate logic and dims.

if __name__ == "__main__":
  tf.test.main()
