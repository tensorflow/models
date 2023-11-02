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

"""Tests for encoder."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.perceiver.modeling.layers import encoder


class EncoderTest(tf.test.TestCase):

  def test_layer_creation(self):
    test_layer = encoder.Encoder(
        self_attention_num_heads=8,
        cross_attention_num_heads=8)
    sequence_length = 80
    embedding_width = 800
    lantent_length = 8
    latent_width = 80
    data_input = tf_keras.Input(
        shape=(sequence_length, embedding_width))
    latent_input = tf_keras.Input(
        shape=(lantent_length, latent_width))

    output_tensor = test_layer((data_input, latent_input))
    self.assertEqual(
        latent_input.shape.as_list(),
        output_tensor.shape.as_list())

  def test_layer_creation_with_mask(self):
    test_layer = encoder.Encoder(
        self_attention_num_heads=8,
        cross_attention_num_heads=8)
    sequence_length = 80
    embedding_width = 800
    lantent_length = 8
    latent_width = 80
    data_input = tf_keras.Input(
        shape=(sequence_length, embedding_width))
    latent_input = tf_keras.Input(
        shape=(lantent_length, latent_width))
    mask_tensor = tf_keras.Input(
        shape=(sequence_length),
        dtype=tf.int32)
    output_tensor = test_layer(
        (data_input, latent_input),
        input_mask=mask_tensor)
    self.assertEqual(
        latent_input.shape.as_list(),
        output_tensor.shape.as_list())

  def test_layer_invocation(self):
    test_layer = encoder.Encoder(
        self_attention_num_heads=8,
        cross_attention_num_heads=8)
    sequence_length = 80
    embedding_width = 800
    lantent_length = 8
    latent_width = 80
    data_input = tf_keras.Input(
        shape=(sequence_length, embedding_width))
    latent_input = tf_keras.Input(
        shape=(lantent_length, latent_width))
    mask_tensor = tf_keras.Input(
        shape=(sequence_length),
        dtype=tf.int32)

    output_tensor = test_layer(
        (data_input, latent_input),
        input_mask=mask_tensor)

    # Create a model from the test layer.
    model = tf_keras.Model(
        ((data_input, latent_input), mask_tensor),
        output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, embedding_width))
    mask_data = tf.ones((batch_size, sequence_length), dtype=tf.int32)
    latent_data = tf.ones((batch_size, lantent_length, latent_width),
                          dtype=tf.float32)
    _ = model.predict(((input_data, latent_data), mask_data))

  def test_self_attention_widening_factor(self):
    last_dim = 160
    self_attention_widening_factor = 2
    test_layer = encoder.Encoder(
        self_attention_widening_factor=self_attention_widening_factor,
        v_last_dim=last_dim)

    some_sequence_length = 80
    some_embedding_width = 800
    some_lantent_length = 8
    some_latent_width = last_dim
    data_input = tf_keras.Input(
        shape=(some_sequence_length, some_embedding_width))
    latent_input = tf_keras.Input(
        shape=(some_lantent_length, some_latent_width))
    mask_tensor = tf_keras.Input(shape=(some_sequence_length), dtype=tf.int32)
    test_layer((data_input, latent_input), input_mask=mask_tensor)
    value = test_layer._self_attention_encoder_blocks[
        0]._intermediate_dense.get_config()['output_shape'].pop()
    self.assertEqual(last_dim * self_attention_widening_factor, value)

  def test_cross_attention_widening_factor(self):
    last_dim = 160
    cross_attention_widening_factor = 2
    test_layer = encoder.Encoder(
        cross_attention_widening_factor=cross_attention_widening_factor,
        v_last_dim=last_dim)

    some_sequence_length = 80
    some_embedding_width = 800
    some_lantent_length = 8
    some_latent_width = last_dim
    data_input = tf_keras.Input(
        shape=(some_sequence_length, some_embedding_width))
    latent_input = tf_keras.Input(
        shape=(some_lantent_length, some_latent_width))
    mask_tensor = tf_keras.Input(shape=(some_sequence_length), dtype=tf.int32)
    test_layer((data_input, latent_input), input_mask=mask_tensor)
    value = test_layer._cross_attention_encoder_block._intermediate_dense.get_config(
    )['output_shape'].pop()
    self.assertEqual(last_dim * cross_attention_widening_factor, value)

  def test_self_attention_num_heads(self):
    # TODO(b/222634115) parameterize test.
    self_attention_num_heads = 16
    test_layer = encoder.Encoder(
        self_attention_num_heads=self_attention_num_heads)

    some_sequence_length = 80
    some_embedding_width = 800
    some_lantent_length = 8
    some_latent_width = 64
    data_input = tf_keras.Input(
        shape=(some_sequence_length, some_embedding_width))
    latent_input = tf_keras.Input(
        shape=(some_lantent_length, some_latent_width))
    mask_tensor = tf_keras.Input(shape=(some_sequence_length), dtype=tf.int32)
    test_layer((data_input, latent_input), input_mask=mask_tensor)
    value = test_layer._self_attention_encoder_blocks[
        0]._attention_layer.get_config()['num_heads']
    self.assertEqual(self_attention_num_heads, value)

  def test_cross_attention_num_heads(self):
    # TODO(b/222634115) parameterize test.
    cross_attention_num_heads = 16
    test_layer = encoder.Encoder(
        cross_attention_num_heads=cross_attention_num_heads)

    some_sequence_length = 80
    some_embedding_width = 800
    some_lantent_length = 8
    some_latent_width = 64
    data_input = tf_keras.Input(
        shape=(some_sequence_length, some_embedding_width))
    latent_input = tf_keras.Input(
        shape=(some_lantent_length, some_latent_width))
    mask_tensor = tf_keras.Input(shape=(some_sequence_length), dtype=tf.int32)
    test_layer((data_input, latent_input), input_mask=mask_tensor)
    value = test_layer._cross_attention_encoder_block._attention_layer.get_config(
    )['num_heads']
    self.assertEqual(cross_attention_num_heads, value)

  def test_num_self_attends_per_block(self):
    # TODO(b/222634115) parameterize test.
    num_self_attends_per_block = 3
    test_layer = encoder.Encoder(
        num_self_attends_per_block=num_self_attends_per_block)

    some_sequence_length = 80
    some_embedding_width = 800
    some_lantent_length = 8
    some_latent_width = 64
    data_input = tf_keras.Input(
        shape=(some_sequence_length, some_embedding_width))
    latent_input = tf_keras.Input(
        shape=(some_lantent_length, some_latent_width))
    mask_tensor = tf_keras.Input(shape=(some_sequence_length), dtype=tf.int32)
    test_layer((data_input, latent_input), input_mask=mask_tensor)
    self.assertLen(
        test_layer._self_attention_encoder_blocks,
        num_self_attends_per_block)

  # TODO(b/222634115) num_blocks
  # TODO(b/222634115) qk_last_dim validations
  # TODO(b/222634115) v_last_dim validations
  # TODO(b/222634115) dropout_prob validation
  # TODO(b/222634115) dropout_attn_prob validation
  # TODO(b/222634115) att_init_scale validation
  # TODO(b/222634115) dense_init_scale validation
  # TODO(b/222634115) cross_attention_use_query_residual validation
  #             (value passed correctly)
  # TODO(b/222634115) norm_epsilon
  # TODO(b/222634115) check latent dims
  # TODO(b/222634115) make cross att mask validation when input_mask is None
  # TODO(b/222634115) make cross att mask validation when input_mask is not None

if __name__ == '__main__':
  tf.test.main()
