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
"""Tests for Keras-based transformer block layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.keras_nlp.layers.transformer_encoder_block import TransformerEncoderBlock


@keras_parameterized.run_all_keras_modes
@parameterized.named_parameters(
    ('base', TransformerEncoderBlock))
class TransformerEncoderBlockLayerTest(keras_parameterized.TestCase):

  def tearDown(self):
    super(TransformerEncoderBlockLayerTest, self).tearDown()
    tf.keras.mixed_precision.experimental.set_policy('float32')

  def test_layer_creation(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  def test_layer_creation_with_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  def test_layer_invocation(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)

    # Create a model from the test layer.
    model = tf.keras.Model(data_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    _ = model.predict(input_data)

  def test_layer_invocation_with_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])

  def test_layer_output_range(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    output_tensor = test_layer([input_data, mask_data])

    # The layer only attends to the first token and outputs the first token
    # embedding.
    new_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        output_range=1)
    _ = new_layer([input_data, mask_data])
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor = new_layer([input_data, mask_data])
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

  def test_layer_output_range_without_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048,
        inner_activation='relu', norm_first=True)
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    output_tensor = test_layer(input_data)

    # The layer only attends to the first token and outputs the first token
    # embedding.
    new_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        output_range=1,
        norm_first=True)
    _ = new_layer(input_data)
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor = new_layer(input_data)
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

  def test_layer_output_range_with_pre_norm(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048,
        inner_activation='relu', norm_first=True)
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    output_tensor = test_layer([input_data, mask_data])

    # The layer only attends to the first token and outputs the first token
    # embedding.
    new_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        output_range=1,
        norm_first=True)
    _ = new_layer([input_data, mask_data])
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor = new_layer([input_data, mask_data])
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

  def test_layer_invocation_with_float16_dtype(self, transformer_cls):
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf.keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = (10 * np.random.random_sample(
        (batch_size, sequence_length, width)))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])

  def test_transform_with_initializer(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output.shape.as_list())

  def test_dynamic_layer_sequence(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 3-dimensional input (the first dimension is implicit).
    width = 30
    input_tensor = tf.keras.Input(shape=(None, width))
    output_tensor = test_layer(input_tensor)
    model = tf.keras.Model(input_tensor, output_tensor)

    input_length = 17
    input_data = np.ones((1, input_length, width))
    output_data = model.predict(input_data)

    self.assertAllEqual([1, input_length, width], output_data.shape)

  def test_separate_qkv(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=2,
        inner_dim=128,
        inner_activation='relu',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    # Forward path.
    q_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    kv_tensor = tf.zeros([2, 8, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
    inputs = [q_tensor, kv_tensor, dummy_mask]
    output = test_layer(inputs)
    self.assertEqual(output.shape, q_tensor.shape)


@keras_parameterized.run_all_keras_modes
class TransformerArgumentTest(keras_parameterized.TestCase):

  def test_use_bias_norm_first(self):
    num_attention_heads = 2
    hidden_size = 16
    encoder_block = TransformerEncoderBlock(
        num_attention_heads=num_attention_heads,
        inner_dim=32,
        inner_activation='relu',
        output_dropout=0.1,
        attention_dropout=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        inner_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_mask]
    output = encoder_block(inputs)
    self.assertEqual(output.shape, (2, 4, hidden_size))

  def test_get_config(self):
    num_attention_heads = 2
    encoder_block = TransformerEncoderBlock(
        num_attention_heads=num_attention_heads,
        inner_dim=32,
        inner_activation='relu',
        output_dropout=0.1,
        attention_dropout=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        inner_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    encoder_block_config = encoder_block.get_config()
    new_encoder_block = TransformerEncoderBlock.from_config(
        encoder_block_config)
    self.assertEqual(encoder_block_config, new_encoder_block.get_config())


if __name__ == '__main__':
  tf.test.main()
