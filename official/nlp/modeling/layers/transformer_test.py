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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import transformer


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
@parameterized.named_parameters(('base', transformer.Transformer),
                                ('xla', transformer.CompiledTransformer))
class TransformerLayerTest(keras_parameterized.TestCase):

  def tearDown(self):
    super(TransformerLayerTest, self).tearDown()
    tf.keras.mixed_precision.experimental.set_policy('float32')

  def test_layer_creation(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  def test_layer_creation_with_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  def test_layer_creation_with_incorrect_mask_fails(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf.keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf.keras.Input(shape=(sequence_length, sequence_length - 3))
    with self.assertRaisesRegex(ValueError, 'When passing a mask tensor.*'):
      _ = test_layer([data_tensor, mask_tensor])

  def test_layer_invocation(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
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
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
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

  def test_layer_output_range(self, _):
    # XLA has an obvious numeric issue in this test case.
    test_layer = transformer.Transformer(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    output_tensor = test_layer([input_data, mask_data])

    # The layer only attends to the first token and outputs the first token
    # embeeding.
    new_layer = transformer.Transformer(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu',
        output_range=1)
    _ = new_layer([input_data, mask_data])
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor = new_layer([input_data, mask_data])
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=5e-5, rtol=0.003)

  def test_layer_invocation_with_float16_dtype(self, transformer_cls):
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    test_layer = transformer_cls(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu')
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
        intermediate_size=2048,
        intermediate_activation='relu',
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
        intermediate_size=2048,
        intermediate_activation='relu',
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


@keras_parameterized.run_all_keras_modes
class TransformerArgumentTest(keras_parameterized.TestCase):

  def test_use_bias_norm_first(self):
    num_attention_heads = 2
    hidden_size = 16
    encoder_block = transformer.Transformer(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.1,
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
    encoder_block = transformer.Transformer(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    encoder_block_config = encoder_block.get_config()
    new_encoder_block = transformer.Transformer.from_config(
        encoder_block_config)
    self.assertEqual(encoder_block_config, new_encoder_block.get_config())


def _create_cache(batch_size, init_decode_length, num_heads, head_size):
  return {
      'key':
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32),
      'value':
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32)
  }


@keras_parameterized.run_all_keras_modes
class TransformerDecoderLayerTest(keras_parameterized.TestCase):

  def test_decoder_block_with_cache(self):
    num_attention_heads = 2
    hidden_size = 16
    decoder_block = transformer.TransformerDecoderLayer(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1)
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_tensor, dummy_mask, dummy_mask]
    cache = _create_cache(2, 0, num_attention_heads,
                          hidden_size // num_attention_heads)
    output, cache = decoder_block(inputs, cache)
    self.assertEqual(output.shape, (2, 4, hidden_size))
    self.assertEqual(cache['value'].shape, (2, 4, 2, 8))

  def test_use_bias_norm_first(self):
    num_attention_heads = 2
    hidden_size = 16
    decoder_block = transformer.TransformerDecoderLayer(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_tensor, dummy_mask, dummy_mask]
    output, _ = decoder_block(inputs)
    self.assertEqual(output.shape, (2, 4, hidden_size))

  def test_get_config(self):
    num_attention_heads = 2
    decoder_block = transformer.TransformerDecoderLayer(
        num_attention_heads=num_attention_heads,
        intermediate_size=32,
        intermediate_activation='relu',
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.1,
        attention_initializer=tf.keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    decoder_block_config = decoder_block.get_config()
    new_decoder_block = transformer.TransformerDecoderLayer.from_config(
        decoder_block_config)
    self.assertEqual(decoder_block_config, new_decoder_block.get_config())


if __name__ == '__main__':
  tf.test.main()
