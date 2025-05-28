# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Keras-based transformer block layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import reuse_transformer


@parameterized.named_parameters(
    ('base', reuse_transformer.ReuseTransformer))
class ReuseTransformerLayerTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(ReuseTransformerLayerTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy('float32')

  def test_layer_creation(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor, _ = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  def test_layer_creation_with_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor, _ = test_layer([data_tensor, mask_tensor])
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  def test_layer_invocation(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)

    # Create a model from the test layer.
    model = tf_keras.Model(data_tensor, output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = np.random.random_sample(
        (batch_size, sequence_length, width))
    _ = model.predict(input_data)

  def test_layer_invocation_with_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = np.random.random_sample(
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
    input_data = np.random.random_sample(
        (batch_size, sequence_length, width))
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    output_tensor, _ = test_layer([input_data, mask_data])

    # The layer only attends to the first token and outputs the first token
    # embedding.
    new_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        output_range=1)
    _ = new_layer([input_data, mask_data])
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor, _ = new_layer([input_data, mask_data])
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=0.002, rtol=0.01)

  def test_layer_output_range_with_relative_pe(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu',
        use_relative_pe=True)
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = np.random.random_sample(
        (batch_size, sequence_length, width))
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    output_tensor, _ = test_layer([input_data, mask_data])

    # The layer only attends to the first token and outputs the first token
    # embedding.
    new_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        output_range=1,
        use_relative_pe=True)
    _ = new_layer([input_data, mask_data])
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor, _ = new_layer([input_data, mask_data])
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=0.002, rtol=0.01)

  def test_layer_output_range_without_mask(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048,
        inner_activation='relu', norm_first=True)
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = np.random.random_sample(
        (batch_size, sequence_length, width))
    output_tensor, _ = test_layer(input_data)

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
    new_output_tensor, _ = new_layer(input_data)
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=0.002, rtol=0.01)

  def test_layer_output_range_with_pre_norm(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048,
        inner_activation='relu', norm_first=True)
    sequence_length = 21
    width = 80

    batch_size = 6
    input_data = np.random.random_sample(
        (batch_size, sequence_length, width))
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    output_tensor, _ = test_layer([input_data, mask_data])

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
    new_output_tensor, _ = new_layer([input_data, mask_data])
    self.assertAllClose(
        new_output_tensor, output_tensor[:, 0:1, :], atol=0.002, rtol=0.01)

  def test_layer_invocation_with_float16_dtype(self, transformer_cls):
    tf_keras.mixed_precision.set_global_policy('mixed_float16')
    test_layer = transformer_cls(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu')
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = (np.random.random_sample(
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
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02))
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output, _ = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output.shape.as_list())

  def test_dynamic_layer_sequence(self, transformer_cls):
    test_layer = transformer_cls(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 3-dimensional input (the first dimension is implicit).
    width = 30
    input_tensor = tf_keras.Input(shape=(None, width))
    output_tensor, _ = test_layer(input_tensor)
    model = tf_keras.Model(input_tensor, output_tensor)

    input_length = 17
    input_data = np.ones((1, input_length, width))
    output_data = model.predict(input_data)

    self.assertAllEqual([1, input_length, width], output_data.shape)


class ReuseTransformerArgumentTest(tf.test.TestCase, parameterized.TestCase):

  def test_use_bias_norm_first(self):
    num_attention_heads = 2
    hidden_size = 16
    encoder_block = reuse_transformer.ReuseTransformer(
        num_attention_heads=num_attention_heads,
        inner_dim=32,
        inner_activation='relu',
        output_dropout=0.1,
        attention_dropout=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        inner_dropout=0.1,
        attention_initializer=tf_keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_mask]
    output, _ = encoder_block(inputs)
    self.assertEqual(output.shape, (2, 4, hidden_size))

  def test_get_config(self):
    num_attention_heads = 2
    encoder_block = reuse_transformer.ReuseTransformer(
        num_attention_heads=num_attention_heads,
        inner_dim=32,
        inner_activation='relu',
        output_dropout=0.1,
        attention_dropout=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        inner_dropout=0.1,
        attention_initializer=tf_keras.initializers.RandomUniform(
            minval=0., maxval=1.))
    encoder_block_config = encoder_block.get_config()
    new_encoder_block = reuse_transformer.ReuseTransformer.from_config(
        encoder_block_config)
    self.assertEqual(encoder_block_config, new_encoder_block.get_config())

  @parameterized.parameters({'attention_axes': None}, {'attention_axes': [1]},
                            {'attention_axes': [2]}, {'attention_axes': [1, 2]})
  def test_several_attention_axes(self, attention_axes):
    test_layer = reuse_transformer.ReuseTransformer(
        inner_dim=32,
        inner_activation='relu',
        output_dropout=0.1,
        attention_dropout=0.1,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        inner_dropout=0.1,
        num_attention_heads=10,
        attention_axes=attention_axes)
    num_rows = 21
    num_cols = 13
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(num_rows, num_cols, width))
    output_tensor, _ = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  @parameterized.named_parameters(
      ('plain_returnscore', False, False),
      ('plain_with_relative_pe', False, True),
      ('reuse_all_returnscore', True, False),
      ('reuse_all_with_relative_pe', True, True),
      ('reuse_5_returnscore', 5, False),
      ('reuse_5_with_relative_pe', 5, True),
  )
  def test_layer_invocation_with_mask(self, reuse_attention, use_relative_pe):
    test_layer = reuse_transformer.ReuseTransformer(
        num_attention_heads=10,
        inner_dim=2048,
        inner_activation='relu',
        reuse_attention=reuse_attention,
        use_relative_pe=use_relative_pe)
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    reuse_attention_scores = tf_keras.Input(
        shape=(10, sequence_length, sequence_length))
    output_tensor, _ = test_layer(
        [data_tensor, mask_tensor, reuse_attention_scores])

    # Create a model from the test layer.
    model = tf_keras.Model(
        [
            data_tensor,
            mask_tensor,
            reuse_attention_scores,
        ],
        output_tensor,
    )

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = np.random.random_sample(
        (batch_size, sequence_length, width))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    reuse_scores = np.random.rand(
        batch_size, 10, sequence_length, sequence_length)
    _ = model.predict([input_data, mask_data, reuse_scores])

  @parameterized.named_parameters(
      ('without_relative_pe_with_pe_max_seq_length_10', False, 10),
      ('with_relative_pe_with_pe_max_seq_length_10', True, 10),
      ('without_relative_pe_with_pe_max_seq_length_100', False, 100),
      ('with_relative_pe_with_pe_max_seq_length_100', True, 100))
  def test_layer_invocation_with_float16_with_relative_pe(
      self, use_relative_pe, pe_max_seq_length):
    tf_keras.mixed_precision.set_global_policy('mixed_float16')
    test_layer = reuse_transformer.ReuseTransformer(
        num_attention_heads=10, inner_dim=2048, inner_activation='relu',
        use_relative_pe=use_relative_pe, pe_max_seq_length=pe_max_seq_length)
    sequence_length = 21
    width = 80
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    # Create a 2-dimensional input (the first dimension is implicit).
    mask_tensor = tf_keras.Input(shape=(sequence_length, sequence_length))
    output_tensor = test_layer([data_tensor, mask_tensor])

    # Create a model from the test layer.
    model = tf_keras.Model([data_tensor, mask_tensor], output_tensor)

    # Invoke the model on test data. We can't validate the output data itself
    # (the NN is too complex) but this will rule out structural runtime errors.
    batch_size = 6
    input_data = (np.random.random_sample(
        (batch_size, sequence_length, width)))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])

if __name__ == '__main__':
  tf.test.main()
