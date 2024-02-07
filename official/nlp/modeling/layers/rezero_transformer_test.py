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

"""Tests for Keras-based rezero-transformer block layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import rezero_transformer


class TransformerWithReZeroLayerTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(TransformerWithReZeroLayerTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy('float32')

  @parameterized.named_parameters(('no_share_attn_ffn', False),
                                  ('share_attn_ffn', True))
  def test_layer_invocation_with_float16_dtype(self, share_rezero):
    tf_keras.mixed_precision.set_global_policy('mixed_float16')
    test_layer = rezero_transformer.ReZeroTransformer(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu',
        share_rezero=share_rezero)
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
    input_data = (10 * np.random.random_sample(
        (batch_size, sequence_length, width)))
    # The attention mask should be of shape (batch, from_seq_len, to_seq_len),
    # which here is (batch, sequence_length, sequence_length)
    mask_data = np.random.randint(
        2, size=(batch_size, sequence_length, sequence_length))
    _ = model.predict([input_data, mask_data])

  def test_rezero_without_layer_norm(self):
    test_layer = rezero_transformer.ReZeroTransformer(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu',
        use_layer_norm=False)

    input_length, width = 16, 30
    input_tensor = tf_keras.Input(shape=(input_length, width))
    output_tensor = test_layer(input_tensor)
    model = tf_keras.Model(input_tensor, output_tensor)

    input_data = np.random.rand(2, input_length, width)
    test_layer._rezero_a.assign(1.0)
    test_layer.reset_rezero()
    output_data = model.predict(input_data)

    self.assertAllClose(input_data, output_data)

  def test_rezero_with_layer_norm(self):
    test_layer = rezero_transformer.ReZeroTransformer(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu',
        use_layer_norm=True)

    input_length, width = 16, 30
    input_tensor = tf_keras.Input(shape=(input_length, width))
    output_tensor = test_layer(input_tensor)
    model = tf_keras.Model(input_tensor, output_tensor)

    input_data = np.random.rand(2, input_length, width) + 2.0
    output_data = model.predict(input_data)
    input_data_normed = (input_data -
                         np.mean(input_data, axis=-1, keepdims=True)) / (
                             np.std(input_data, axis=-1, keepdims=True))

    self.assertAllClose(input_data_normed, output_data)

  def test_layer_output_range(self):
    test_layer = rezero_transformer.ReZeroTransformer(
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
    new_layer = rezero_transformer.ReZeroTransformer(
        num_attention_heads=10,
        intermediate_size=2048,
        intermediate_activation='relu',
        output_range=1)
    _ = new_layer([input_data, mask_data])
    new_layer.set_weights(test_layer.get_weights())
    new_output_tensor = new_layer([input_data, mask_data])
    self.assertAllClose(new_output_tensor, output_tensor[:, 0:1, :])

    output_tensor = test_layer([input_data, mask_data], output_range=1)
    self.assertAllClose(new_output_tensor, output_tensor, atol=5e-5, rtol=0.003)

  def test_separate_qkv(self):
    test_layer = rezero_transformer.ReZeroTransformer(
        num_attention_heads=2,
        intermediate_size=128,
        intermediate_activation='relu',
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02))
    # Forward path.
    q_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
    kv_tensor = tf.zeros([2, 8, 16], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 8], dtype=tf.float32)
    inputs = [q_tensor, kv_tensor, dummy_mask]
    output = test_layer(inputs)
    self.assertEqual(output.shape, q_tensor.shape)


if __name__ == '__main__':
  tf.test.main()
