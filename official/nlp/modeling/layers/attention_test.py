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
"""Tests for the attention layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import attention


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class MultiHeadAttentionTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ("key_value_same_proj", None, None, [40, 80]),
      ("key_value_different_proj", 32, 60, [40, 60]),
  )
  def test_non_masked_attention(self, value_size, output_shape, output_dims):
    """Test that the attention layer can be created without a mask tensor."""
    test_layer = attention.MultiHeadAttention(
        num_heads=12,
        key_size=64,
        value_size=value_size,
        output_shape=output_shape)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf.keras.Input(shape=(40, 80))
    value = tf.keras.Input(shape=(20, 80))
    output = test_layer([query, value])
    self.assertEqual(output.shape.as_list(), [None] + output_dims)

  def test_non_masked_self_attention(self):
    """Test with one input (self-attenntion) and no mask tensor."""
    test_layer = attention.MultiHeadAttention(num_heads=12, key_size=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf.keras.Input(shape=(40, 80))
    output = test_layer([query, query])
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  @parameterized.parameters(True, False)
  def test_masked_attention(self, use_bias):
    """Test with a mask tensor."""
    test_layer = attention.MultiHeadAttention(
        num_heads=2, key_size=2, use_bias=use_bias)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf.keras.Input(shape=(4, 8))
    value = tf.keras.Input(shape=(2, 8))
    mask_tensor = tf.keras.Input(shape=(4, 2))
    output = test_layer([query, value], mask_tensor)

    # Create a model containing the test layer.
    model = tf.keras.Model([query, value, mask_tensor], output)

    # Generate data for the input (non-mask) tensors.
    from_data = 10 * np.random.random_sample((3, 4, 8))
    to_data = 10 * np.random.random_sample((3, 2, 8))

    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(2, size=(3, 4, 2))
    masked_output_data = model.predict([from_data, to_data, mask_data])

    # Invoke the same data, but with a null mask (where no elements are masked).
    null_mask_data = np.ones((3, 4, 2))
    unmasked_output_data = model.predict([from_data, to_data, null_mask_data])

    # Because one data is masked and one is not, the outputs should not be the
    # same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    # Tests the layer with three inputs: Q, K, V.
    key = tf.keras.Input(shape=(2, 8))
    output = test_layer([query, value, key], mask_tensor)
    model = tf.keras.Model([query, value, key, mask_tensor], output)

    masked_output_data = model.predict([from_data, to_data, to_data, mask_data])
    unmasked_output_data = model.predict(
        [from_data, to_data, to_data, null_mask_data])
    # Because one data is masked and one is not, the outputs should not be the
    # same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = attention.MultiHeadAttention(
        num_heads=12,
        key_size=64,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf.keras.Input(shape=(40, 80))
    output = test_layer([query, query])
    self.assertEqual(output.shape.as_list(), [None, 40, 80])


def _create_cache(batch_size, init_decode_length, num_heads, head_size):
  return {
      "key":
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32),
      "value":
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32)
  }


@keras_parameterized.run_all_keras_modes
class CachedAttentionTest(keras_parameterized.TestCase):

  def test_masked_attention(self):
    """Test with a mask tensor."""
    num_heads, head_size = 2, 2
    # Create a 3-dimensional input (the first dimension is implicit).
    from_seq_length = 4
    batch_size = 3
    # GPU/CPU case.
    init_decode_length = 0
    # Directly tests the keras layer.
    cache = _create_cache(batch_size, init_decode_length, num_heads, head_size)
    layer = attention.CachedAttention(num_heads=num_heads, key_size=head_size)

    # Generate data for the input (non-mask) tensors.
    from_data = tf.zeros((batch_size, from_seq_length, 8), dtype=np.float32)
    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    mask_data = np.random.randint(
        2, size=(batch_size, from_seq_length, from_seq_length))
    masked_output_data, cache = layer([from_data, from_data, mask_data, cache])
    self.assertEqual(masked_output_data.shape, (3, 4, 8))
    self.assertEqual(cache["value"].shape, (3, 4, 2, 2))

    # Tests inputs without cache.
    masked_output_data, cache = layer([from_data, from_data, mask_data])
    self.assertEqual(masked_output_data.shape, (3, 4, 8))
    self.assertIsNone(cache)

  def test_padded_decode(self):
    """Test with a mask tensor."""
    num_heads, head_size = 2, 2
    from_seq_length = 4
    # TPU decoding should pre-allocate the entire sequence.
    batch_size = 3
    init_decode_length = from_seq_length

    # Directly tests the keras layer.
    cache = _create_cache(batch_size, init_decode_length, num_heads, head_size)
    layer = attention.CachedAttention(num_heads=num_heads, key_size=head_size)

    # Generate data for the input (non-mask) tensors.
    from_data = tf.zeros((batch_size, from_seq_length, 8), dtype=np.float32)
    decode_loop_step = 2
    mask_data = np.random.randint(
        2, size=(batch_size, from_seq_length, from_seq_length), dtype=np.int32)
    # Testing the invocation directly as Keras cannot consume inputs correctly.
    masked_output_data, cache = layer([from_data, from_data, mask_data, cache],
                                      decode_loop_step=decode_loop_step)
    self.assertEqual(masked_output_data.shape, (3, 4, 8))
    self.assertEqual(cache["value"].shape, (3, 4, 2, 2))


if __name__ == "__main__":
  tf.test.main()
