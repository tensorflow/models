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

"""Tests for multi-query attention layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import multi_query_attention


class MultiQueryAttentionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("key_value_same_proj_mqa", 1, None, None, [40, 80]),
      ("key_value_different_proj_mqa", 1, 32, 60, [40, 60]),
      ("key_value_same_proj_gqa", 3, None, None, [40, 80]),
      ("key_value_different_proj_gqa", 3, 32, 60, [40, 60]),
  )
  def test_non_masked_attention(
      self, num_kv_heads, value_dim, output_shape, output_dims
  ):
    """Test that the attention layer can be created without a mask tensor."""
    test_layer = multi_query_attention.MultiHeadAttention(
        num_heads=12,
        num_kv_heads=num_kv_heads,
        key_dim=64,
        value_dim=value_dim,
        output_shape=output_shape,
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    value = tf_keras.Input(shape=(20, 80))
    output = test_layer(query=query, value=value)
    self.assertEqual(output.shape.as_list(), [None] + output_dims)

  @parameterized.named_parameters(
      ("_mqa", 1),
      ("_gqa", 3),
  )
  def test_non_masked_self_attention(self, num_kv_heads):
    """Test with one input (self-attenntion) and no mask tensor."""
    test_layer = multi_query_attention.MultiHeadAttention(
        num_heads=12, num_kv_heads=num_kv_heads, key_dim=64
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  @parameterized.named_parameters(
      ("_mqa", 1),
      ("_gqa", 3),
  )
  def test_attention_scores(self, num_kv_heads):
    """Test attention outputs with coefficients."""
    test_layer = multi_query_attention.MultiHeadAttention(
        num_heads=12, num_kv_heads=num_kv_heads, key_dim=64
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    output, coef = test_layer(query, query, return_attention_scores=True)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])
    self.assertEqual(coef.shape.as_list(), [None, 12, 40, 40])

  @parameterized.named_parameters(
      ("_mqa", 1),
      ("_gqa", 3),
  )
  def test_attention_scores_with_values(self, num_kv_heads):
    """Test attention outputs with coefficients."""
    test_layer = multi_query_attention.MultiHeadAttention(
        num_heads=12, num_kv_heads=num_kv_heads, key_dim=64
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    value = tf_keras.Input(shape=(60, 80))
    output, coef = test_layer(query, value, return_attention_scores=True)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])
    self.assertEqual(coef.shape.as_list(), [None, 12, 40, 60])

  @parameterized.named_parameters(
      ("with_bias_mqa", 1, True),
      ("no_bias_mqa", 1, False),
      ("with_bias_gqa", 2, True),
      ("no_bias_gqa", 2, False),
  )
  def test_masked_attention(self, num_kv_heads, use_bias):
    """Test with a mask tensor."""
    test_layer = multi_query_attention.MultiHeadAttention(
        num_heads=4, num_kv_heads=num_kv_heads, key_dim=2, use_bias=use_bias
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    batch_size = 3
    query = tf_keras.Input(shape=(4, 8))
    value = tf_keras.Input(shape=(2, 8))
    mask_tensor = tf_keras.Input(shape=(4, 2))
    output = test_layer(query=query, value=value, attention_mask=mask_tensor)

    # Create a model containing the test layer.
    model = tf_keras.Model([query, value, mask_tensor], output)

    # Generate data for the input (non-mask) tensors.
    from_data = 10 * np.random.random_sample((batch_size, 4, 8))
    to_data = 10 * np.random.random_sample((batch_size, 2, 8))

    # Invoke the data with a random set of mask data. This should mask at
    # least one element.
    mask_data = np.random.randint(2, size=(batch_size, 4, 2))
    masked_output_data = model.predict([from_data, to_data, mask_data])

    # Invoke the same data, but with a null mask (where no elements are
    # masked).
    null_mask_data = np.ones((batch_size, 4, 2))
    unmasked_output_data = model.predict([from_data, to_data, null_mask_data])

    # Because one data is masked and one is not, the outputs should not be
    # the same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    # Tests the layer with three inputs: Q, K, V.
    key = tf_keras.Input(shape=(2, 8))
    output = test_layer(
        query, value=value, key=key, attention_mask=mask_tensor
    )
    model = tf_keras.Model([query, value, key, mask_tensor], output)

    masked_output_data = model.predict(
        [from_data, to_data, to_data, mask_data]
    )
    unmasked_output_data = model.predict(
        [from_data, to_data, to_data, null_mask_data]
    )
    # Because one data is masked and one is not, the outputs should not be
    # the same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    if use_bias:
      self.assertLen(test_layer._query_dense.trainable_variables, 2)
      self.assertLen(test_layer._output_dense.trainable_variables, 2)
    else:
      self.assertLen(test_layer._query_dense.trainable_variables, 1)
      self.assertLen(test_layer._output_dense.trainable_variables, 1)

  @parameterized.named_parameters(
      ("_mqa", 1),
      ("_gqa", 2),
  )
  def test_masked_attention_with_scores(self, num_kv_heads):
    """Test with a mask tensor."""
    test_layer = multi_query_attention.MultiHeadAttention(
        num_heads=4, num_kv_heads=num_kv_heads, key_dim=2
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    batch_size = 3
    query = tf_keras.Input(shape=(4, 8))
    value = tf_keras.Input(shape=(2, 8))
    mask_tensor = tf_keras.Input(shape=(4, 2))
    output = test_layer(query=query, value=value, attention_mask=mask_tensor)

    # Create a model containing the test layer.
    model = tf_keras.Model([query, value, mask_tensor], output)

    # Generate data for the input (non-mask) tensors.
    from_data = 10 * np.random.random_sample((batch_size, 4, 8))
    to_data = 10 * np.random.random_sample((batch_size, 2, 8))

    # Invoke the data with a random set of mask data. This should mask at
    # least one element.
    mask_data = np.random.randint(2, size=(batch_size, 4, 2))
    masked_output_data = model.predict([from_data, to_data, mask_data])

    # Invoke the same data, but with a null mask (where no elements are
    # masked).
    null_mask_data = np.ones((batch_size, 4, 2))
    unmasked_output_data = model.predict([from_data, to_data, null_mask_data])

    # Because one data is masked and one is not, the outputs should not be
    # the same.
    self.assertNotAllClose(masked_output_data, unmasked_output_data)

    # Create a model containing attention scores.
    output, scores = test_layer(
        query=query,
        value=value,
        attention_mask=mask_tensor,
        return_attention_scores=True,
    )
    model = tf_keras.Model([query, value, mask_tensor], [output, scores])
    masked_output_data_score, masked_score = model.predict(
        [from_data, to_data, mask_data]
    )
    unmasked_output_data_score, unmasked_score = model.predict(
        [from_data, to_data, null_mask_data]
    )
    self.assertNotAllClose(masked_output_data_score, unmasked_output_data_score)
    self.assertAllClose(masked_output_data, masked_output_data_score)
    self.assertAllClose(unmasked_output_data, unmasked_output_data_score)
    self.assertNotAllClose(masked_score, unmasked_score)


if __name__ == "__main__":
  tf.test.main()
