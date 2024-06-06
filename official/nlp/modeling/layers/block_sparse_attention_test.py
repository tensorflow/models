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

"""Tests for block sparse attention layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import block_sparse_attention


class BlockSparseAttentionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("key_value_same_proj", None, None, [40, 80]),
      ("key_value_different_proj", 32, 60, [40, 60]),
  )
  def test_non_masked_attention(self, value_dim, output_shape, output_dims):
    """Test that the attention layer can be created without a mask tensor."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=12,
        key_dim=64,
        value_dim=value_dim,
        output_shape=output_shape,
        src_block_size=10,
        tgt_block_size=5,
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    value = tf_keras.Input(shape=(20, 80))
    output = test_layer(query=query, value=value)
    self.assertEqual(output.shape.as_list(), [None] + output_dims)

  def test_non_masked_self_attention(self):
    """Test with one input (self-attenntion) and no mask tensor."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=12, key_dim=64, src_block_size=10, tgt_block_size=10
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

  @parameterized.named_parameters(("with_bias", True), ("no_bias", False))
  def test_masked_attention(self, use_bias):
    """Test with a mask tensor."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=4, key_dim=2, use_bias=use_bias, src_block_size=2,
        tgt_block_size=1,
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

  def test_masked_attention_with_scores(self):
    """Test with a mask tensor."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=4, key_dim=2, src_block_size=2, tgt_block_size=1,
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

  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=12,
        key_dim=64,
        src_block_size=10,
        kernel_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02),
    )
    # Create a 3-dimensional input (the first dimension is implicit).
    query = tf_keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])

    # Make sure the sub layers have different kernel init value, and not
    # reusing the initializers.
    self.assertNotAllClose(
        tf_keras.backend.eval(test_layer._query_dense.kernel),
        tf_keras.backend.eval(test_layer._key_dense.kernel),
    )
    self.assertNotAllClose(
        tf_keras.backend.eval(test_layer._query_dense.kernel),
        tf_keras.backend.eval(test_layer._value_dense.kernel),
    )
    self.assertNotAllClose(
        tf_keras.backend.eval(test_layer._query_dense.kernel),
        tf_keras.backend.eval(test_layer._output_dense.kernel),
    )

  @parameterized.named_parameters(
      ("bfloat16", tf.bfloat16),
      ("float16", tf.float16),
      ("float32", tf.float32),
      ("float64", tf.float64),
  )
  def test_sublayer_dtypes(self, dtype):
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=12, key_dim=64, src_block_size=10, dtype=dtype
    )

    query = tf_keras.Input(shape=(40, 80), dtype=dtype)
    # Build the layer
    test_layer(query=query, value=query)

    self.assertEqual(test_layer._query_dense.dtype, dtype)
    self.assertEqual(test_layer._key_dense.dtype, dtype)
    self.assertEqual(test_layer._value_dense.dtype, dtype)
    self.assertEqual(test_layer._output_dense.dtype, dtype)

  def test_dropout(self):
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=2, key_dim=2, dropout=0.5, src_block_size=2, tgt_block_size=1,
    )

    # Generate data for the input (non-mask) tensors.
    from_data = tf_keras.backend.ones(shape=(32, 4, 8))
    to_data = tf_keras.backend.ones(shape=(32, 2, 8))
    train_out = test_layer(from_data, to_data, None, None, None, True)
    test_out = test_layer(from_data, to_data, None, None, None, False)

    # Output should be close when not in training mode,
    # and should not be close when enabling dropout in training mode.
    self.assertNotAllClose(
        tf_keras.backend.eval(train_out), tf_keras.backend.eval(test_out)
    )

  def test_query_mask_progagation(self):
    """Test automatic propagation of the query's mask."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=2,
        key_dim=2,
        src_block_size=2,
        tgt_block_size=1,
    )
    self.assertTrue(test_layer.supports_masking)
    query = tf.constant(
        [[1, 2, 3, 0, 0, 0], [3, 3, 1, 1, 2, 0], [1, 1, 0, 0, 0, 0]]
    )
    masked_query = tf_keras.layers.Embedding(4, 8, mask_zero=True)(query)
    value = tf.random.normal((3, 3, 8))
    output = test_layer(query=masked_query, value=value)
    self.assertTrue(hasattr(output, "_keras_mask"))
    self.assertAllEqual(masked_query._keras_mask, output._keras_mask)

  def test_value_mask(self):
    """Test that the value mask is taken into account."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=2,
        key_dim=2,
        src_block_size=2,
        tgt_block_size=1,
    )
    query = tf.constant(
        [[1, 2, 3, 0, 0, 0], [3, 3, 1, 1, 2, 0], [1, 1, 0, 0, 0, 0]]
    )
    masked_query = tf_keras.layers.Embedding(4, 8, mask_zero=True)(query)
    value = tf.constant([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
    masked_value = tf_keras.layers.Embedding(6, 8, mask_zero=True)(value)
    output = test_layer(
        query=masked_query,
        value=masked_value,
    )
    mask = tf.constant(
        [[[True, True, False]] * 3 + [[False, False, False]] * 2]
        + [[[True, False, False]] * 5]
        + [[[True, True, True]] + [[False, False, False]] * 4]
    )
    del masked_query._keras_mask
    del masked_value._keras_mask
    output_with_manual_mask = test_layer(
        query=masked_query, value=masked_value, attention_mask=mask
    )
    self.assertAllClose(output, output_with_manual_mask)

  def test_masks_are_cast_to_bool(self):
    """Test that the implicit and explicit masks are cast to bool."""
    test_layer = block_sparse_attention.MultiHeadAttention(
        num_heads=2, key_dim=2, src_block_size=2, tgt_block_size=1,
    )
    query = np.array(
        [[1, 2, 3, 0, 0, 0], [3, 3, 1, 1, 2, 0], [1, 1, 0, 0, 0, 0]]
    )
    masked_query = tf_keras.layers.Embedding(4, 8, mask_zero=True)(query)
    masked_query._keras_mask = tf.cast(masked_query._keras_mask, tf.float32)
    value = np.array([[5, 4, 0], [3, 0, 0], [2, 1, 1]])
    masked_value = tf_keras.layers.Embedding(6, 8, mask_zero=True)(value)
    masked_value._keras_mask = tf.cast(masked_value._keras_mask, tf.float32)
    float_mask = tf.constant([[[1.0]]])
    # if all works well, the following should not raise any exception:
    _ = test_layer(
        query=masked_query,
        value=masked_value,
        attention_mask=float_mask,
    )


if __name__ == "__main__":
  tf.test.main()
