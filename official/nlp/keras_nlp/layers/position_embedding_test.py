# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Keras-based positional embedding layer."""

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.keras_nlp.layers import position_embedding


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class PositionEmbeddingLayerTest(keras_parameterized.TestCase):

  def test_static_layer_output_shape(self):
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_length = 21
    test_layer = position_embedding.PositionEmbedding(
        max_length=sequence_length)
    width = 30
    input_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(input_tensor)

    # When using static positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions save batch.
    expected_output_shape = [None, sequence_length, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    # The default output dtype for this layer should be tf.float32.
    self.assertEqual(tf.float32, output_tensor.dtype)

  def test_non_default_axis_static(self):
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_length = 21
    test_layer = position_embedding.PositionEmbedding(
        max_length=sequence_length, seq_axis=2)
    width = 30
    input_tensor = tf.keras.Input(shape=(width, sequence_length, width))
    output_tensor = test_layer(input_tensor)

    # When using static positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions save batch.
    expected_output_shape = [None, width, sequence_length, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    # The default output dtype for this layer should be tf.float32.
    self.assertEqual(tf.float32, output_tensor.dtype)

  def test_float16_dtype(self):
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_length = 21
    test_layer = position_embedding.PositionEmbedding(
        max_length=sequence_length, dtype="float16")
    width = 30
    input_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(input_tensor)

    # When using static positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions save batch.
    expected_output_shape = [None, sequence_length, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    # The default output dtype for this layer should be tf.float32.
    self.assertEqual(tf.float16, output_tensor.dtype)

  def test_dynamic_layer_output_shape(self):
    max_sequence_length = 40
    test_layer = position_embedding.PositionEmbedding(
        max_length=max_sequence_length)
    # Create a 3-dimensional input (the first dimension is implicit).
    width = 30
    input_tensor = tf.keras.Input(shape=(None, width))
    output_tensor = test_layer(input_tensor)

    # When using dynamic positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions - but may be None if
    # the input shape is None there.
    expected_output_shape = [None, None, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())

  def test_non_default_axis_dynamic(self):
    max_sequence_length = 60
    test_layer = position_embedding.PositionEmbedding(
        max_length=max_sequence_length, seq_axis=2)
    # Create a 3-dimensional input (the first dimension is implicit).
    width = 30
    input_tensor = tf.keras.Input(shape=(None, None, width))
    output_tensor = test_layer(input_tensor)

    # When using dynamic positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions - but may be None if
    # the input shape is None there.
    expected_output_shape = [None, None, None, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())

  def test_dynamic_layer_slicing(self):
    max_sequence_length = 40
    test_layer = position_embedding.PositionEmbedding(
        max_length=max_sequence_length)
    # Create a 3-dimensional input (the first dimension is implicit).
    width = 30
    input_tensor = tf.keras.Input(shape=(None, width))
    output_tensor = test_layer(input_tensor)

    model = tf.keras.Model(input_tensor, output_tensor)

    # Create input data that is shorter than max_sequence_length, which should
    # trigger a down-slice.
    input_length = 17
    # Note: This test explicitly uses a batch size of 1. This is to get around
    # Keras' restriction on Model invocations: inputs are expected to have the
    # same batch cardinality as outputs. In practice, this layer should be used
    # inside a model, where it can be projected when added to another tensor.
    input_data = np.ones((1, input_length, width))
    output_data = model.predict(input_data)

    self.assertAllEqual([1, input_length, width], output_data.shape)


if __name__ == "__main__":
  tf.test.main()
