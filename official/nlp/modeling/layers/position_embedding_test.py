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
"""Tests for Keras-based positional embedding layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import position_embedding


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class PositionEmbeddingLayerTest(keras_parameterized.TestCase):

  def test_static_layer_output_shape(self):
    test_layer = position_embedding.PositionEmbedding()
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_length = 21
    width = 30
    input_tensor = tf.keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(input_tensor)

    # When using static positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions save batch.
    expected_output_shape = [None, sequence_length, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
    # The default output dtype for this layer should be tf.float32.
    self.assertEqual(tf.float32, output_tensor.dtype)

  def test_float16_dtype(self):
    test_layer = position_embedding.PositionEmbedding(dtype="float16")
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_length = 21
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
        use_dynamic_slicing=True, max_sequence_length=max_sequence_length)
    # Create a 3-dimensional input (the first dimension is implicit).
    width = 30
    input_tensor = tf.keras.Input(shape=(None, width))
    output_tensor = test_layer(input_tensor)

    # When using dynamic positional embedding shapes, the output is expected
    # to be the same as the input shape in all dimensions - but may be None if
    # the input shape is None there.
    expected_output_shape = [None, None, width]
    self.assertEqual(expected_output_shape, output_tensor.shape.as_list())

  def test_dynamic_layer_slicing(self):
    max_sequence_length = 40
    test_layer = position_embedding.PositionEmbedding(
        use_dynamic_slicing=True, max_sequence_length=max_sequence_length)
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

  def test_relative_tensor_input(self):
    hidden_size = 8
    test_layer = position_embedding.RelativePositionEmbedding(
        hidden_size=hidden_size)

    # create a 3-dimensional input for test_layer to infer length as 1.
    input_tensor = tf.constant([[[0] * hidden_size]])
    output_tensor = test_layer(input_tensor)

    # expected output is the theoretical result of the input based on
    # sine cosine relative position embedding formula.
    expected_output_tensor = tf.constant([[0, 0, 0, 0, 1, 1, 1, 1]])
    self.assertAllEqual(output_tensor, expected_output_tensor)

  def test_relative_length_input(self):
    hidden_size = 8

    # When we do not have tensor as input, we explicitly specify length
    # value when initializing test_layer.
    test_layer = position_embedding.RelativePositionEmbedding(
        hidden_size=hidden_size)
    input_tensor = None
    output_tensor = test_layer(input_tensor, length=1)

    # expected output is the theoretical result of the input based on
    # sine cosine relative position embedding formula.
    expected_output_tensor = tf.constant([[0, 0, 0, 0, 1, 1, 1, 1]])
    self.assertAllEqual(output_tensor, expected_output_tensor)


if __name__ == "__main__":
  tf.test.main()
