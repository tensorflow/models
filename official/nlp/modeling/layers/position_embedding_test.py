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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import position_embedding


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class RelativePositionEmbeddingLayerTest(keras_parameterized.TestCase):

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


@keras_parameterized.run_all_keras_modes
class RelativePositionBiasTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(("bidirectional", True),
                                  ("unidirectional", False))
  def test_relative_position_bias(self, bidirectional):
    query = tf.zeros((4, 4, 2))
    key = tf.zeros((4, 2, 2))
    l = position_embedding.RelativePositionBias(
        num_heads=3,
        bidirectional=bidirectional,
        name="foo")
    self.assertEqual(l(query, key).shape, (4, 3, 4, 2))
    self.assertLen(l.trainable_variables, 1)
    self.assertEqual(l.trainable_variables[0].name, "foo/rel_embedding:0")

  def test_relative_position_bucket(self):
    context_position = tf.range(3)[:, None]
    memory_position = tf.range(2)[None, :]
    relative_position = memory_position - context_position
    outputs = position_embedding._relative_position_bucket(relative_position)
    self.assertAllEqual(outputs.numpy(), np.array([[0, 17], [1, 0], [2, 1]]))
    outputs = position_embedding._relative_position_bucket(
        relative_position, bidirectional=False)
    self.assertAllEqual(outputs.numpy(), np.array([[0, 0], [1, 0], [2, 1]]))


if __name__ == "__main__":
  tf.test.main()
