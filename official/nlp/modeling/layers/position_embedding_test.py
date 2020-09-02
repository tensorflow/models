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


if __name__ == "__main__":
  tf.test.main()
