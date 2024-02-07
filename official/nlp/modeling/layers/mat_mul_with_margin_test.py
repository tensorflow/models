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

"""Tests for mat_mul_with_margin layer."""

import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import mat_mul_with_margin


class MatMulWithMarginTest(tf.test.TestCase):

  def test_layer_invocation(self):
    """Validate that the Keras object can be created and invoked."""
    input_width = 512
    test_layer = mat_mul_with_margin.MatMulWithMargin()
    # Create a 2-dimensional input (the first dimension is implicit).
    left_encoded = tf_keras.Input(shape=(input_width,), dtype=tf.float32)
    right_encoded = tf_keras.Input(shape=(input_width,), dtype=tf.float32)
    left_logits, right_logits = test_layer(left_encoded, right_encoded)

    # Validate that the outputs are of the expected shape.
    expected_output_shape = [None, None]
    self.assertEqual(expected_output_shape, left_logits.shape.as_list())
    self.assertEqual(expected_output_shape, right_logits.shape.as_list())

  def test_serialize_deserialize(self):
    # Create a layer object that sets all of its config options.
    layer = mat_mul_with_margin.MatMulWithMargin()

    # Create another layer object from the first object's config.
    new_layer = mat_mul_with_margin.MatMulWithMargin.from_config(
        layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(layer.get_config(), new_layer.get_config())


if __name__ == '__main__':
  tf.test.main()
