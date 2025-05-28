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

"""Tests for Keras-based gated feedforward layer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import gated_feedforward


class GatedFeedforwardTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(GatedFeedforwardTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy("float32")

  @parameterized.parameters(
      (True, 1, "after_residual", "float32"),
      (True, 1, "after_residual", "mixed_float16"),
      (False, 4, "before_residual", "float32"),
      (False, 4, "before_residual", "mixed_float16"),
      (True, 4, "after_residual", "float32"),
      (True, 4, "after_residual", "mixed_float16"),
      (False, 1, "before_residual", "float32"),
      (False, 1, "before_residual", "mixed_float16"),
  )
  def test_layer_creation(self, use_gate, num_blocks, dropout_position, dtype):
    tf_keras.mixed_precision.set_global_policy(dtype)
    kwargs = dict(
        inner_dim=128,
        inner_activation="relu",
        dropout=0.1,
        use_gate=use_gate,
        num_blocks=num_blocks,
        dropout_position=dropout_position,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros")
    test_layer = gated_feedforward.GatedFeedforward(**kwargs)

    sequence_length = 64
    width = 128
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)
    # The default output of a transformer layer should be the same as the input.
    self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

  @parameterized.parameters(
      (True, 1, "after_residual", "float32"),
      (True, 1, "after_residual", "mixed_float16"),
      (False, 4, "before_residual", "float32"),
      (False, 4, "before_residual", "mixed_float16"),
      (True, 4, "after_residual", "float32"),
      (True, 4, "after_residual", "mixed_float16"),
      (False, 1, "before_residual", "float32"),
      (False, 1, "before_residual", "mixed_float16"),
  )
  def test_layer_invocation(self, use_gate, num_blocks, dropout_position,
                            dtype):
    tf_keras.mixed_precision.set_global_policy(dtype)
    kwargs = dict(
        inner_dim=16,
        inner_activation="relu",
        dropout=0.1,
        use_gate=use_gate,
        num_blocks=num_blocks,
        dropout_position=dropout_position,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros")
    test_layer = gated_feedforward.GatedFeedforward(**kwargs)

    sequence_length = 16
    width = 32
    # Create a 3-dimensional input (the first dimension is implicit).
    data_tensor = tf_keras.Input(shape=(sequence_length, width))
    output_tensor = test_layer(data_tensor)

    # Create a model from the test layer.
    model = tf_keras.Model(data_tensor, output_tensor)

    # Invoke the model on test data.
    batch_size = 6
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, width))
    output_data = model.predict(input_data)
    self.assertEqual(output_data.shape, (batch_size, sequence_length, width))

  def test_serialize_deserialize(self):
    kwargs = dict(
        inner_dim=16,
        inner_activation="relu",
        dropout=0.1,
        use_gate=False,
        num_blocks=4,
        dropout_position="after_residual",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros")
    test_layer = gated_feedforward.GatedFeedforward(**kwargs)
    new_layer = gated_feedforward.GatedFeedforward.from_config(
        test_layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_layer.get_config(), new_layer.get_config())


if __name__ == "__main__":
  tf.test.main()
