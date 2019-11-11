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
"""Tests for Keras-based einsum layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import dense_einsum


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class DenseEinsumLayer(keras_parameterized.TestCase):

  def test_3D_einsum_with_two_bound_dimensions(self):
    test_layer = dense_einsum.DenseEinsum(
        output_shape=(64,), num_summed_dimensions=2)
    # Create a 4-dimensional input (the first dimension is implicit).
    input_tensor = tf.keras.Input(shape=(None, 40, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(test_layer._einsum_string, "abcd,cde->abe")
    self.assertEqual(test_layer._kernel_shape, (40, 80, 64))

  def test_3D_einsum_with_one_bound_dimensions(self):
    test_layer = dense_einsum.DenseEinsum(
        output_shape=(64, 32), num_summed_dimensions=1)
    # Create a 3-dimensional input (the first dimension is implicit).
    input_tensor = tf.keras.Input(shape=(None, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(test_layer._einsum_string, "abc,cde->abde")
    self.assertEqual(test_layer._kernel_shape, (80, 64, 32))

  def test_2D_einsum_with_one_bound_dimensions(self):
    test_layer = dense_einsum.DenseEinsum(
        output_shape=(64,), num_summed_dimensions=1)
    # Create a 3-dimensional input (the first dimension is implicit).
    input_tensor = tf.keras.Input(shape=(None, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(test_layer._einsum_string, "abc,cd->abd")
    self.assertEqual(test_layer._kernel_shape, (80, 64))

  def test_bias_term_can_be_disabled(self):
    # A layer created using the bias should have two weights.
    test_layer = dense_einsum.DenseEinsum(
        output_shape=64, num_summed_dimensions=1, use_bias=True)
    input_tensor = tf.keras.Input(shape=(None, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(2, len(test_layer.get_weights()))

    # A layer created without the bias should have only one weight.
    test_layer = dense_einsum.DenseEinsum(
        output_shape=64, num_summed_dimensions=1, use_bias=False)
    input_tensor = tf.keras.Input(shape=(None, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(1, len(test_layer.get_weights()))

  def test_activation(self):
    # Create a model that does not use an activation.
    no_activation_layer = dense_einsum.DenseEinsum(
        output_shape=64, num_summed_dimensions=1, activation=None)
    input_tensor = tf.keras.Input(shape=(None, 80))
    output_tensor = no_activation_layer(input_tensor)
    no_activation_model = tf.keras.Model(input_tensor, output_tensor)

    # Create a model that uses a softmax activation.
    activation_layer = dense_einsum.DenseEinsum(
        output_shape=64, num_summed_dimensions=1, activation="softmax")
    input_tensor = tf.keras.Input(shape=(None, 80))
    output_tensor = activation_layer(input_tensor)
    activation_model = tf.keras.Model(input_tensor, output_tensor)

    # Make sure the models' weights are identical.
    activation_model.set_weights(no_activation_model.get_weights())

    # Predict using each model on the same input data. The output should be
    # different, since one is using a softmax - even though the models' weights
    # are the same.
    input_values = 10 * np.random.random_sample((10, 4, 80))
    non_activated_data = no_activation_model.predict(input_values)
    activated_data = activation_model.predict(input_values)
    self.assertNotAllClose(activated_data, non_activated_data)

  def test_non_iterable_output_shape(self):
    test_layer = dense_einsum.DenseEinsum(
        output_shape=64, num_summed_dimensions=1)
    # Create a 3-dimensional input (the first dimension is implicit).
    input_tensor = tf.keras.Input(shape=(None, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(test_layer._einsum_string, "abc,cd->abd")
    self.assertEqual(test_layer._kernel_shape, (80, 64))

  def test_with_explicit_initializer(self):
    test_layer = dense_einsum.DenseEinsum(
        output_shape=(64,),
        num_summed_dimensions=2,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 4-dimensional input (the first dimension is implicit).
    input_tensor = tf.keras.Input(shape=(None, 40, 80))
    _ = test_layer(input_tensor)
    self.assertEqual(test_layer._einsum_string, "abcd,cde->abe")
    self.assertEqual(test_layer._kernel_shape, (40, 80, 64))


if __name__ == "__main__":
  tf.test.main()
