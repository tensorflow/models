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
"""Tests for span_labeling network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.networks import span_labeling


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class SpanLabelingTest(keras_parameterized.TestCase):

  def test_network_creation(self):
    """Validate that the Keras object can be created."""
    sequence_length = 15
    input_width = 512
    test_network = span_labeling.SpanLabeling(
        input_width=input_width, output='predictions')
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_data = tf.keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    start_outputs, end_outputs = test_network(sequence_data)

    # Validate that the outputs are of the expected shape.
    expected_output_shape = [None, sequence_length]
    self.assertEqual(expected_output_shape, start_outputs.shape.as_list())
    self.assertEqual(expected_output_shape, end_outputs.shape.as_list())

  def test_network_invocation(self):
    """Validate that the Keras object can be invoked."""
    sequence_length = 15
    input_width = 512
    test_network = span_labeling.SpanLabeling(input_width=input_width)

    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_data = tf.keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    outputs = test_network(sequence_data)
    model = tf.keras.Model(sequence_data, outputs)

    # Invoke the network as part of a Model.
    batch_size = 3
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, input_width))
    start_outputs, end_outputs = model.predict(input_data)

    # Validate that the outputs are of the expected shape.
    expected_output_shape = (batch_size, sequence_length)
    self.assertEqual(expected_output_shape, start_outputs.shape)
    self.assertEqual(expected_output_shape, end_outputs.shape)

  def test_network_invocation_with_internal_logit_output(self):
    """Validate that the logit outputs are correct."""
    sequence_length = 15
    input_width = 512
    test_network = span_labeling.SpanLabeling(
        input_width=input_width, output='predictions')
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_data = tf.keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    output = test_network(sequence_data)
    model = tf.keras.Model(sequence_data, output)
    logit_model = tf.keras.Model(
        test_network.inputs,
        [test_network.start_logits, test_network.end_logits])

    batch_size = 3
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, input_width))
    start_outputs, end_outputs = model.predict(input_data)
    start_logits, end_logits = logit_model.predict(input_data)

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, sequence_length)
    self.assertEqual(expected_output_shape, start_outputs.shape)
    self.assertEqual(expected_output_shape, end_outputs.shape)
    self.assertEqual(expected_output_shape, start_logits.shape)
    self.assertEqual(expected_output_shape, end_logits.shape)

    # Ensure that the logits, when softmaxed, create the outputs.
    input_tensor = tf.keras.Input(expected_output_shape[1:])
    output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf.keras.Model(input_tensor, output_tensor)

    start_softmax = softmax_model.predict(start_logits)
    self.assertAllClose(start_outputs, start_softmax)
    end_softmax = softmax_model.predict(end_logits)
    self.assertAllClose(end_outputs, end_softmax)

  def test_network_invocation_with_external_logit_output(self):
    """Validate that the logit outputs are correct."""
    sequence_length = 15
    input_width = 512
    test_network = span_labeling.SpanLabeling(
        input_width=input_width, output='predictions')
    logit_network = span_labeling.SpanLabeling(
        input_width=input_width, output='logits')
    logit_network.set_weights(test_network.get_weights())

    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_data = tf.keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    output = test_network(sequence_data)
    logit_output = logit_network(sequence_data)
    model = tf.keras.Model(sequence_data, output)
    logit_model = tf.keras.Model(sequence_data, logit_output)

    batch_size = 3
    input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, input_width))
    start_outputs, end_outputs = model.predict(input_data)
    start_logits, end_logits = logit_model.predict(input_data)

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, sequence_length)
    self.assertEqual(expected_output_shape, start_outputs.shape)
    self.assertEqual(expected_output_shape, end_outputs.shape)
    self.assertEqual(expected_output_shape, start_logits.shape)
    self.assertEqual(expected_output_shape, end_logits.shape)

    # Ensure that the logits, when softmaxed, create the outputs.
    input_tensor = tf.keras.Input(expected_output_shape[1:])
    output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf.keras.Model(input_tensor, output_tensor)

    start_softmax = softmax_model.predict(start_logits)
    self.assertAllClose(start_outputs, start_softmax)
    end_softmax = softmax_model.predict(end_logits)
    self.assertAllClose(end_outputs, end_softmax)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    network = span_labeling.SpanLabeling(
        input_width=128,
        activation='relu',
        initializer='zeros',
        output='predictions')

    # Create another network object from the first object's config.
    new_network = span_labeling.SpanLabeling.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  def test_unknown_output_type_fails(self):
    with self.assertRaisesRegex(ValueError, 'Unknown `output` value "bad".*'):
      _ = span_labeling.SpanLabeling(input_width=10, output='bad')


if __name__ == '__main__':
  tf.test.main()
