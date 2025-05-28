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

"""Tests for span_labeling network."""
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.networks import span_labeling


class SpanLabelingTest(tf.test.TestCase):

  def test_network_creation(self):
    """Validate that the Keras object can be created."""
    sequence_length = 15
    input_width = 512
    test_network = span_labeling.SpanLabeling(
        input_width=input_width, output='predictions')
    # Create a 3-dimensional input (the first dimension is implicit).
    sequence_data = tf_keras.Input(
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
    sequence_data = tf_keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    outputs = test_network(sequence_data)
    model = tf_keras.Model(sequence_data, outputs)

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
    sequence_data = tf_keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    output = test_network(sequence_data)
    model = tf_keras.Model(sequence_data, output)
    logit_model = tf_keras.Model(
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
    input_tensor = tf_keras.Input(expected_output_shape[1:])
    output_tensor = tf_keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf_keras.Model(input_tensor, output_tensor)

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
    sequence_data = tf_keras.Input(
        shape=(sequence_length, input_width), dtype=tf.float32)
    output = test_network(sequence_data)
    logit_output = logit_network(sequence_data)
    model = tf_keras.Model(sequence_data, output)
    logit_model = tf_keras.Model(sequence_data, logit_output)

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
    input_tensor = tf_keras.Input(expected_output_shape[1:])
    output_tensor = tf_keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf_keras.Model(input_tensor, output_tensor)

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


class XLNetSpanLabelingTest(tf.test.TestCase):

  def test_basic_invocation_train(self):
    batch_size = 2
    seq_length = 8
    hidden_size = 4
    sequence_data = np.random.uniform(
        size=(batch_size, seq_length, hidden_size)).astype('float32')
    paragraph_mask = np.random.uniform(
        size=(batch_size, seq_length)).astype('float32')
    class_index = np.random.uniform(size=(batch_size)).astype('uint8')
    start_positions = np.zeros(shape=(batch_size)).astype('uint8')

    layer = span_labeling.XLNetSpanLabeling(
        input_width=hidden_size,
        start_n_top=2,
        end_n_top=2,
        activation='tanh',
        dropout_rate=0.,
        initializer='glorot_uniform')
    output = layer(sequence_data=sequence_data,
                   class_index=class_index,
                   paragraph_mask=paragraph_mask,
                   start_positions=start_positions,
                   training=True)

    expected_keys = {
        'start_logits', 'end_logits', 'class_logits', 'start_predictions',
        'end_predictions',
    }
    self.assertSetEqual(expected_keys, set(output.keys()))

  def test_basic_invocation_beam_search(self):
    batch_size = 2
    seq_length = 8
    hidden_size = 4
    top_n = 5
    sequence_data = np.random.uniform(
        size=(batch_size, seq_length, hidden_size)).astype('float32')
    paragraph_mask = np.random.uniform(
        size=(batch_size, seq_length)).astype('float32')
    class_index = np.random.uniform(size=(batch_size)).astype('uint8')

    layer = span_labeling.XLNetSpanLabeling(
        input_width=hidden_size,
        start_n_top=top_n,
        end_n_top=top_n,
        activation='tanh',
        dropout_rate=0.,
        initializer='glorot_uniform')
    output = layer(sequence_data=sequence_data,
                   class_index=class_index,
                   paragraph_mask=paragraph_mask,
                   training=False)
    expected_keys = {
        'start_top_predictions', 'end_top_predictions', 'class_logits',
        'start_top_index', 'end_top_index', 'start_logits',
        'end_logits', 'start_predictions', 'end_predictions'
    }
    self.assertSetEqual(expected_keys, set(output.keys()))

  def test_subclass_invocation(self):
    """Tests basic invocation of this layer wrapped in a subclass."""
    seq_length = 8
    hidden_size = 4
    batch_size = 2

    sequence_data = tf_keras.Input(shape=(seq_length, hidden_size),
                                   dtype=tf.float32)
    class_index = tf_keras.Input(shape=(), dtype=tf.uint8)
    paragraph_mask = tf_keras.Input(shape=(seq_length), dtype=tf.float32)
    start_positions = tf_keras.Input(shape=(), dtype=tf.int32)

    layer = span_labeling.XLNetSpanLabeling(
        input_width=hidden_size,
        start_n_top=5,
        end_n_top=5,
        activation='tanh',
        dropout_rate=0.,
        initializer='glorot_uniform')

    output = layer(sequence_data=sequence_data,
                   class_index=class_index,
                   paragraph_mask=paragraph_mask,
                   start_positions=start_positions)
    model = tf_keras.Model(
        inputs={
            'sequence_data': sequence_data,
            'class_index': class_index,
            'paragraph_mask': paragraph_mask,
            'start_positions': start_positions,
        },
        outputs=output)

    sequence_data = tf.random.uniform(
        shape=(batch_size, seq_length, hidden_size), dtype=tf.float32)
    paragraph_mask = tf.random.uniform(
        shape=(batch_size, seq_length), dtype=tf.float32)
    class_index = tf.ones(shape=(batch_size,), dtype=tf.uint8)
    start_positions = tf.random.uniform(
        shape=(batch_size,), maxval=5, dtype=tf.int32)

    inputs = dict(sequence_data=sequence_data,
                  paragraph_mask=paragraph_mask,
                  class_index=class_index,
                  start_positions=start_positions)

    output = model(inputs)
    self.assertIsInstance(output, dict)

    # Test `call` without training flag.
    output = model(inputs, training=False)
    self.assertIsInstance(output, dict)

    # Test `call` with training flag.
    # Note: this fails due to incompatibility with the functional API.
    with self.assertRaisesRegex(AssertionError,
                                'Could not compute output KerasTensor'):
      model(inputs, training=True)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    network = span_labeling.XLNetSpanLabeling(
        input_width=128,
        start_n_top=5,
        end_n_top=1,
        activation='tanh',
        dropout_rate=0.34,
        initializer='zeros')

    # Create another network object from the first object's config.
    new_network = span_labeling.XLNetSpanLabeling.from_config(
        network.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
