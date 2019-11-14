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
"""Tests for masked language model network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import

from official.nlp.modeling.networks import masked_lm
from official.nlp.modeling.networks import transformer_encoder


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class MaskedLMTest(keras_parameterized.TestCase):

  def create_network(self,
                     vocab_size,
                     sequence_length,
                     hidden_size,
                     num_predictions,
                     output='predictions',
                     xformer_stack=None):
    # First, create a transformer stack that we can use to get the LM's
    # vocabulary weight.
    if xformer_stack is None:
      xformer_stack = transformer_encoder.TransformerEncoder(
          vocab_size=vocab_size,
          num_layers=1,
          sequence_length=sequence_length,
          hidden_size=hidden_size,
          num_attention_heads=4,
      )
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    lm_outputs, _ = xformer_stack([word_ids, mask, type_ids])

    # Create a maskedLM from the transformer stack.
    test_network = masked_lm.MaskedLM(
        num_predictions=num_predictions,
        input_width=lm_outputs.shape[-1],
        source_network=xformer_stack,
        output=output)
    return test_network

  def test_network_creation(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    num_predictions = 21
    test_network = self.create_network(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions)

    # Make sure that the output tensor of the masked LM is the right shape.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_lm_positions = tf.keras.Input(
        shape=(num_predictions,), dtype=tf.int32)
    output = test_network([lm_input_tensor, masked_lm_positions])

    expected_output_shape = [None, num_predictions, vocab_size]
    self.assertEqual(expected_output_shape, output.shape.as_list())

  def test_network_invocation_with_internal_logits(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    num_predictions = 21
    test_network = self.create_network(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions)

    # Create a model from the masked LM layer.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_lm_positions = tf.keras.Input(
        shape=(num_predictions,), dtype=tf.int32)
    output = test_network([lm_input_tensor, masked_lm_positions])
    model = tf.keras.Model([lm_input_tensor, masked_lm_positions], output)
    logits_model = tf.keras.Model(test_network.inputs, test_network.logits)

    # Invoke the masked LM on some fake data to make sure there are no runtime
    # errors in the code.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        2, size=(batch_size, num_predictions))
    outputs = model.predict([lm_input_data, masked_position_data])
    logits = logits_model.predict([lm_input_data, masked_position_data])

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, num_predictions, vocab_size)
    self.assertEqual(expected_output_shape, outputs.shape)
    self.assertEqual(expected_output_shape, logits.shape)

    # Ensure that the logits, when softmaxed, create the outputs.
    input_tensor = tf.keras.Input(expected_output_shape[1:])
    output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf.keras.Model(input_tensor, output_tensor)

    calculated_softmax = softmax_model.predict(logits)
    self.assertAllClose(outputs, calculated_softmax)

  def test_network_invocation_with_external_logits(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    num_predictions = 21
    xformer_stack = transformer_encoder.TransformerEncoder(
        vocab_size=vocab_size,
        num_layers=1,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_attention_heads=4,
    )
    test_network = self.create_network(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions,
        xformer_stack=xformer_stack,
        output='predictions')
    logit_network = self.create_network(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions,
        xformer_stack=xformer_stack,
        output='logits')
    logit_network.set_weights(test_network.get_weights())

    # Create a model from the masked LM layer.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_lm_positions = tf.keras.Input(
        shape=(num_predictions,), dtype=tf.int32)
    output = test_network([lm_input_tensor, masked_lm_positions])
    logit_output = logit_network([lm_input_tensor, masked_lm_positions])

    model = tf.keras.Model([lm_input_tensor, masked_lm_positions], output)
    logits_model = tf.keras.Model(([lm_input_tensor, masked_lm_positions]),
                                  logit_output)

    # Invoke the masked LM on some fake data to make sure there are no runtime
    # errors in the code.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        2, size=(batch_size, num_predictions))
    outputs = model.predict([lm_input_data, masked_position_data])
    logits = logits_model.predict([lm_input_data, masked_position_data])

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, num_predictions, vocab_size)
    self.assertEqual(expected_output_shape, outputs.shape)
    self.assertEqual(expected_output_shape, logits.shape)

    # Ensure that the logits, when softmaxed, create the outputs.
    input_tensor = tf.keras.Input(expected_output_shape[1:])
    output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf.keras.Model(input_tensor, output_tensor)

    calculated_softmax = softmax_model.predict(logits)
    self.assertAllClose(outputs, calculated_softmax)

  def test_network_invocation(self):
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    num_predictions = 21
    test_network = self.create_network(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions)

    # Create a model from the masked LM layer.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_lm_positions = tf.keras.Input(
        shape=(num_predictions,), dtype=tf.int32)
    output = test_network([lm_input_tensor, masked_lm_positions])
    model = tf.keras.Model([lm_input_tensor, masked_lm_positions], output)

    # Invoke the masked LM on some fake data to make sure there are no runtime
    # errors in the code.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        2, size=(batch_size, num_predictions))
    _ = model.predict([lm_input_data, masked_position_data])

  def test_unknown_output_type_fails(self):
    with self.assertRaisesRegex(ValueError, 'Unknown `output` value "bad".*'):
      _ = self.create_network(
          vocab_size=8,
          sequence_length=8,
          hidden_size=8,
          num_predictions=8,
          output='bad')


if __name__ == '__main__':
  tf.test.main()
