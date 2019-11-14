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
"""Tests for transformer-based text encoder network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.networks import transformer_encoder


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class TransformerEncoderTest(keras_parameterized.TestCase):

  def test_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small TransformerEncoder for testing.
    test_network = transformer_encoder.TransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_attention_heads=2,
        num_layers=3)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    data, pooled = test_network([word_ids, mask, type_ids])

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  def test_network_creation_with_float16_dtype(self):
    hidden_size = 32
    sequence_length = 21
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    # Create a small TransformerEncoder for testing.
    test_network = transformer_encoder.TransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_attention_heads=2,
        num_layers=3,
        float_dtype="float16")
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    data, pooled = test_network([word_ids, mask, type_ids])

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # If float_dtype is set to float16, the output should always be float16.
    self.assertAllEqual(tf.float16, data.dtype)
    self.assertAllEqual(tf.float16, pooled.dtype)

  def test_network_invocation(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    tf.keras.mixed_precision.experimental.set_policy("float32")
    # Create a small TransformerEncoder for testing.
    test_network = transformer_encoder.TransformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types)
    self.assertTrue(
        test_network._position_embedding_layer._use_dynamic_slicing)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    data, pooled = test_network([word_ids, mask, type_ids])

    # Create a model based off of this network:
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    _ = model.predict([word_id_data, mask_data, type_id_data])

    # Creates a TransformerEncoder with max_sequence_length != sequence_length
    max_sequence_length = 128
    test_network = transformer_encoder.TransformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types)
    self.assertTrue(test_network._position_embedding_layer._use_dynamic_slicing)
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    _ = model.predict([word_id_data, mask_data, type_id_data])

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        vocab_size=100,
        hidden_size=32,
        num_layers=3,
        num_attention_heads=2,
        sequence_length=21,
        max_sequence_length=21,
        type_vocab_size=12,
        intermediate_size=1223,
        activation="relu",
        dropout_rate=0.05,
        attention_dropout_rate=0.22,
        initializer="glorot_uniform",
        float_dtype="float16")
    network = transformer_encoder.TransformerEncoder(**kwargs)

    expected_config = dict(kwargs)
    expected_config["activation"] = tf.keras.activations.serialize(
        tf.keras.activations.get(expected_config["activation"]))
    expected_config["initializer"] = tf.keras.initializers.serialize(
        tf.keras.initializers.get(expected_config["initializer"]))
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = transformer_encoder.TransformerEncoder.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == "__main__":
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
