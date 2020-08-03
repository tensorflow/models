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
"""Tests for ALBERT transformer-based text encoder network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.networks import albert_transformer_encoder


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class AlbertTransformerEncoderTest(keras_parameterized.TestCase):

  def tearDown(self):
    super(AlbertTransformerEncoderTest, self).tearDown()
    tf.keras.mixed_precision.experimental.set_policy("float32")

  @parameterized.named_parameters(
      dict(testcase_name="default", expected_dtype=tf.float32),
      dict(
          testcase_name="with_float16_dtype",
          expected_dtype=tf.float16),
  )
  def test_network_creation(self, expected_dtype):
    hidden_size = 32
    sequence_length = 21

    kwargs = dict(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3)
    if expected_dtype == tf.float16:
      tf.keras.mixed_precision.experimental.set_policy("mixed_float16")

    # Create a small TransformerEncoder for testing.
    test_network = albert_transformer_encoder.AlbertTransformerEncoder(**kwargs)

    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    data, pooled = test_network([word_ids, mask, type_ids])

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # If float_dtype is set to float16, the data output is float32 (from a layer
    # norm) and pool output should be float16.
    self.assertEqual(tf.float32, data.dtype)
    self.assertEqual(expected_dtype, pooled.dtype)

    # ALBERT has additonal 'embedding_hidden_mapping_in' weights and
    # it shares transformer weights.
    self.assertNotEmpty(
        [x for x in test_network.weights if "embedding_projection/" in x.name])
    self.assertNotEmpty(
        [x for x in test_network.weights if "transformer/" in x.name])
    self.assertEmpty(
        [x for x in test_network.weights if "transformer/layer" in x.name])

  def test_network_invocation(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    # Create a small TransformerEncoder for testing.
    test_network = albert_transformer_encoder.AlbertTransformerEncoder(
        vocab_size=vocab_size,
        embedding_width=8,
        hidden_size=hidden_size,
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
    test_network = albert_transformer_encoder.AlbertTransformerEncoder(
        vocab_size=vocab_size,
        embedding_width=8,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types)
    self.assertTrue(test_network._position_embedding_layer._use_dynamic_slicing)
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    _ = model.predict([word_id_data, mask_data, type_id_data])

  def test_serialize_deserialize(self):
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    # Create a network object that sets all of its config options.
    kwargs = dict(
        vocab_size=100,
        embedding_width=8,
        hidden_size=32,
        num_layers=3,
        num_attention_heads=2,
        max_sequence_length=21,
        type_vocab_size=12,
        intermediate_size=1223,
        activation="relu",
        dropout_rate=0.05,
        attention_dropout_rate=0.22,
        initializer="glorot_uniform")
    network = albert_transformer_encoder.AlbertTransformerEncoder(**kwargs)

    expected_config = dict(kwargs)
    expected_config["activation"] = tf.keras.activations.serialize(
        tf.keras.activations.get(expected_config["activation"]))
    expected_config["initializer"] = tf.keras.initializers.serialize(
        tf.keras.initializers.get(expected_config["initializer"]))
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = (
        albert_transformer_encoder.AlbertTransformerEncoder.from_config(
            network.get_config()))

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == "__main__":
  tf.test.main()
