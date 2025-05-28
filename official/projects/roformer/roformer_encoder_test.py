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

"""Tests for transformer-based bert encoder network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.roformer import roformer_encoder


class RoformerEncoderTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(RoformerEncoderTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy("float32")

  def test_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = roformer_encoder.RoformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, 3)
    self.assertIsInstance(test_network.pooler_layer, tf_keras.layers.Dense)

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  def test_all_encoder_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = roformer_encoder.RoformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    all_encoder_outputs = dict_outputs["encoder_outputs"]
    pooled = dict_outputs["pooled_output"]

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertLen(all_encoder_outputs, 3)
    for data in all_encoder_outputs:
      self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  def test_network_creation_with_float16_dtype(self):
    hidden_size = 32
    sequence_length = 21
    tf_keras.mixed_precision.set_global_policy("mixed_float16")
    # Create a small BertEncoder for testing.
    test_network = roformer_encoder.RoformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # If float_dtype is set to float16, the data output is float32 (from a layer
    # norm) and pool output should be float16.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float16, pooled.dtype)

  @parameterized.named_parameters(
      ("all_sequence", None, 21),
      ("output_range", 1, 1),
  )
  def test_network_invocation(self, output_range, out_seq_len):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    # Create a small BertEncoder for testing.
    test_network = roformer_encoder.RoformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        output_range=output_range)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    # Create a model based off of this network:
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], out_seq_len)

    # Creates a BertEncoder with max_sequence_length != sequence_length
    max_sequence_length = 128
    test_network = roformer_encoder.RoformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    test_network = roformer_encoder.RoformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        vocab_size=100,
        hidden_size=32,
        num_layers=3,
        num_attention_heads=2,
        max_sequence_length=21,
        type_vocab_size=12,
        inner_dim=512,
        inner_activation="relu",
        output_dropout=0.05,
        attention_dropout=0.22,
        initializer="glorot_uniform",
        output_range=-1,
        embedding_width=16,
        embedding_layer=None,
        norm_first=False)
    network = roformer_encoder.RoformerEncoder(**kwargs)
    expected_config = dict(kwargs)
    expected_config["inner_activation"] = tf_keras.activations.serialize(
        tf_keras.activations.get(expected_config["inner_activation"]))
    expected_config["initializer"] = tf_keras.initializers.serialize(
        tf_keras.initializers.get(expected_config["initializer"]))
    self.assertEqual(network.get_config(), expected_config)
    # Create another network object from the first object's config.
    new_network = roformer_encoder.RoformerEncoder.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

    # Tests model saving/loading.
    model_path = self.get_temp_dir() + "/model"
    network.save(model_path)
    _ = tf_keras.models.load_model(model_path)


if __name__ == "__main__":
  tf.test.main()
