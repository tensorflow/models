# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.networks import bert_encoder


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class BertEncoderTest(keras_parameterized.TestCase):

  def tearDown(self):
    super(BertEncoderTest, self).tearDown()
    tf.keras.mixed_precision.set_global_policy("float32")

  def test_v2_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, 3)
    self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  def test_v2_all_encoder_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
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

  def test_v2_network_creation_with_float16_dtype(self):
    hidden_size = 32
    sequence_length = 21
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
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
  def test_v2_network_invocation(self, output_range, out_seq_len):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        output_range=output_range,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

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
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], out_seq_len)

    # Creates a BertEncoder with max_sequence_length != sequence_length
    max_sequence_length = 128
    test_network = bert_encoder.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        dict_outputs=True)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    test_network = bert_encoder.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16,
        dict_outputs=True)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
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
        inner_dim=1223,
        inner_activation="relu",
        output_dropout=0.05,
        attention_dropout=0.22,
        initializer="glorot_uniform",
        output_range=-1,
        embedding_width=16,
        embedding_layer=None,
        norm_first=False)
    network = bert_encoder.BertEncoder(**kwargs)

    # Validate that the config can be forced to JSON.
    _ = network.to_json()

    # Tests model saving/loading.
    model_path = self.get_temp_dir() + "/model"
    network.save(model_path)
    _ = tf.keras.models.load_model(model_path)

  def test_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    data, pooled = test_network([word_ids, mask, type_ids])

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, 3)
    self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

    test_network_dict = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    inputs = dict(
        input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids)
    _ = test_network_dict(inputs)

    test_network_dict.set_weights(test_network.get_weights())
    batch_size = 2
    vocab_size = 100
    num_types = 2
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    list_outputs = test_network([word_id_data, mask_data, type_id_data])
    dict_outputs = test_network_dict(
        dict(
            input_word_ids=word_id_data,
            input_mask=mask_data,
            input_type_ids=type_id_data))
    self.assertAllEqual(list_outputs[0], dict_outputs["sequence_output"])
    self.assertAllEqual(list_outputs[1], dict_outputs["pooled_output"])

  def test_all_encoder_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        return_all_encoder_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    all_encoder_outputs, pooled = test_network([word_ids, mask, type_ids])

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
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
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
    test_network = bert_encoder.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        output_range=output_range)
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
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], out_seq_len)

    # Creates a BertEncoder with max_sequence_length != sequence_length
    max_sequence_length = 128
    test_network = bert_encoder.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types)
    data, pooled = test_network([word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    test_network = bert_encoder.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16)
    data, pooled = test_network([word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))


if __name__ == "__main__":
  tf.test.main()
