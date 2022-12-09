# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for transformer-based bert encoder network with dense features as inputs."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.nlp.modeling.networks import bert_encoder


class BertEncoderV2Test(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(BertEncoderV2Test, self).tearDown()
    tf.keras.mixed_precision.set_global_policy("float32")

  def test_dict_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    dense_sequence_length = 20
    # Create a small dense BertEncoderV2 for testing.
    kwargs = {}
    test_network = bert_encoder.BertEncoderV2(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        with_dense_inputs=True,
        **kwargs)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    dense_inputs = tf.keras.Input(
        shape=(dense_sequence_length, hidden_size), dtype=tf.float32)
    dense_mask = tf.keras.Input(shape=(dense_sequence_length,), dtype=tf.int32)
    dense_type_ids = tf.keras.Input(
        shape=(dense_sequence_length,), dtype=tf.int32)

    dict_outputs = test_network(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, 3)
    self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

    expected_data_shape = [
        None, sequence_length + dense_sequence_length, hidden_size
    ]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  def test_dict_outputs_all_encoder_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    dense_sequence_length = 20
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoderV2(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True,
        with_dense_inputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    dense_inputs = tf.keras.Input(
        shape=(dense_sequence_length, hidden_size), dtype=tf.float32)
    dense_mask = tf.keras.Input(shape=(dense_sequence_length,), dtype=tf.int32)
    dense_type_ids = tf.keras.Input(
        shape=(dense_sequence_length,), dtype=tf.int32)

    dict_outputs = test_network(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))

    all_encoder_outputs = dict_outputs["encoder_outputs"]
    pooled = dict_outputs["pooled_output"]

    expected_data_shape = [
        None, sequence_length + dense_sequence_length, hidden_size
    ]
    expected_pooled_shape = [None, hidden_size]
    self.assertLen(all_encoder_outputs, 3)
    for data in all_encoder_outputs:
      self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  def test_dict_outputs_network_creation_with_float16_dtype(self):
    hidden_size = 32
    sequence_length = 21
    dense_sequence_length = 20
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoderV2(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True,
        with_dense_inputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    dense_inputs = tf.keras.Input(
        shape=(dense_sequence_length, hidden_size), dtype=tf.float32)
    dense_mask = tf.keras.Input(shape=(dense_sequence_length,), dtype=tf.int32)
    dense_type_ids = tf.keras.Input(
        shape=(dense_sequence_length,), dtype=tf.int32)

    dict_outputs = test_network(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))

    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    expected_data_shape = [
        None, sequence_length + dense_sequence_length, hidden_size
    ]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # If float_dtype is set to float16, the data output is float32 (from a layer
    # norm) and pool output should be float16.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float16, pooled.dtype)

  @parameterized.named_parameters(
      ("all_sequence_encoder_v2", bert_encoder.BertEncoderV2, None, 41),
      ("output_range_encoder_v2", bert_encoder.BertEncoderV2, 1, 1),
  )
  def test_dict_outputs_network_invocation(
      self, encoder_cls, output_range, out_seq_len):
    hidden_size = 32
    sequence_length = 21
    dense_sequence_length = 20
    vocab_size = 57
    num_types = 7
    # Create a small BertEncoder for testing.
    test_network = encoder_cls(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        dict_outputs=True,
        with_dense_inputs=True,
        output_range=output_range)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dense_inputs = tf.keras.Input(
        shape=(dense_sequence_length, hidden_size), dtype=tf.float32)
    dense_mask = tf.keras.Input(shape=(dense_sequence_length,), dtype=tf.int32)
    dense_type_ids = tf.keras.Input(
        shape=(dense_sequence_length,), dtype=tf.int32)

    dict_outputs = test_network(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    # Create a model based off of this network:
    model = tf.keras.Model(
        [word_ids, mask, type_ids, dense_inputs, dense_mask, dense_type_ids],
        [data, pooled])

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))

    dense_input_data = np.random.rand(batch_size, dense_sequence_length,
                                      hidden_size)
    dense_mask_data = np.random.randint(
        2, size=(batch_size, dense_sequence_length))
    dense_type_ids_data = np.random.randint(
        num_types, size=(batch_size, dense_sequence_length))

    outputs = model.predict([
        word_id_data, mask_data, type_id_data, dense_input_data,
        dense_mask_data, dense_type_ids_data
    ])
    self.assertEqual(outputs[0].shape[1], out_seq_len)

    # Creates a BertEncoder with max_sequence_length != sequence_length
    max_sequence_length = 128
    test_network = encoder_cls(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        dict_outputs=True)
    dict_outputs = test_network(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf.keras.Model(
        [word_ids, mask, type_ids, dense_inputs, dense_mask, dense_type_ids],
        [data, pooled])
    outputs = model.predict([
        word_id_data, mask_data, type_id_data, dense_input_data,
        dense_mask_data, dense_type_ids_data
    ])
    self.assertEqual(outputs[0].shape[1],
                     sequence_length + dense_sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    embedding_width = 16
    test_network = bert_encoder.BertEncoderV2(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=embedding_width,
        dict_outputs=True)

    dense_inputs = tf.keras.Input(
        shape=(dense_sequence_length, embedding_width), dtype=tf.float32)
    dense_input_data = np.zeros(
        (batch_size, dense_sequence_length, embedding_width), dtype=float)

    dict_outputs = test_network(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf.keras.Model(
        [word_ids, mask, type_ids, dense_inputs, dense_mask, dense_type_ids],
        [data, pooled])
    outputs = model.predict([
        word_id_data, mask_data, type_id_data, dense_input_data,
        dense_mask_data, dense_type_ids_data
    ])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))

  def test_embeddings_as_inputs(self):
    hidden_size = 32
    sequence_length = 21
    dense_sequence_length = 20
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoderV2(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        with_dense_inputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    dense_inputs = tf.keras.Input(
        shape=(dense_sequence_length, hidden_size), dtype=tf.float32)
    dense_mask = tf.keras.Input(shape=(dense_sequence_length,), dtype=tf.int32)
    dense_type_ids = tf.keras.Input(
        shape=(dense_sequence_length,), dtype=tf.int32)

    test_network.build(
        dict(
            input_word_ids=word_ids,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))
    embeddings = test_network.get_embedding_layer()(word_ids)
    # Calls with the embeddings.
    dict_outputs = test_network(
        dict(
            input_word_embeddings=embeddings,
            input_mask=mask,
            input_type_ids=type_ids,
            dense_inputs=dense_inputs,
            dense_mask=dense_mask,
            dense_type_ids=dense_type_ids))

    all_encoder_outputs = dict_outputs["encoder_outputs"]
    pooled = dict_outputs["pooled_output"]

    expected_data_shape = [
        None, sequence_length + dense_sequence_length, hidden_size
    ]
    expected_pooled_shape = [None, hidden_size]
    self.assertLen(all_encoder_outputs, 3)
    for data in all_encoder_outputs:
      self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)


if __name__ == "__main__":
  tf.test.main()
