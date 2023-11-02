# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf, tf_keras

from official.nlp.modeling.networks import bert_encoder
from official.projects.token_dropping import encoder


class TokenDropBertEncoderTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(TokenDropBertEncoderTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy("float32")

  def test_dict_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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

  def test_dict_outputs_all_encoder_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True,
        token_keep_k=sequence_length,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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

  def test_dict_outputs_network_creation_with_float16_dtype(self):
    hidden_size = 32
    sequence_length = 21
    tf_keras.mixed_precision.set_global_policy("mixed_float16")
    # Create a small BertEncoder for testing.
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=4,
        dict_outputs=True,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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
      ("all_sequence_encoder", None, 21),
      ("output_range_encoder", 1, 1),
  )
  def test_dict_outputs_network_invocation(
      self, output_range, out_seq_len):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    # Create a small BertEncoder for testing.
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        dict_outputs=True,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids),
        output_range=output_range)
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
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        dict_outputs=True,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16,
        dict_outputs=True,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))

  def test_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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

    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    inputs = dict(
        input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids)
    _ = test_network(inputs)

  def test_all_encoder_outputs_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        return_all_encoder_outputs=True,
        token_keep_k=sequence_length,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=4,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids),
        output_range=output_range)
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
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    test_network = encoder.TokenDropBertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16,
        token_keep_k=2,
        token_allow_list=(),
        token_deny_list=())
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))


class TokenDropCompatibilityTest(tf.test.TestCase):

  def tearDown(self):
    super().tearDown()
    tf_keras.mixed_precision.set_global_policy("float32")

  def test_checkpoint_forward_compatible(self):
    batch_size = 3

    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7

    kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        output_range=None)

    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    data = dict(
        input_word_ids=word_id_data,
        input_mask=mask_data,
        input_type_ids=type_id_data)

    old_net = bert_encoder.BertEncoderV2(**kwargs)
    old_net_outputs = old_net(data)
    ckpt = tf.train.Checkpoint(net=old_net)
    path = ckpt.save(self.get_temp_dir())
    new_net = encoder.TokenDropBertEncoder(
        token_keep_k=sequence_length,
        token_allow_list=(),
        token_deny_list=(),
        **kwargs)
    new_ckpt = tf.train.Checkpoint(net=new_net)
    status = new_ckpt.restore(path)
    status.assert_existing_objects_matched()
    # assert_consumed will fail because the old model has redundant nodes.
    new_net_outputs = new_net(data)

    self.assertAllEqual(old_net_outputs.keys(), new_net_outputs.keys())
    for key in old_net_outputs:
      self.assertAllClose(old_net_outputs[key], new_net_outputs[key])

  def test_keras_model_checkpoint_forward_compatible(self):
    batch_size = 3

    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7

    kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        output_range=None)

    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    data = dict(
        input_word_ids=word_id_data,
        input_mask=mask_data,
        input_type_ids=type_id_data)

    old_net = bert_encoder.BertEncoderV2(**kwargs)
    inputs = old_net.inputs
    outputs = old_net(inputs)
    old_model = tf_keras.Model(inputs=inputs, outputs=outputs)
    old_model_outputs = old_model(data)
    ckpt = tf.train.Checkpoint(net=old_model)
    path = ckpt.save(self.get_temp_dir())
    new_net = encoder.TokenDropBertEncoder(
        token_keep_k=sequence_length,
        token_allow_list=(),
        token_deny_list=(),
        **kwargs)
    inputs = new_net.inputs
    outputs = new_net(inputs)
    new_model = tf_keras.Model(inputs=inputs, outputs=outputs)
    new_ckpt = tf.train.Checkpoint(net=new_model)
    new_ckpt.restore(path)

    new_model_outputs = new_model(data)

    self.assertAllEqual(old_model_outputs.keys(), new_model_outputs.keys())
    for key in old_model_outputs:
      self.assertAllClose(old_model_outputs[key], new_model_outputs[key])


if __name__ == "__main__":
  tf.test.main()
