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
import tensorflow as tf

from official.nlp.modeling.networks import bert_encoder


class BertEncoderTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(BertEncoderTest, self).tearDown()
    tf.keras.mixed_precision.set_global_policy("float32")

  @parameterized.named_parameters(
      ("encoder_v2", bert_encoder.BertEncoderV2),
      ("encoder_v1", bert_encoder.BertEncoder),
  )
  def test_dict_outputs_network_creation(self, encoder_cls):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    if encoder_cls is bert_encoder.BertEncoderV2:
      kwargs = {}
    else:
      kwargs = dict(dict_outputs=True)
    test_network = encoder_cls(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        **kwargs)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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

  @parameterized.named_parameters(
      ("encoder_v2", bert_encoder.BertEncoderV2),
      ("encoder_v1", bert_encoder.BertEncoder),
  )
  def test_dict_outputs_all_encoder_outputs_network_creation(self, encoder_cls):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = encoder_cls(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
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

  @parameterized.named_parameters(
      ("encoder_v2", bert_encoder.BertEncoderV2),
      ("encoder_v1", bert_encoder.BertEncoder),
  )
  def test_dict_outputs_network_creation_return_attention_scores(
      self, encoder_cls):
    hidden_size = 32
    sequence_length = 21
    num_attention_heads = 5
    num_layers = 3
    # Create a small BertEncoder for testing.
    test_network = encoder_cls(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
        return_attention_scores=True,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    all_attention_outputs = dict_outputs["attention_scores"]

    expected_data_shape = [
        None, num_attention_heads, sequence_length, sequence_length
    ]
    self.assertLen(all_attention_outputs, num_layers)
    for data in all_attention_outputs:
      self.assertAllEqual(expected_data_shape, data.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, all_attention_outputs[-1].dtype)

  @parameterized.named_parameters(
      ("encoder_v2", bert_encoder.BertEncoderV2),
      ("encoder_v1", bert_encoder.BertEncoder),
  )
  def test_dict_outputs_network_creation_with_float16_dtype(self, encoder_cls):
    hidden_size = 32
    sequence_length = 21
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # Create a small BertEncoder for testing.
    test_network = encoder_cls(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        dict_outputs=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
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
      ("all_sequence_encoder_v1", bert_encoder.BertEncoder, None, 21),
      ("output_range_encoder_v1", bert_encoder.BertEncoder, 1, 1),
      ("all_sequence_encoder_v2", bert_encoder.BertEncoderV2, None, 21),
      ("output_range_encoder_v2", bert_encoder.BertEncoderV2, 1, 1),
  )
  def test_dict_outputs_network_invocation(
      self, encoder_cls, output_range, out_seq_len):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    # Create a small BertEncoder for testing.
    test_network = encoder_cls(
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
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
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
    test_network = encoder_cls(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        dict_outputs=True)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[1], sequence_length)

    # Creates a BertEncoder with embedding_width != hidden_size
    test_network = encoder_cls(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16,
        dict_outputs=True)
    dict_outputs = test_network(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))

  def test_embeddings_as_inputs(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoderV2(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    test_network.build(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids))
    embeddings = test_network.get_embedding_layer()(word_ids)
    # Calls with the embeddings.
    dict_outputs = test_network(
        dict(
            input_word_embeddings=embeddings,
            input_mask=mask,
            input_type_ids=type_ids))
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

    with self.subTest("BertEncoder"):
      network = bert_encoder.BertEncoder(**kwargs)

      # Validate that the config can be forced to JSON.
      _ = network.to_json()

      # Tests model saving/loading with SavedModel.
      model_path = self.get_temp_dir() + "/model"
      network.save(model_path)
      _ = tf.keras.models.load_model(model_path)

      # Test model saving/loading with Keras V3.
      keras_path = self.get_temp_dir() + "/model.keras"
      network.save(keras_path)
      _ = tf.keras.models.load_model(keras_path)

    with self.subTest("BertEncoderV2"):
      new_net = bert_encoder.BertEncoderV2(**kwargs)
      inputs = new_net.inputs
      outputs = new_net(inputs)
      network_v2 = tf.keras.Model(inputs=inputs, outputs=outputs)

      # Validate that the config can be forced to JSON.
      _ = network_v2.to_json()

      # Tests model saving/loading with SavedModel.
      model_path = self.get_temp_dir() + "/v2_model"
      network_v2.save(model_path)
      _ = tf.keras.models.load_model(model_path)

      # Test model saving/loading with Keras V3.
      keras_path = self.get_temp_dir() + "/v2_model.keras"
      network_v2.save(keras_path)
      _ = tf.keras.models.load_model(keras_path)

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

  def test_attention_scores_output_network_creation(self):
    hidden_size = 32
    sequence_length = 21
    num_attention_heads = 5
    num_layers = 3
    # Create a small BertEncoder for testing.
    test_network = bert_encoder.BertEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
        return_attention_scores=True)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    _, _, all_attention_outputs = test_network([word_ids, mask, type_ids])

    expected_data_shape = [
        None, num_attention_heads, sequence_length, sequence_length
    ]
    self.assertLen(all_attention_outputs, num_layers)
    for data in all_attention_outputs:
      self.assertAllEqual(expected_data_shape, data.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, all_attention_outputs[-1].dtype)

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


class BertEncoderV2CompatibilityTest(tf.test.TestCase):

  def tearDown(self):
    super().tearDown()
    tf.keras.mixed_precision.set_global_policy("float32")

  def test_weights_forward_compatible(self):
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
        type_vocab_size=num_types)

    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    data = dict(
        input_word_ids=word_id_data,
        input_mask=mask_data,
        input_type_ids=type_id_data)

    # Create small BertEncoders for testing.
    new_net = bert_encoder.BertEncoderV2(**kwargs)
    _ = new_net(data)
    kwargs["dict_outputs"] = True
    old_net = bert_encoder.BertEncoder(**kwargs)
    _ = old_net(data)
    new_net._embedding_layer.set_weights(old_net._embedding_layer.get_weights())
    new_net._position_embedding_layer.set_weights(
        old_net._position_embedding_layer.get_weights())
    new_net._type_embedding_layer.set_weights(
        old_net._type_embedding_layer.get_weights())
    new_net._embedding_norm_layer.set_weights(
        old_net._embedding_norm_layer.get_weights())
    # embedding_dropout has no weights.

    if hasattr(old_net, "_embedding_projection"):
      new_net._embedding_projection.set_weights(
          old_net._embedding_projection.get_weights())
    # attention_mask_layer has no weights.

    new_net._pooler_layer.set_weights(old_net._pooler_layer.get_weights())

    for otl, ntl in zip(old_net._transformer_layers,
                        new_net._transformer_layers):
      ntl.set_weights(otl.get_weights())

    def check_output_close(data, net1, net2):
      output1 = net1(data)
      output2 = net2(data)
      for key in output1:
        self.assertAllClose(output1[key], output2[key])

    check_output_close(data, old_net, new_net)

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
        type_vocab_size=num_types)

    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    data = dict(
        input_word_ids=word_id_data,
        input_mask=mask_data,
        input_type_ids=type_id_data)

    kwargs["dict_outputs"] = True
    old_net = bert_encoder.BertEncoder(**kwargs)
    old_net_outputs = old_net(data)
    ckpt = tf.train.Checkpoint(net=old_net)
    path = ckpt.save(self.get_temp_dir())
    del kwargs["dict_outputs"]
    new_net = bert_encoder.BertEncoderV2(**kwargs)
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

    kwargs["dict_outputs"] = True
    old_net = bert_encoder.BertEncoder(**kwargs)
    inputs = old_net.inputs
    outputs = old_net(inputs)
    old_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    old_model_outputs = old_model(data)
    ckpt = tf.train.Checkpoint(net=old_model)
    path = ckpt.save(self.get_temp_dir())
    del kwargs["dict_outputs"]
    new_net = bert_encoder.BertEncoderV2(**kwargs)
    inputs = new_net.inputs
    outputs = new_net(inputs)
    new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    new_ckpt = tf.train.Checkpoint(net=new_model)
    status = new_ckpt.restore(path)

    status.assert_existing_objects_matched()
    new_model_outputs = new_model(data)

    self.assertAllEqual(old_model_outputs.keys(), new_model_outputs.keys())
    for key in old_model_outputs:
      self.assertAllClose(old_model_outputs[key], new_model_outputs[key])


if __name__ == "__main__":
  tf.test.main()
