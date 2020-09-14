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
"""Tests for EncoderScaffold network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations
from official.nlp.modeling import layers
from official.nlp.modeling.networks import encoder_scaffold


# Test class that wraps a standard transformer layer. If this layer is called
# at any point, the list passed to the config object will be filled with a
# boolean 'True'. We register this class as a Keras serializable so we can
# test serialization below.
@tf.keras.utils.register_keras_serializable(package="TestOnly")
class ValidatedTransformerLayer(layers.Transformer):

  def __init__(self, call_list, **kwargs):
    super(ValidatedTransformerLayer, self).__init__(**kwargs)
    self.list = call_list

  def call(self, inputs):
    self.list.append(True)
    return super(ValidatedTransformerLayer, self).call(inputs)

  def get_config(self):
    config = super(ValidatedTransformerLayer, self).get_config()
    config["call_list"] = []
    return config


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class EncoderScaffoldLayerClassTest(keras_parameterized.TestCase):

  def tearDown(self):
    super(EncoderScaffoldLayerClassTest, self).tearDown()
    tf.keras.mixed_precision.experimental.set_policy("float32")

  @parameterized.named_parameters(
      dict(testcase_name="only_final_output", return_all_layer_outputs=False),
      dict(testcase_name="all_layer_outputs", return_all_layer_outputs=True))
  def test_network_creation(self, return_all_layer_outputs):
    hidden_size = 32
    sequence_length = 21
    num_hidden_instances = 3
    embedding_cfg = {
        "vocab_size": 100,
        "type_vocab_size": 16,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }

    call_list = []
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "call_list":
            call_list
    }
    # Create a small EncoderScaffold for testing.
    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=num_hidden_instances,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cls=ValidatedTransformerLayer,
        hidden_cfg=hidden_cfg,
        embedding_cfg=embedding_cfg,
        return_all_layer_outputs=return_all_layer_outputs)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    output_data, pooled = test_network([word_ids, mask, type_ids])

    if return_all_layer_outputs:
      self.assertIsInstance(output_data, list)
      self.assertLen(output_data, num_hidden_instances)
      data = output_data[-1]
    else:
      data = output_data
    self.assertIsInstance(test_network.hidden_layers, list)
    self.assertLen(test_network.hidden_layers, num_hidden_instances)
    self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

    # If call_list[0] exists and is True, the passed layer class was
    # instantiated from the given config properly.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_network_creation_with_float16_dtype(self):
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    hidden_size = 32
    sequence_length = 21
    embedding_cfg = {
        "vocab_size": 100,
        "type_vocab_size": 16,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }
    # Create a small EncoderScaffold for testing.
    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cfg=hidden_cfg,
        embedding_cfg=embedding_cfg)
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

  def test_network_invocation(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    embedding_cfg = {
        "vocab_size": vocab_size,
        "type_vocab_size": num_types,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }
    # Create a small EncoderScaffold for testing.
    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cfg=hidden_cfg,
        embedding_cfg=embedding_cfg,
        dict_outputs=True)

    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    outputs = test_network([word_ids, mask, type_ids])

    # Create a model based off of this network:
    model = tf.keras.Model([word_ids, mask, type_ids], outputs)

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    preds = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(preds["pooled_output"].shape, (3, hidden_size))

    # Creates a EncoderScaffold with max_sequence_length != sequence_length
    num_types = 7
    embedding_cfg = {
        "vocab_size": vocab_size,
        "type_vocab_size": num_types,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length * 2,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }
    # Create a small EncoderScaffold for testing.
    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cfg=hidden_cfg,
        embedding_cfg=embedding_cfg)
    outputs = test_network([word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids], outputs)
    _ = model.predict([word_id_data, mask_data, type_id_data])

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    hidden_size = 32
    sequence_length = 21
    embedding_cfg = {
        "vocab_size": 100,
        "type_vocab_size": 16,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }
    # Create a small EncoderScaffold for testing.
    network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cfg=hidden_cfg,
        embedding_cfg=embedding_cfg)

    # Create another network object from the first object's config.
    new_network = encoder_scaffold.EncoderScaffold.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


class Embeddings(tf.keras.Model):

  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.inputs = [
        tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name="input_word_ids"),
        tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
    ]
    self.attention_mask = layers.SelfAttentionMask()
    self.embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="word_embeddings")

  def call(self, inputs):
    word_ids, mask = inputs
    word_embeddings = self.embedding_layer(word_ids)
    return word_embeddings, self.attention_mask([word_embeddings, mask])


@keras_parameterized.run_all_keras_modes
class EncoderScaffoldEmbeddingNetworkTest(keras_parameterized.TestCase):

  def test_network_invocation(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57

    # Build an embedding network to swap in for the default network. This one
    # will have 2 inputs (mask and word_ids) instead of 3, and won't use
    # positional embeddings.
    network = Embeddings(vocab_size, hidden_size)

    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }

    # Create a small EncoderScaffold for testing.
    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cfg=hidden_cfg,
        embedding_cls=network)

    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    data, pooled = test_network([word_ids, mask])

    # Create a model based off of this network:
    model = tf.keras.Model([word_ids, mask], [data, pooled])

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    _ = model.predict([word_id_data, mask_data])

  def test_serialize_deserialize(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57

    # Build an embedding network to swap in for the default network. This one
    # will have 2 inputs (mask and word_ids) instead of 3, and won't use
    # positional embeddings.

    word_ids = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name="input_word_ids")
    mask = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name="input_mask")
    embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="word_embeddings")
    word_embeddings = embedding_layer(word_ids)
    attention_mask = layers.SelfAttentionMask()([word_embeddings, mask])
    network = tf.keras.Model([word_ids, mask],
                             [word_embeddings, attention_mask])

    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }

    # Create a small EncoderScaffold for testing.
    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cfg=hidden_cfg,
        embedding_cls=network,
        embedding_data=embedding_layer.embeddings)

    # Create another network object from the first object's config.
    new_network = encoder_scaffold.EncoderScaffold.from_config(
        test_network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_network.get_config(), new_network.get_config())

    # Create a model based off of the old and new networks:
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    data, pooled = new_network([word_ids, mask])
    new_model = tf.keras.Model([word_ids, mask], [data, pooled])

    data, pooled = test_network([word_ids, mask])
    model = tf.keras.Model([word_ids, mask], [data, pooled])

    # Copy the weights between models.
    new_model.set_weights(model.get_weights())

    # Invoke the models.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    data, cls = model.predict([word_id_data, mask_data])
    new_data, new_cls = new_model.predict([word_id_data, mask_data])

    # The output should be equal.
    self.assertAllEqual(data, new_data)
    self.assertAllEqual(cls, new_cls)

    # We should not be able to get a reference to the embedding data.
    with self.assertRaisesRegex(RuntimeError, ".*does not have a reference.*"):
      new_network.get_embedding_table()


@keras_parameterized.run_all_keras_modes
class EncoderScaffoldHiddenInstanceTest(keras_parameterized.TestCase):

  def test_network_invocation(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7

    embedding_cfg = {
        "vocab_size": vocab_size,
        "type_vocab_size": num_types,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }

    call_list = []
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "call_list":
            call_list
    }
    # Create a small EncoderScaffold for testing. This time, we pass an already-
    # instantiated layer object.

    xformer = ValidatedTransformerLayer(**hidden_cfg)

    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cls=xformer,
        embedding_cfg=embedding_cfg)

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

    # If call_list[0] exists and is True, the passed layer class was
    # called as part of the graph creation.
    self.assertNotEmpty(call_list)
    self.assertTrue(call_list[0], "The passed layer class wasn't instantiated.")

  def test_serialize_deserialize(self):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7

    embedding_cfg = {
        "vocab_size": vocab_size,
        "type_vocab_size": num_types,
        "hidden_size": hidden_size,
        "seq_length": sequence_length,
        "max_seq_length": sequence_length,
        "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "dropout_rate": 0.1,
    }

    call_list = []
    hidden_cfg = {
        "num_attention_heads":
            2,
        "intermediate_size":
            3072,
        "intermediate_activation":
            activations.gelu,
        "dropout_rate":
            0.1,
        "attention_dropout_rate":
            0.1,
        "kernel_initializer":
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
        "call_list":
            call_list
    }
    # Create a small EncoderScaffold for testing. This time, we pass an already-
    # instantiated layer object.

    xformer = ValidatedTransformerLayer(**hidden_cfg)

    test_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=3,
        pooled_output_dim=hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        hidden_cls=xformer,
        embedding_cfg=embedding_cfg)

    # Create another network object from the first object's config.
    new_network = encoder_scaffold.EncoderScaffold.from_config(
        test_network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_network.get_config(), new_network.get_config())

    # Create a model based off of the old and new networks:
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    data, pooled = new_network([word_ids, mask, type_ids])
    new_model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])

    data, pooled = test_network([word_ids, mask, type_ids])
    model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])

    # Copy the weights between models.
    new_model.set_weights(model.get_weights())

    # Invoke the models.
    batch_size = 3
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))
    data, cls = model.predict([word_id_data, mask_data, type_id_data])
    new_data, new_cls = new_model.predict(
        [word_id_data, mask_data, type_id_data])

    # The output should be equal.
    self.assertAllEqual(data, new_data)
    self.assertAllEqual(cls, new_cls)


if __name__ == "__main__":
  tf.test.main()
