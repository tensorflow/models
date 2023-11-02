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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.networks import funnel_transformer


class SingleLayerModel(tf_keras.Model):

  def __init__(self, layer):
    super().__init__()
    self.layer = layer

  def call(self, inputs):
    return self.layer(inputs)


class FunnelTransformerEncoderTest(parameterized.TestCase, tf.test.TestCase):

  def tearDown(self):
    super(FunnelTransformerEncoderTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy("float32")

  @parameterized.named_parameters(
      ("mix_truncated_avg_rezero", "mixed_float16", tf.float16, "truncated_avg",
       "ReZeroTransformer"), ("float32_truncated_avg_rezero", "float32",
                              tf.float32, "truncated_avg", "ReZeroTransformer"),
      ("mix_truncated_avg", "mixed_float16", tf.float16, "truncated_avg",
       "TransformerEncoderBlock"),
      ("float32_truncated_avg", "float32", tf.float32, "truncated_avg",
       "TransformerEncoderBlock"), ("mix_max", "mixed_float16", tf.float16,
                                    "max", "TransformerEncoderBlock"),
      ("float32_max", "float32", tf.float32, "max", "TransformerEncoderBlock"),
      ("mix_avg", "mixed_float16", tf.float16, "avg",
       "TransformerEncoderBlock"),
      ("float32_avg", "float32", tf.float32, "avg", "TransformerEncoderBlock"))
  def test_network_creation(self, policy, pooled_dtype, pool_type,
                            transformer_cls):
    tf_keras.mixed_precision.set_global_policy(policy)

    hidden_size = 32
    sequence_length = 21
    pool_stride = 2
    num_layers = 3
    # Create a small FunnelTransformerEncoder for testing.
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=num_layers,
        pool_stride=pool_stride,
        pool_type=pool_type,
        max_sequence_length=sequence_length,
        unpool_length=0,
        transformer_cls=transformer_cls)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, num_layers)
    self.assertIsInstance(test_network.pooler_layer, tf_keras.layers.Dense)

    # Stride=2 compresses sequence length to half the size at each layer.
    # For pool_type = max or avg,
    # this configuration gives each layer of seq length: 21->11->6->3.
    # For pool_type = truncated_avg,
    # seq length: 21->10->5->2.
    if pool_type in ["max", "avg"]:
      expected_data_shape = [None, 3, hidden_size]
    else:
      expected_data_shape = [None, 2, hidden_size]
    expected_pooled_shape = [None, hidden_size]

    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    # If float_dtype is set to float16, the data output is float32 (from a layer
    # norm) and pool output should be float16.
    self.assertAllEqual(tf.float32, data.dtype)
    self.assertAllEqual(pooled_dtype, pooled.dtype)

  @parameterized.named_parameters(
      ("append_dense_inputs", True),
      ("dense_inputs_at_sequence_begin", False),
  )
  def test_network_creation_dense(self, append_dense_inputs):
    tf_keras.mixed_precision.set_global_policy("mixed_float16")
    pool_type = "avg"

    hidden_size = 32
    sequence_length = 21
    dense_sequence_length = 3
    pool_stride = 2
    num_layers = 3
    # Create a small FunnelTransformerEncoder for testing.
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=num_layers,
        pool_stride=pool_stride,
        pool_type=pool_type,
        max_sequence_length=sequence_length + dense_sequence_length,
        unpool_length=0,
        transformer_cls="TransformerEncoderBlock",
        append_dense_inputs=append_dense_inputs)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)

    dense_inputs = tf_keras.Input(
        shape=(dense_sequence_length, hidden_size), dtype=tf.float32)
    dense_mask = tf_keras.Input(shape=(dense_sequence_length,), dtype=tf.int32)
    dense_type_ids = tf_keras.Input(
        shape=(dense_sequence_length,), dtype=tf.int32)

    dict_outputs = test_network(
        [word_ids, mask, type_ids, dense_inputs, dense_mask, dense_type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]

    self.assertIsInstance(test_network.transformer_layers, list)
    self.assertLen(test_network.transformer_layers, num_layers)
    self.assertIsInstance(test_network.pooler_layer, tf_keras.layers.Dense)

    # Stride=2 compresses sequence length to half the size at each layer.
    # For pool_type = max or avg,
    # this configuration gives each layer of seq length: 24->12->6->3.
    expected_data_shape = [None, 3, hidden_size]
    expected_pooled_shape = [None, hidden_size]

    self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

  @parameterized.named_parameters(
      ("frac_pool_rezero", "ReZeroTransformer"),
      ("frac_pool_vanilla", "TransformerEncoderBlock"),
      )
  def test_fractional_pooling(self, transformer_cls):
    hidden_size = 16
    sequence_length = 32
    pool_strides = [1.33333, 3, 2, 1]
    num_layers = 4
    pool_type = "truncated_avg"
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=num_layers,
        pool_stride=pool_strides,
        pool_type=pool_type,
        max_sequence_length=sequence_length,
        unpool_length=0,
        transformer_cls=transformer_cls)
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]

    expected_data_shape = [None, 4, hidden_size]

    self.assertAllEqual(expected_data_shape, data.shape.as_list())

  def test_invalid_stride_and_num_layers(self):
    hidden_size = 32
    num_layers = 3
    pool_stride = [2, 2]
    unpool_length = 1
    with self.assertRaisesRegex(ValueError,
                                "pool_stride and num_layers are not equal"):
      _ = funnel_transformer.FunnelTransformerEncoder(
          vocab_size=100,
          hidden_size=hidden_size,
          num_attention_heads=2,
          num_layers=num_layers,
          pool_stride=pool_stride,
          unpool_length=unpool_length)

  @parameterized.named_parameters(
      ("no_stride_no_unpool", 1, 0),
      ("stride_list_with_unpool", [2, 3, 4], 1),
      ("large_stride_with_unpool", 3, 1),
      ("large_stride_with_large_unpool", 5, 10),
      ("no_stride_with_unpool", 1, 1),
  )
  def test_all_encoder_outputs_network_creation(self, pool_stride,
                                                unpool_length):
    hidden_size = 32
    sequence_length = 21
    num_layers = 3
    # Create a small FunnelTransformerEncoder for testing.
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=num_layers,
        pool_stride=pool_stride,
        unpool_length=unpool_length)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids])
    all_encoder_outputs = dict_outputs["encoder_outputs"]
    pooled = dict_outputs["pooled_output"]

    expected_data_shape = [None, sequence_length, hidden_size]
    expected_pooled_shape = [None, hidden_size]
    self.assertLen(all_encoder_outputs, num_layers)
    if isinstance(pool_stride, int):
      pool_stride = [pool_stride] * num_layers
    for layer_pool_stride, data in zip(pool_stride, all_encoder_outputs):
      expected_data_shape[1] = unpool_length + (
          expected_data_shape[1] + layer_pool_stride - 1 -
          unpool_length) // layer_pool_stride
      self.assertAllEqual(expected_data_shape, data.shape.as_list())
    self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

    # The default output dtype is float32.
    self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
    self.assertAllEqual(tf.float32, pooled.dtype)

  @parameterized.named_parameters(
      ("all_sequence", None, 3, 0, 2),
      ("output_range", 1, 1, 0, 2),
      ("all_sequence_with_unpool", None, 4, 1, 2),
      ("output_range_with_unpool", 1, 1, 1, 2),
      ("output_range_with_large_unpool", 1, 1, 2, 2),
      ("output_range_with_no_pooling", 1, 1, 0, 1),
      ("output_range_with_unpool_and_no_pooling", 1, 1, 1, 1),
  )
  def test_network_invocation(
      self,
      output_range,
      out_seq_len,
      unpool_length,
      pool_stride,
  ):
    hidden_size = 32
    sequence_length = 21
    vocab_size = 57
    num_types = 7
    num_layers = 3
    # Create a small FunnelTransformerEncoder for testing.
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=num_layers,
        type_vocab_size=num_types,
        pool_stride=pool_stride,
        unpool_length=unpool_length)
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    dict_outputs = test_network([word_ids, mask, type_ids],
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
    self.assertEqual(outputs[0].shape[1], out_seq_len)  # output_range

    # Creates a FunnelTransformerEncoder with max_sequence_length !=
    # sequence_length
    max_sequence_length = 128
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=num_layers,
        type_vocab_size=num_types,
        pool_stride=pool_stride)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    expected_sequence_length = float(sequence_length)
    for _ in range(num_layers):
      expected_sequence_length = np.ceil(expected_sequence_length / pool_stride)
    self.assertEqual(outputs[0].shape[1], expected_sequence_length)

    # Creates a FunnelTransformerEncoder with embedding_width != hidden_size
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_attention_heads=2,
        num_layers=3,
        type_vocab_size=num_types,
        embedding_width=16,
        pool_stride=pool_stride)
    dict_outputs = test_network([word_ids, mask, type_ids])
    data = dict_outputs["sequence_output"]
    pooled = dict_outputs["pooled_output"]
    model = tf_keras.Model([word_ids, mask, type_ids], [data, pooled])
    outputs = model.predict([word_id_data, mask_data, type_id_data])
    self.assertEqual(outputs[0].shape[-1], hidden_size)
    self.assertTrue(hasattr(test_network, "_embedding_projection"))

  def test_embeddings_as_inputs(self):
    hidden_size = 32
    sequence_length = 21
    # Create a small BertEncoder for testing.
    test_network = funnel_transformer.FunnelTransformerEncoder(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_layers=3,
        pool_stride=2,
    )
    # Create the inputs (note that the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    test_network.build(
        dict(input_word_ids=word_ids, input_mask=mask, input_type_ids=type_ids)
    )
    embeddings = test_network.get_embedding_layer()(word_ids)
    # Calls with the embeddings.
    dict_outputs = test_network(
        dict(
            input_word_embeddings=embeddings,
            input_mask=mask,
            input_type_ids=type_ids,
        )
    )
    all_encoder_outputs = dict_outputs["encoder_outputs"]
    pooled = dict_outputs["pooled_output"]

    expected_pooled_shape = [None, hidden_size]
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
        norm_first=False,
        pool_type="max",
        pool_stride=2,
        unpool_length=0,
        transformer_cls="TransformerEncoderBlock")
    network = funnel_transformer.FunnelTransformerEncoder(**kwargs)
    expected_config = dict(kwargs)
    expected_config["inner_activation"] = tf_keras.activations.serialize(
        tf_keras.activations.get(expected_config["inner_activation"]))
    expected_config["initializer"] = tf_keras.initializers.serialize(
        tf_keras.initializers.get(expected_config["initializer"]))
    self.assertEqual(network.get_config(), expected_config)
    # Create another network object from the first object's config.
    new_network = funnel_transformer.FunnelTransformerEncoder.from_config(
        network.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

    # Tests model saving/loading.
    model_path = self.get_temp_dir() + "/model"
    network_wrapper = SingleLayerModel(network)
    # One forward-path to ensure input_shape.
    batch_size = 3
    sequence_length = 21
    vocab_size = 100
    num_types = 12
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(
        num_types, size=(batch_size, sequence_length))

    _ = network_wrapper.predict([word_id_data, mask_data, type_id_data])
    network_wrapper.save(model_path)
    _ = tf_keras.models.load_model(model_path)


if __name__ == "__main__":
  tf.test.main()
