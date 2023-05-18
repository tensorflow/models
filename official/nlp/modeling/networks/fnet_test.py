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

"""Tests for FNet encoder network."""

from typing import Sequence

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.modeling import layers
from official.nlp.modeling.networks import fnet


class FNetTest(parameterized.TestCase, tf.test.TestCase):

  def tearDown(self):
    super(FNetTest, self).tearDown()
    tf.keras.mixed_precision.set_global_policy("float32")

  @parameterized.named_parameters(
      ("fnet", layers.MixingMechanism.FOURIER, ()),
      ("fnet_hybrid", layers.MixingMechanism.FOURIER, (1, 2)),
      ("hnet", layers.MixingMechanism.HARTLEY, ()),
      ("hnet_hybrid", layers.MixingMechanism.HARTLEY, (1, 2)),
      ("linear", layers.MixingMechanism.LINEAR, ()),
      ("linear_hybrid", layers.MixingMechanism.LINEAR, (0,)),
      ("bert", layers.MixingMechanism.FOURIER, (0, 1, 2)),
  )
  def test_network(self, mixing_mechanism: layers.MixingMechanism,
                   attention_layers: Sequence[int]):
    num_layers = 3
    hidden_size = 32
    sequence_length = 21
    test_network = fnet.FNet(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        max_sequence_length=sequence_length,
        num_layers=num_layers,
        mixing_mechanism=mixing_mechanism,
        attention_layers=attention_layers)

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

  def test_embeddings_as_inputs(self):
    hidden_size = 32
    sequence_length = 21
    test_network = fnet.FNet(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=2,
        max_sequence_length=sequence_length,
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


if __name__ == "__main__":
  tf.test.main()
