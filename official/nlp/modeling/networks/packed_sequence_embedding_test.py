# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.modeling.networks.packed_sequence_embedding."""

# Import libraries

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.networks import packed_sequence_embedding


class PackedSequenceEmbeddingTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    super(PackedSequenceEmbeddingTest, self).tearDown()
    tf_keras.mixed_precision.set_global_policy('float32')

  @parameterized.parameters([
      (True, True, True),
      (False, False, True),
      (False, True, False),
      (True, False, False),
  ])
  def test_network_creation(self, use_position_id, pack_multiple_sequences,
                            use_float16):
    """Validate that the Keras object can be created."""
    if use_float16:
      tf_keras.mixed_precision.set_global_policy('mixed_float16')
    seq_length = 16
    vocab_size = 100
    max_position_embeddings = 32
    type_vocab_size = 2
    embedding_width = 16
    hidden_size = 32
    embedding_cfg = dict(
        vocab_size=vocab_size,
        type_vocab_size=2,
        embedding_width=embedding_width,
        hidden_size=hidden_size,
        max_seq_length=max_position_embeddings,
        initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02),
        dropout_rate=0.1,
        use_position_id=use_position_id,
        pack_multiple_sequences=pack_multiple_sequences,
    )
    test_object = packed_sequence_embedding.PackedSequenceEmbedding(
        **embedding_cfg)

    input_word_ids = tf_keras.Input(shape=(seq_length,), dtype=tf.int32)
    input_mask = tf_keras.Input(shape=(seq_length,), dtype=tf.int32)
    input_type_ids = tf_keras.Input(shape=(seq_length,), dtype=tf.int32)
    network_inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids,
    }
    if use_position_id:
      network_inputs['position_ids'] = tf_keras.Input(
          shape=(seq_length,), dtype=tf.int32)

    embedding, mask = test_object(network_inputs)

    # Create a model based off of this network:
    model = tf_keras.Model(network_inputs, [embedding, mask])

    # Invoke the model. We can't validate the output data here (the model is too
    # complex) but this will catch structural runtime errors.
    batch_size = 3
    word_id_data = np.random.randint(vocab_size, size=(batch_size, seq_length))
    mask_data = np.random.randint(2, size=(batch_size, seq_length))
    type_id_data = np.random.randint(
        type_vocab_size, size=(batch_size, seq_length))
    feed_input = {
        'input_word_ids': word_id_data,
        'input_mask': mask_data,
        'input_type_ids': type_id_data,
    }
    if use_position_id:
      feed_input['position_ids'] = np.random.randint(
          seq_length, size=(batch_size, seq_length))
    embeddings, attention_mask = model.predict(feed_input)
    expected_embeddings_shape = [3, seq_length, hidden_size]
    expected_attention_mask_shape = [3, seq_length, seq_length]
    self.assertAllEqual(expected_embeddings_shape, embeddings.shape)
    self.assertAllEqual(expected_attention_mask_shape, attention_mask.shape)

  def test_serialize_deserialize(self):
    tf_keras.mixed_precision.set_global_policy('mixed_float16')
    # Create a network object that sets all of its config options.
    embedding_cfg = dict(
        vocab_size=100,
        type_vocab_size=2,
        embedding_width=64,
        hidden_size=64,
        max_seq_length=32,
        initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02),
        dropout_rate=0.1,
        use_position_id=True,
        pack_multiple_sequences=False,
    )
    network = packed_sequence_embedding.PackedSequenceEmbedding(**embedding_cfg)

    expected_config = dict(embedding_cfg)
    expected_config['initializer'] = tf_keras.initializers.serialize(
        tf_keras.initializers.get(expected_config['initializer']))
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = packed_sequence_embedding.PackedSequenceEmbedding.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
