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

"""Tests for XLNet classifier network."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling import networks
from official.nlp.modeling.models import xlnet


def _get_xlnet_base() -> tf_keras.layers.Layer:
  """Returns a trivial base XLNet model."""
  return networks.XLNetBase(
      vocab_size=100,
      num_layers=2,
      hidden_size=4,
      num_attention_heads=2,
      head_size=2,
      inner_size=2,
      dropout_rate=0.,
      attention_dropout_rate=0.,
      attention_type='bi',
      bi_data=True,
      initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
      two_stream=False,
      tie_attention_biases=True,
      reuse_length=0,
      inner_activation='relu')


class XLNetMaskedLMTest(tf.test.TestCase):

  def test_xlnet_masked_lm_head(self):
    hidden_size = 10
    seq_length = 8
    batch_size = 2
    masked_lm = xlnet.XLNetMaskedLM(vocab_size=10,
                                    hidden_size=hidden_size,
                                    initializer='glorot_uniform')
    sequence_data = np.random.uniform(size=(batch_size, seq_length))
    embedding_table = np.random.uniform(size=(hidden_size, hidden_size))
    mlm_output = masked_lm(sequence_data, embedding_table)
    self.assertAllClose(mlm_output.shape, (batch_size, hidden_size))


class XLNetPretrainerTest(tf.test.TestCase):

  def test_xlnet_trainer(self):
    """Validates that the Keras object can be created."""
    seq_length = 4
    num_predictions = 2
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetPretrainer(network=xlnet_base)
    inputs = dict(
        input_word_ids=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_word_ids'),
        input_type_ids=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_type_ids'),
        input_mask=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_mask'),
        permutation_mask=tf_keras.layers.Input(
            shape=(seq_length, seq_length,), dtype=tf.int32,
            name='permutation_mask'),
        target_mapping=tf_keras.layers.Input(
            shape=(num_predictions, seq_length), dtype=tf.int32,
            name='target_mapping'),
        masked_tokens=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='masked_tokens'))
    logits, _ = xlnet_trainer_model(inputs)

    # [None, hidden_size, vocab_size]
    expected_output_shape = [None, 4, 100]
    self.assertAllEqual(expected_output_shape, logits.shape.as_list())

  def test_xlnet_tensor_call(self):
    """Validates that the Keras object can be invoked."""
    seq_length = 4
    batch_size = 2
    num_predictions = 2
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetPretrainer(network=xlnet_base)

    sequence_shape = (batch_size, seq_length)
    inputs = dict(
        input_word_ids=np.random.randint(
            10, size=sequence_shape, dtype='int32'),
        input_type_ids=np.random.randint(2, size=sequence_shape, dtype='int32'),
        input_mask=np.random.randint(2, size=sequence_shape).astype('int32'),
        permutation_mask=np.random.randint(
            2, size=(batch_size, seq_length, seq_length)).astype('int32'),
        target_mapping=np.random.randint(
            10, size=(num_predictions, seq_length), dtype='int32'),
        masked_tokens=np.random.randint(
            10, size=sequence_shape, dtype='int32'))
    xlnet_trainer_model(inputs)

  def test_serialize_deserialize(self):
    """Validates that the XLNet trainer can be serialized and deserialized."""
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetPretrainer(
        network=xlnet_base,
        mlm_activation='gelu',
        mlm_initializer='random_normal')

    # Create another XLNet trainer via serialization and deserialization.
    config = xlnet_trainer_model.get_config()
    new_xlnet_trainer_model = xlnet.XLNetPretrainer.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_xlnet_trainer_model.to_json()

    # If serialization was successful, then the new config should match the old.
    self.assertAllEqual(xlnet_trainer_model.get_config(),
                        new_xlnet_trainer_model.get_config())


class XLNetClassifierTest(tf.test.TestCase, parameterized.TestCase):

  def test_xlnet_trainer(self):
    """Validate that the Keras object can be created."""
    num_classes = 2
    seq_length = 4
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetClassifier(
        network=xlnet_base,
        num_classes=num_classes,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        summary_type='last',
        dropout_rate=0.1)
    inputs = dict(
        input_word_ids=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_word_ids'),
        input_type_ids=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_type_ids'),
        input_mask=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_mask'),
        permutation_mask=tf_keras.layers.Input(
            shape=(seq_length, seq_length,), dtype=tf.int32,
            name='permutation_mask'),
        masked_tokens=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='masked_tokens'))
    logits = xlnet_trainer_model(inputs)

    expected_classification_shape = [None, num_classes]
    self.assertAllEqual(expected_classification_shape, logits.shape.as_list())

  @parameterized.parameters(1, 2)
  def test_xlnet_tensor_call(self, num_classes):
    """Validates that the Keras object can be invoked."""
    seq_length = 4
    batch_size = 2
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetClassifier(
        network=xlnet_base,
        num_classes=num_classes,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        summary_type='last',
        dropout_rate=0.1)

    sequence_shape = (batch_size, seq_length)
    inputs = dict(
        input_word_ids=np.random.randint(
            10, size=sequence_shape, dtype='int32'),
        input_type_ids=np.random.randint(2, size=sequence_shape, dtype='int32'),
        input_mask=np.random.randint(2, size=sequence_shape).astype('int32'),
        permutation_mask=np.random.randint(
            2, size=(batch_size, seq_length, seq_length)).astype('int32'),
        masked_tokens=np.random.randint(
            10, size=sequence_shape, dtype='int32'))
    xlnet_trainer_model(inputs)

  def test_serialize_deserialize(self):
    """Validates that the XLNet trainer can be serialized and deserialized."""
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetClassifier(
        network=xlnet_base,
        num_classes=2,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        summary_type='last',
        dropout_rate=0.1)

    # Create another XLNet trainer via serialization and deserialization.
    config = xlnet_trainer_model.get_config()
    new_xlnet_trainer_model = xlnet.XLNetClassifier.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_xlnet_trainer_model.to_json()

    # If serialization was successful, then the new config should match the old.
    self.assertAllEqual(xlnet_trainer_model.get_config(),
                        new_xlnet_trainer_model.get_config())


class XLNetSpanLabelerTest(tf.test.TestCase):

  def test_xlnet_trainer(self):
    """Validate that the Keras object can be created."""
    top_n = 2
    seq_length = 4
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetSpanLabeler(
        network=xlnet_base,
        start_n_top=top_n,
        end_n_top=top_n,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        span_labeling_activation='tanh',
        dropout_rate=0.1)
    inputs = dict(
        input_word_ids=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_word_ids'),
        input_type_ids=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_type_ids'),
        input_mask=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_mask'),
        paragraph_mask=tf_keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='paragraph_mask'),
        class_index=tf_keras.layers.Input(
            shape=(), dtype=tf.int32, name='class_index'),
        start_positions=tf_keras.layers.Input(
            shape=(), dtype=tf.int32, name='start_positions'))
    outputs = xlnet_trainer_model(inputs)
    self.assertIsInstance(outputs, dict)

    # Test tensor value calls for the created model.
    batch_size = 2
    sequence_shape = (batch_size, seq_length)
    inputs = dict(
        input_word_ids=np.random.randint(
            10, size=sequence_shape, dtype='int32'),
        input_type_ids=np.random.randint(2, size=sequence_shape, dtype='int32'),
        input_mask=np.random.randint(2, size=sequence_shape).astype('int32'),
        paragraph_mask=np.random.randint(
            1, size=(sequence_shape)).astype('int32'),
        class_index=np.random.randint(1, size=(batch_size)).astype('uint8'),
        start_positions=tf.random.uniform(
            shape=(batch_size,), maxval=5, dtype=tf.int32))

    common_keys = {
        'start_logits', 'end_logits', 'start_predictions', 'end_predictions',
        'class_logits',
    }
    inference_keys = {
        'start_top_predictions', 'end_top_predictions', 'start_top_index',
        'end_top_index',
    }

    outputs = xlnet_trainer_model(inputs)
    self.assertSetEqual(common_keys | inference_keys, set(outputs.keys()))

    outputs = xlnet_trainer_model(inputs, training=True)
    self.assertIsInstance(outputs, dict)
    self.assertSetEqual(common_keys, set(outputs.keys()))
    self.assertIsInstance(outputs, dict)

  def test_serialize_deserialize(self):
    """Validates that the XLNet trainer can be serialized and deserialized."""
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetSpanLabeler(
        network=xlnet_base,
        start_n_top=2,
        end_n_top=2,
        initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
        span_labeling_activation='tanh',
        dropout_rate=0.1)

    # Create another XLNet trainer via serialization and deserialization.
    config = xlnet_trainer_model.get_config()
    new_xlnet_trainer_model = xlnet.XLNetSpanLabeler.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_xlnet_trainer_model.to_json()

    # If serialization was successful, then the new config should match the old.
    self.assertAllEqual(xlnet_trainer_model.get_config(),
                        new_xlnet_trainer_model.get_config())


if __name__ == '__main__':
  tf.test.main()
