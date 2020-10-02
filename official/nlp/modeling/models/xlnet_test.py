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
"""Tests for XLNet classifier network."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling import networks
from official.nlp.modeling.models import xlnet


def _get_xlnet_base() -> tf.keras.layers.Layer:
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
      initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
      two_stream=False,
      tie_attention_biases=True,
      reuse_length=0,
      inner_activation='relu')


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class XLNetClassifierTest(keras_parameterized.TestCase):

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
        initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        summary_type='last',
        dropout_rate=0.1)
    inputs = dict(
        input_ids=tf.keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='input_word_ids'),
        segment_ids=tf.keras.layers.Input(
            shape=(seq_length,), dtype=tf.int32, name='segment_ids'),
        input_mask=tf.keras.layers.Input(
            shape=(seq_length,), dtype=tf.float32, name='input_mask'),
        permutation_mask=tf.keras.layers.Input(
            shape=(seq_length, seq_length,), dtype=tf.float32,
            name='permutation_mask'),
        masked_tokens=tf.keras.layers.Input(
            shape=(seq_length,), dtype=tf.float32, name='masked_tokens'))

    logits, _ = xlnet_trainer_model(inputs)

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
        initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        summary_type='last',
        dropout_rate=0.1)

    sequence_shape = (batch_size, seq_length)
    inputs = dict(
        input_ids=np.random.randint(10, size=sequence_shape, dtype='int32'),
        segment_ids=np.random.randint(2, size=sequence_shape, dtype='int32'),
        input_mask=np.random.randint(2, size=sequence_shape).astype('float32'),
        permutation_mask=np.random.randint(
            2, size=(batch_size, seq_length, seq_length)).astype('float32'),
        masked_tokens=tf.random.uniform(shape=sequence_shape))
    xlnet_trainer_model(inputs)

  def test_serialize_deserialize(self):
    """Validates that the XLNet trainer can be serialized and deserialized."""
    # Build a simple XLNet based network to use with the XLNet trainer.
    xlnet_base = _get_xlnet_base()

    # Create an XLNet trainer with the created network.
    xlnet_trainer_model = xlnet.XLNetClassifier(
        network=xlnet_base,
        num_classes=2,
        initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
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


if __name__ == '__main__':
  tf.test.main()
