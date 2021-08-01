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

"""Tests for dual encoder network."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling import networks
from official.nlp.modeling.models import dual_encoder


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class DualEncoderTest(keras_parameterized.TestCase):

  @parameterized.parameters((192, 'logits'), (768, 'predictions'))
  def test_dual_encoder(self, hidden_size, output):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the dual encoder model.
    vocab_size = 100
    sequence_length = 512
    test_network = networks.BertEncoder(
        vocab_size=vocab_size,
        num_layers=2,
        hidden_size=hidden_size,
        dict_outputs=True)

    # Create a dual encoder model with the created network.
    dual_encoder_model = dual_encoder.DualEncoder(
        test_network, max_seq_length=sequence_length, output=output)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    left_word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    left_mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    left_type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    right_word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    right_mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    right_type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    if output == 'logits':
      outputs = dual_encoder_model([
          left_word_ids, left_mask, left_type_ids, right_word_ids, right_mask,
          right_type_ids
      ])
      _ = outputs['left_logits']
    elif output == 'predictions':
      outputs = dual_encoder_model([left_word_ids, left_mask, left_type_ids])
      # Validate that the outputs are of the expected shape.
      expected_sequence_shape = [None, sequence_length, 768]
      self.assertAllEqual(expected_sequence_shape,
                          outputs['sequence_output'].shape.as_list())
      left_encoded = outputs['pooled_output']
      expected_encoding_shape = [None, 768]
      self.assertAllEqual(expected_encoding_shape, left_encoded.shape.as_list())

  @parameterized.parameters((192, 'logits'), (768, 'predictions'))
  def test_dual_encoder_tensor_call(self, hidden_size, output):
    """Validate that the Keras object can be invoked."""
    # Build a transformer network to use within the dual encoder model.
    sequence_length = 2
    test_network = networks.BertEncoder(vocab_size=100, num_layers=2)

    # Create a dual encoder model with the created network.
    dual_encoder_model = dual_encoder.DualEncoder(
        test_network, max_seq_length=sequence_length, output=output)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
    mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
    type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)

    # Invoke the model model on the tensors. In Eager mode, this does the
    # actual calculation. (We can't validate the outputs, since the network is
    # too complex: this simply ensures we're not hitting runtime errors.)
    if output == 'logits':
      _ = dual_encoder_model(
          [word_ids, mask, type_ids, word_ids, mask, type_ids])
    elif output == 'predictions':
      _ = dual_encoder_model([word_ids, mask, type_ids])

  def test_serialize_deserialize(self):
    """Validate that the dual encoder model can be serialized / deserialized."""
    # Build a transformer network to use within the dual encoder model.
    sequence_length = 32
    test_network = networks.BertEncoder(vocab_size=100, num_layers=2)

    # Create a dual encoder model with the created network. (Note that all the
    # args are different, so we can catch any serialization mismatches.)
    dual_encoder_model = dual_encoder.DualEncoder(
        test_network, max_seq_length=sequence_length, output='predictions')

    # Create another dual encoder moel via serialization and deserialization.
    config = dual_encoder_model.get_config()
    new_dual_encoder = dual_encoder.DualEncoder.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_dual_encoder.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(dual_encoder_model.get_config(),
                        new_dual_encoder.get_config())


if __name__ == '__main__':
  tf.test.main()
