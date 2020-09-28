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
"""Tests for BERT pretrainer model."""
import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_pretrainer


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class BertPretrainerTest(keras_parameterized.TestCase):

  def test_bert_pretrainer(self):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    sequence_length = 512
    test_network = networks.BertEncoder(
        vocab_size=vocab_size,
        num_layers=2,
        max_sequence_length=sequence_length)

    # Create a BERT trainer with the created network.
    num_classes = 3
    num_token_predictions = 2
    bert_trainer_model = bert_pretrainer.BertPretrainer(
        test_network,
        num_classes=num_classes,
        num_token_predictions=num_token_predictions)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    masked_lm_positions = tf.keras.Input(
        shape=(num_token_predictions,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = bert_trainer_model(
        [word_ids, mask, type_ids, masked_lm_positions])

    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [None, num_token_predictions, vocab_size]
    expected_classification_shape = [None, num_classes]
    self.assertAllEqual(expected_lm_shape, outputs['masked_lm'].shape.as_list())
    self.assertAllEqual(expected_classification_shape,
                        outputs['classification'].shape.as_list())

  def test_bert_trainer_tensor_call(self):
    """Validate that the Keras object can be invoked."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_network = networks.BertEncoder(
        vocab_size=100, num_layers=2, sequence_length=2)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_pretrainer.BertPretrainer(
        test_network, num_classes=2, num_token_predictions=2)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
    mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
    type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
    lm_mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)

    # Invoke the trainer model on the tensors. In Eager mode, this does the
    # actual calculation. (We can't validate the outputs, since the network is
    # too complex: this simply ensures we're not hitting runtime errors.)
    _ = bert_trainer_model([word_ids, mask, type_ids, lm_mask])

  def test_serialize_deserialize(self):
    """Validate that the BERT trainer can be serialized and deserialized."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_network = networks.BertEncoder(
        vocab_size=100, num_layers=2, max_sequence_length=5)

    # Create a BERT trainer with the created network. (Note that all the args
    # are different, so we can catch any serialization mismatches.)
    bert_trainer_model = bert_pretrainer.BertPretrainer(
        test_network, num_classes=4, num_token_predictions=3)

    # Create another BERT trainer via serialization and deserialization.
    config = bert_trainer_model.get_config()
    new_bert_trainer_model = bert_pretrainer.BertPretrainer.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_bert_trainer_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(bert_trainer_model.get_config(),
                        new_bert_trainer_model.get_config())

  @parameterized.parameters(itertools.product(
      (False, True),
      (False, True),
  ))
  def test_bert_pretrainerv2(self, dict_outputs, return_all_encoder_outputs):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    sequence_length = 512
    hidden_size = 48
    num_layers = 2
    test_network = networks.BertEncoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        max_sequence_length=sequence_length,
        return_all_encoder_outputs=return_all_encoder_outputs,
        dict_outputs=dict_outputs)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_pretrainer.BertPretrainerV2(
        encoder_network=test_network)
    num_token_predictions = 20
    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    lm_mask = tf.keras.Input(shape=(num_token_predictions,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = bert_trainer_model([word_ids, mask, type_ids, lm_mask])

    has_encoder_outputs = dict_outputs or return_all_encoder_outputs
    if has_encoder_outputs:
      self.assertSameElements(
          outputs.keys(),
          ['sequence_output', 'pooled_output', 'mlm_logits', 'encoder_outputs'])
      self.assertLen(outputs['encoder_outputs'], num_layers)
    else:
      self.assertSameElements(
          outputs.keys(), ['sequence_output', 'pooled_output', 'mlm_logits'])

    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [None, num_token_predictions, vocab_size]
    self.assertAllEqual(expected_lm_shape,
                        outputs['mlm_logits'].shape.as_list())

    expected_sequence_output_shape = [None, sequence_length, hidden_size]
    self.assertAllEqual(expected_sequence_output_shape,
                        outputs['sequence_output'].shape.as_list())

    expected_pooled_output_shape = [None, hidden_size]
    self.assertAllEqual(expected_pooled_output_shape,
                        outputs['pooled_output'].shape.as_list())

  def test_v2_serialize_deserialize(self):
    """Validate that the BERT trainer can be serialized and deserialized."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_network = networks.BertEncoder(
        vocab_size=100, num_layers=2, sequence_length=5)

    # Create a BERT trainer with the created network. (Note that all the args
    # are different, so we can catch any serialization mismatches.)
    bert_trainer_model = bert_pretrainer.BertPretrainerV2(
        encoder_network=test_network)

    # Create another BERT trainer via serialization and deserialization.
    config = bert_trainer_model.get_config()
    new_bert_trainer_model = bert_pretrainer.BertPretrainerV2.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_bert_trainer_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(bert_trainer_model.get_config(),
                        new_bert_trainer_model.get_config())


if __name__ == '__main__':
  tf.test.main()
