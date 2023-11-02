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

"""Tests for BERT pretrainer model."""
import itertools

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_pretrainer


class BertPretrainerTest(tf.test.TestCase, parameterized.TestCase):

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
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    masked_lm_positions = tf_keras.Input(
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
    # Build a transformer network to use within the BERT trainer.
    test_network = networks.BertEncoder(vocab_size=100, num_layers=2)

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


class BertPretrainerV2Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (False, True),
      (False, True),
      (False, True),
      (False, True),
  ))
  def test_bert_pretrainerv2(self, dict_outputs, return_all_encoder_outputs,
                             use_customized_masked_lm, has_masked_lm_positions):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    del dict_outputs, return_all_encoder_outputs
    vocab_size = 100
    sequence_length = 512
    hidden_size = 48
    num_layers = 2
    test_network = networks.BertEncoderV2(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        max_sequence_length=sequence_length)
    _ = test_network(test_network.inputs)

    # Create a BERT trainer with the created network.
    if use_customized_masked_lm:
      customized_masked_lm = layers.MaskedLM(
          embedding_table=test_network.get_embedding_table())
    else:
      customized_masked_lm = None

    bert_trainer_model = bert_pretrainer.BertPretrainerV2(
        encoder_network=test_network, customized_masked_lm=customized_masked_lm)
    num_token_predictions = 20
    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    inputs = dict(
        input_word_ids=tf_keras.Input(shape=(sequence_length,), dtype=tf.int32),
        input_mask=tf_keras.Input(shape=(sequence_length,), dtype=tf.int32),
        input_type_ids=tf_keras.Input(shape=(sequence_length,), dtype=tf.int32))
    if has_masked_lm_positions:
      inputs['masked_lm_positions'] = tf_keras.Input(
          shape=(num_token_predictions,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = bert_trainer_model(inputs)

    has_encoder_outputs = True  # dict_outputs or return_all_encoder_outputs
    expected_keys = ['sequence_output', 'pooled_output']
    if has_encoder_outputs:
      expected_keys.append('encoder_outputs')
    if has_masked_lm_positions:
      expected_keys.append('mlm_logits')

    self.assertSameElements(outputs.keys(), expected_keys)
    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [None, num_token_predictions, vocab_size]
    if has_masked_lm_positions:
      self.assertAllEqual(expected_lm_shape,
                          outputs['mlm_logits'].shape.as_list())

    expected_sequence_output_shape = [None, sequence_length, hidden_size]
    self.assertAllEqual(expected_sequence_output_shape,
                        outputs['sequence_output'].shape.as_list())

    expected_pooled_output_shape = [None, hidden_size]
    self.assertAllEqual(expected_pooled_output_shape,
                        outputs['pooled_output'].shape.as_list())

  def test_multiple_cls_outputs(self):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    sequence_length = 512
    hidden_size = 48
    num_layers = 2
    test_network = networks.BertEncoderV2(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        max_sequence_length=sequence_length)

    bert_trainer_model = bert_pretrainer.BertPretrainerV2(
        encoder_network=test_network,
        classification_heads=[layers.MultiClsHeads(
            inner_dim=5, cls_list=[('foo', 2), ('bar', 3)])])
    num_token_predictions = 20
    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    inputs = dict(
        input_word_ids=tf_keras.Input(shape=(sequence_length,), dtype=tf.int32),
        input_mask=tf_keras.Input(shape=(sequence_length,), dtype=tf.int32),
        input_type_ids=tf_keras.Input(shape=(sequence_length,), dtype=tf.int32),
        masked_lm_positions=tf_keras.Input(
            shape=(num_token_predictions,), dtype=tf.int32))

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = bert_trainer_model(inputs)
    self.assertEqual(outputs['foo'].shape.as_list(), [None, 2])
    self.assertEqual(outputs['bar'].shape.as_list(), [None, 3])

  def test_v2_serialize_deserialize(self):
    """Validate that the BERT trainer can be serialized and deserialized."""
    # Build a transformer network to use within the BERT trainer.
    test_network = networks.BertEncoderV2(vocab_size=100, num_layers=2)

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
