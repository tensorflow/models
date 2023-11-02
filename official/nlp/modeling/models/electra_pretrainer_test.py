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

"""Tests for ELECTRA pre trainer network."""

import tensorflow as tf, tf_keras

from official.nlp.modeling import networks
from official.nlp.modeling.models import electra_pretrainer


class ElectraPretrainerTest(tf.test.TestCase):

  def test_electra_pretrainer(self):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the ELECTRA trainer.
    vocab_size = 100
    sequence_length = 512
    test_generator_network = networks.BertEncoder(
        vocab_size=vocab_size,
        num_layers=2,
        max_sequence_length=sequence_length,
        dict_outputs=True)
    test_discriminator_network = networks.BertEncoder(
        vocab_size=vocab_size,
        num_layers=2,
        max_sequence_length=sequence_length,
        dict_outputs=True)

    # Create a ELECTRA trainer with the created network.
    num_classes = 3
    num_token_predictions = 2
    eletrca_trainer_model = electra_pretrainer.ElectraPretrainer(
        generator_network=test_generator_network,
        discriminator_network=test_discriminator_network,
        vocab_size=vocab_size,
        num_classes=num_classes,
        num_token_predictions=num_token_predictions,
        disallow_correct=True)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    lm_positions = tf_keras.Input(
        shape=(num_token_predictions,), dtype=tf.int32)
    lm_ids = tf_keras.Input(shape=(num_token_predictions,), dtype=tf.int32)
    inputs = {
        'input_word_ids': word_ids,
        'input_mask': mask,
        'input_type_ids': type_ids,
        'masked_lm_positions': lm_positions,
        'masked_lm_ids': lm_ids
    }

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = eletrca_trainer_model(inputs)
    lm_outs = outputs['lm_outputs']
    cls_outs = outputs['sentence_outputs']
    disc_logits = outputs['disc_logits']
    disc_label = outputs['disc_label']

    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [None, num_token_predictions, vocab_size]
    expected_classification_shape = [None, num_classes]
    expected_disc_logits_shape = [None, sequence_length]
    expected_disc_label_shape = [None, sequence_length]
    self.assertAllEqual(expected_lm_shape, lm_outs.shape.as_list())
    self.assertAllEqual(expected_classification_shape, cls_outs.shape.as_list())
    self.assertAllEqual(expected_disc_logits_shape, disc_logits.shape.as_list())
    self.assertAllEqual(expected_disc_label_shape, disc_label.shape.as_list())

  def test_electra_trainer_tensor_call(self):
    """Validate that the Keras object can be invoked."""
    # Build a transformer network to use within the ELECTRA trainer. (Here, we
    # use a short sequence_length for convenience.)
    test_generator_network = networks.BertEncoder(
        vocab_size=100, num_layers=4, max_sequence_length=3, dict_outputs=True)
    test_discriminator_network = networks.BertEncoder(
        vocab_size=100, num_layers=4, max_sequence_length=3, dict_outputs=True)

    # Create a ELECTRA trainer with the created network.
    eletrca_trainer_model = electra_pretrainer.ElectraPretrainer(
        generator_network=test_generator_network,
        discriminator_network=test_discriminator_network,
        vocab_size=100,
        num_classes=2,
        num_token_predictions=2)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_ids = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.int32)
    mask = tf.constant([[1, 1, 1], [1, 0, 0]], dtype=tf.int32)
    type_ids = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.int32)
    lm_positions = tf.constant([[0, 1], [0, 2]], dtype=tf.int32)
    lm_ids = tf.constant([[10, 20], [20, 30]], dtype=tf.int32)
    inputs = {
        'input_word_ids': word_ids,
        'input_mask': mask,
        'input_type_ids': type_ids,
        'masked_lm_positions': lm_positions,
        'masked_lm_ids': lm_ids
    }

    # Invoke the trainer model on the tensors. In Eager mode, this does the
    # actual calculation. (We can't validate the outputs, since the network is
    # too complex: this simply ensures we're not hitting runtime errors.)
    _ = eletrca_trainer_model(inputs)

  def test_serialize_deserialize(self):
    """Validate that the ELECTRA trainer can be serialized and deserialized."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_generator_network = networks.BertEncoder(
        vocab_size=100, num_layers=4, max_sequence_length=3)
    test_discriminator_network = networks.BertEncoder(
        vocab_size=100, num_layers=4, max_sequence_length=3)

    # Create a ELECTRA trainer with the created network. (Note that all the args
    # are different, so we can catch any serialization mismatches.)
    electra_trainer_model = electra_pretrainer.ElectraPretrainer(
        generator_network=test_generator_network,
        discriminator_network=test_discriminator_network,
        vocab_size=100,
        num_classes=2,
        num_token_predictions=2)

    # Create another BERT trainer via serialization and deserialization.
    config = electra_trainer_model.get_config()
    new_electra_trainer_model = electra_pretrainer.ElectraPretrainer.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_electra_trainer_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(electra_trainer_model.get_config(),
                        new_electra_trainer_model.get_config())


if __name__ == '__main__':
  tf.test.main()
