# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for TEAMS pre trainer network."""

import tensorflow as tf

from official.modeling import activations
from official.nlp.modeling.networks import encoder_scaffold
from official.nlp.modeling.networks import packed_sequence_embedding
from official.projects.teams import teams_pretrainer


class TeamsPretrainerTest(tf.test.TestCase):

  # Build a transformer network to use within the TEAMS trainer.
  def _get_network(self, vocab_size):
    sequence_length = 512
    hidden_size = 50
    embedding_cfg = {
        'vocab_size': vocab_size,
        'type_vocab_size': 1,
        'hidden_size': hidden_size,
        'embedding_width': hidden_size,
        'max_seq_length': sequence_length,
        'initializer': tf.keras.initializers.TruncatedNormal(stddev=0.02),
        'dropout_rate': 0.1,
    }
    embedding_inst = packed_sequence_embedding.PackedSequenceEmbedding(
        **embedding_cfg)
    hidden_cfg = {
        'num_attention_heads':
            2,
        'intermediate_size':
            3072,
        'intermediate_activation':
            activations.gelu,
        'dropout_rate':
            0.1,
        'attention_dropout_rate':
            0.1,
        'kernel_initializer':
            tf.keras.initializers.TruncatedNormal(stddev=0.02),
    }
    return encoder_scaffold.EncoderScaffold(
        num_hidden_instances=2,
        pooled_output_dim=hidden_size,
        embedding_cfg=embedding_cfg,
        embedding_cls=embedding_inst,
        hidden_cfg=hidden_cfg,
        dict_outputs=True)

  def test_teams_pretrainer(self):
    """Validate that the Keras object can be created."""
    vocab_size = 100
    test_generator_network = self._get_network(vocab_size)
    test_discriminator_network = self._get_network(vocab_size)

    # Create a TEAMS trainer with the created network.
    candidate_size = 3
    teams_trainer_model = teams_pretrainer.TeamsPretrainer(
        generator_network=test_generator_network,
        discriminator_mws_network=test_discriminator_network,
        num_discriminator_task_agnostic_layers=1,
        vocab_size=vocab_size,
        candidate_size=candidate_size)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    num_token_predictions = 2
    sequence_length = 128
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    lm_positions = tf.keras.Input(
        shape=(num_token_predictions,), dtype=tf.int32)
    lm_ids = tf.keras.Input(shape=(num_token_predictions,), dtype=tf.int32)
    inputs = {
        'input_word_ids': word_ids,
        'input_mask': mask,
        'input_type_ids': type_ids,
        'masked_lm_positions': lm_positions,
        'masked_lm_ids': lm_ids
    }

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = teams_trainer_model(inputs)
    lm_outs = outputs['lm_outputs']
    disc_rtd_logits = outputs['disc_rtd_logits']
    disc_rtd_label = outputs['disc_rtd_label']
    disc_mws_logits = outputs['disc_mws_logits']
    disc_mws_label = outputs['disc_mws_label']

    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [None, num_token_predictions, vocab_size]
    expected_disc_rtd_logits_shape = [None, sequence_length]
    expected_disc_rtd_label_shape = [None, sequence_length]
    expected_disc_disc_mws_logits_shape = [
        None, num_token_predictions, candidate_size
    ]
    expected_disc_disc_mws_label_shape = [None, num_token_predictions]
    self.assertAllEqual(expected_lm_shape, lm_outs.shape.as_list())
    self.assertAllEqual(expected_disc_rtd_logits_shape,
                        disc_rtd_logits.shape.as_list())
    self.assertAllEqual(expected_disc_rtd_label_shape,
                        disc_rtd_label.shape.as_list())
    self.assertAllEqual(expected_disc_disc_mws_logits_shape,
                        disc_mws_logits.shape.as_list())
    self.assertAllEqual(expected_disc_disc_mws_label_shape,
                        disc_mws_label.shape.as_list())

  def test_teams_trainer_tensor_call(self):
    """Validate that the Keras object can be invoked."""
    vocab_size = 100
    test_generator_network = self._get_network(vocab_size)
    test_discriminator_network = self._get_network(vocab_size)

    # Create a TEAMS trainer with the created network.
    teams_trainer_model = teams_pretrainer.TeamsPretrainer(
        generator_network=test_generator_network,
        discriminator_mws_network=test_discriminator_network,
        num_discriminator_task_agnostic_layers=2,
        vocab_size=vocab_size,
        candidate_size=2)

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
    _ = teams_trainer_model(inputs)

  def test_serialize_deserialize(self):
    """Validate that the TEAMS trainer can be serialized and deserialized."""
    vocab_size = 100
    test_generator_network = self._get_network(vocab_size)
    test_discriminator_network = self._get_network(vocab_size)

    # Create a TEAMS trainer with the created network. (Note that all the args
    # are different, so we can catch any serialization mismatches.)
    teams_trainer_model = teams_pretrainer.TeamsPretrainer(
        generator_network=test_generator_network,
        discriminator_mws_network=test_discriminator_network,
        num_discriminator_task_agnostic_layers=2,
        vocab_size=vocab_size,
        candidate_size=2)

    # Create another TEAMS trainer via serialization and deserialization.
    config = teams_trainer_model.get_config()
    new_teams_trainer_model = teams_pretrainer.TeamsPretrainer.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_teams_trainer_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(teams_trainer_model.get_config(),
                        new_teams_trainer_model.get_config())


if __name__ == '__main__':
  tf.test.main()
