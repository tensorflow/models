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

"""Tests for Perceiver pretrainer model."""
import itertools

from absl.testing import parameterized
import tensorflow as tf

from official.nlp.modeling import layers
from official.projects.perceiver.configs import encoders
from official.projects.perceiver.configs import perceiver as cfg
from official.projects.perceiver.modeling.layers import decoder
from official.projects.perceiver.modeling.models import pretrainer
from official.projects.perceiver.modeling.networks import positional_decoder


class PretrainerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (False, True),
      (False, True),
  ))
  def test_perceiver_pretrainer(self, use_customized_masked_lm,
                                has_masked_lm_positions):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the Perceiver trainer.
    vocab_size = 100
    sequence_length = 512
    d_model = 64
    d_latents = 48
    num_layers = 2
    encoder_cfg = cfg.EncoderConfig(
        v_last_dim=d_latents,
        num_self_attends_per_block=num_layers)
    sequence_encoder_cfg = cfg.SequenceEncoderConfig(
        d_model=d_model,
        d_latents=d_latents,
        vocab_size=vocab_size,
        encoder=encoder_cfg)
    test_network = encoders.build_encoder(sequence_encoder_cfg)

    _ = test_network(test_network.inputs)

    deocder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents)
    perceiver_mlm_decoder_cfg = cfg.MaskedLMDecoderConfig(
        d_model=d_model,
        decoder=deocder_cfg,
        d_latents=d_latents)
    decoder_ = decoder.Decoder(
        **perceiver_mlm_decoder_cfg.decoder.as_dict())
    positional_decoder_ = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=perceiver_mlm_decoder_cfg.output_index_dim,
        z_index_dim=perceiver_mlm_decoder_cfg.z_index_dim,
        d_latents=perceiver_mlm_decoder_cfg.d_latents,
        d_model=perceiver_mlm_decoder_cfg.d_model,
        position_encoding_intializer_stddev=perceiver_mlm_decoder_cfg
        .position_encoding_intializer_stddev)

    if use_customized_masked_lm:
      customized_masked_lm = layers.MaskedLM(
          embedding_table=test_network.get_embedding_table())
    else:
      customized_masked_lm = None

    # Create a Perceiver trainer with the created network.
    perceiver_trainer_model = pretrainer.Pretrainer(
        encoder=test_network,
        decoder=positional_decoder_,
        customized_masked_lm=customized_masked_lm)
    num_token_predictions = 20
    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    inputs = dict(
        input_word_ids=tf.keras.Input(shape=(sequence_length,), dtype=tf.int32),
        input_mask=tf.keras.Input(shape=(sequence_length,), dtype=tf.int32),
        input_type_ids=tf.keras.Input(shape=(sequence_length,), dtype=tf.int32))
    if has_masked_lm_positions:
      inputs['masked_lm_positions'] = tf.keras.Input(
          shape=(num_token_predictions,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = perceiver_trainer_model(inputs)

    expected_keys = ['sequence_output']
    if has_masked_lm_positions:
      expected_keys.append('mlm_logits')

    self.assertSameElements(outputs.keys(), expected_keys)
    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [None, num_token_predictions, vocab_size]
    if has_masked_lm_positions:
      self.assertAllEqual(expected_lm_shape,
                          outputs['mlm_logits'].shape.as_list())

    expected_sequence_output_shape = [None, sequence_length, d_model]
    self.assertAllEqual(expected_sequence_output_shape,
                        outputs['sequence_output'].shape.as_list())

  def test_serialize_deserialize(self):
    """Validate that the trainer can be serialized and deserialized."""
    vocab_size = 100
    d_model = 64
    d_latents = 48
    num_layers = 2
    encoder_cfg = cfg.EncoderConfig(
        v_last_dim=d_latents,
        num_self_attends_per_block=num_layers)
    sequence_encoder_cfg = cfg.SequenceEncoderConfig(
        d_model=d_model,
        d_latents=d_latents,
        vocab_size=vocab_size,
        encoder=encoder_cfg)
    test_network = encoders.build_encoder(sequence_encoder_cfg)

    _ = test_network(test_network.inputs)

    deocder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents)
    perceiver_mlm_decoder_cfg = cfg.MaskedLMDecoderConfig(
        d_model=d_model,
        decoder=deocder_cfg,
        d_latents=d_latents)
    decoder_ = decoder.Decoder(
        **perceiver_mlm_decoder_cfg.decoder.as_dict())
    positional_decoder_ = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=perceiver_mlm_decoder_cfg.output_index_dim,
        z_index_dim=perceiver_mlm_decoder_cfg.z_index_dim,
        d_latents=perceiver_mlm_decoder_cfg.d_latents,
        d_model=perceiver_mlm_decoder_cfg.d_model,
        position_encoding_intializer_stddev=perceiver_mlm_decoder_cfg
        .position_encoding_intializer_stddev)

    # Create a Perceiver trainer with the created network.
    perceiver_trainer_model = pretrainer.Pretrainer(
        encoder=test_network,
        decoder=positional_decoder_)

    config = perceiver_trainer_model.get_config()
    new_perceiver_trainer_model = pretrainer.Pretrainer.from_config(config)

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(perceiver_trainer_model.get_config(),
                        new_perceiver_trainer_model.get_config())

# TODO(b/222634115) add test coverage.

if __name__ == '__main__':
  tf.test.main()
