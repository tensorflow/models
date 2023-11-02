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

"""Tests for classifier."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.nlp.modeling import layers
from official.projects.perceiver.configs import encoders
from official.projects.perceiver.configs import perceiver as cfg
from official.projects.perceiver.modeling.layers import decoder
from official.projects.perceiver.modeling.models import classifier
from official.projects.perceiver.modeling.networks import positional_decoder


class ClassifierTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('single_cls', 1), ('3_cls', 3))
  def test_perceiver_trainer(self, num_classes):
    """Validate that the Keras object can be created."""
    # Build a perceiver sequence encoder network to use within the perceiver
    # trainer.

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

    deocder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents)
    perceiver_classification_decoder_cfg = cfg.ClassificationDecoderConfig(
        d_model=d_model,
        decoder=deocder_cfg,
        d_latents=d_latents)
    decoder_ = decoder.Decoder(
        **perceiver_classification_decoder_cfg.decoder.as_dict())
    positional_decoder_ = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=perceiver_classification_decoder_cfg.output_index_dim,
        z_index_dim=perceiver_classification_decoder_cfg.z_index_dim,
        d_latents=perceiver_classification_decoder_cfg.d_latents,
        d_model=perceiver_classification_decoder_cfg.d_model,
        position_encoding_intializer_stddev=perceiver_classification_decoder_cfg
        .position_encoding_intializer_stddev)

    # Create a classifier with the created network.
    trainer_model = classifier.Classifier(
        network=test_network,
        decoder=positional_decoder_,
        num_classes=num_classes)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    cls_outs = trainer_model({
        'input_word_ids': word_ids,
        'input_mask': mask,
        'input_type_ids': type_ids})

    # Validate that the outputs are of the expected shape.
    expected_classification_shape = [None, num_classes]
    self.assertAllEqual(expected_classification_shape, cls_outs.shape.as_list())

  @parameterized.named_parameters(
      ('single_cls', 1, False),
      ('2_cls', 2, False),
      ('single_cls_custom_head', 1, True),
      ('2_cls_custom_head', 2, True))
  def test_perceiver_trainer_tensor_call(self, num_classes, use_custom_head):
    """Validate that the Keras object can be invoked."""
    # Build a perceiver sequence encoder network to use within the perceiver
    # trainer.
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

    deocder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents)
    perceiver_classification_decoder_cfg = cfg.ClassificationDecoderConfig(
        d_model=d_model,
        decoder=deocder_cfg,
        d_latents=d_latents)
    decoder_ = decoder.Decoder(
        **perceiver_classification_decoder_cfg.decoder.as_dict())
    positional_decoder_ = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=perceiver_classification_decoder_cfg.output_index_dim,
        z_index_dim=perceiver_classification_decoder_cfg.z_index_dim,
        d_latents=perceiver_classification_decoder_cfg.d_latents,
        d_model=perceiver_classification_decoder_cfg.d_model,
        position_encoding_intializer_stddev=perceiver_classification_decoder_cfg
        .position_encoding_intializer_stddev)

    cls_head = layers.GaussianProcessClassificationHead(
        inner_dim=0, num_classes=num_classes) if use_custom_head else None

    # Create a classifier with the created network.
    trainer_model = classifier.Classifier(
        network=test_network,
        decoder=positional_decoder_,
        cls_head=cls_head,
        num_classes=num_classes)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
    mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
    type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)

    # Invoke the trainer model on the tensors. In Eager mode, this does the
    # actual calculation. (We can't validate the outputs, since the network is
    # too complex: this simply ensures we're not hitting runtime errors.)
    _ = trainer_model({
        'input_word_ids': word_ids,
        'input_mask': mask,
        'input_type_ids': type_ids})

  @parameterized.named_parameters(
      ('default_cls_head', None),
      ('sngp_cls_head', layers.GaussianProcessClassificationHead(
          inner_dim=0, num_classes=4)))
  def test_serialize_deserialize(self, cls_head):
    """Validate that the trainer can be serialized and deserialized."""
    del cls_head
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

    deocder_cfg = cfg.DecoderConfig(
        output_last_dim=d_latents,
        v_last_dim=d_latents)
    perceiver_classification_decoder_cfg = cfg.ClassificationDecoderConfig(
        d_model=d_model,
        decoder=deocder_cfg,
        d_latents=d_latents)
    decoder_ = decoder.Decoder(
        **perceiver_classification_decoder_cfg.decoder.as_dict())
    positional_decoder_ = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=perceiver_classification_decoder_cfg.output_index_dim,
        z_index_dim=perceiver_classification_decoder_cfg.z_index_dim,
        d_latents=perceiver_classification_decoder_cfg.d_latents,
        d_model=perceiver_classification_decoder_cfg.d_model,
        position_encoding_intializer_stddev=perceiver_classification_decoder_cfg
        .position_encoding_intializer_stddev)

    # Create a classifier with the created network.
    trainer_model = classifier.Classifier(
        network=test_network,
        decoder=positional_decoder_,
        num_classes=4)

    # Create another trainer via serialization and deserialization.
    config = trainer_model.get_config()
    new_trainer_model = classifier.Classifier.from_config(config)

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(trainer_model.get_config(),
                        new_trainer_model.get_config())

# TODO(b/222634115) add test coverage.

if __name__ == '__main__':
  tf.test.main()
