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

"""Tests for mobilebert_edgetpu.model_builder.py."""

import tensorflow as tf, tf_keras

from official.nlp import modeling
from official.nlp.configs import encoders
from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import model_builder


class ModelBuilderTest(tf.test.TestCase):

  def setUp(self):
    super(ModelBuilderTest, self).setUp()
    self.pretrainer_config = params.PretrainerModelParams(
        encoder=encoders.EncoderConfig(type='mobilebert'))

  def test_default_initialization(self):
    """Initializes pretrainer model from stratch."""
    pretrainer = model_builder.build_bert_pretrainer(
        pretrainer_cfg=self.pretrainer_config,
        name='test_model')
    # Makes sure the pretrainer variables are created.
    _ = pretrainer(pretrainer.inputs)
    self.assertEqual(pretrainer.name, 'test_model')
    encoder = pretrainer.encoder_network
    default_number_layer = encoders.MobileBertEncoderConfig().num_blocks
    encoder_transformer_layer_counter = 0
    for layer in encoder.layers:
      if isinstance(layer, modeling.layers.MobileBertTransformer):
        encoder_transformer_layer_counter += 1
    self.assertEqual(default_number_layer, encoder_transformer_layer_counter)

  def test_initialization_with_encoder(self):
    """Initializes pretrainer model with an existing encoder network."""
    encoder = encoders.build_encoder(
        config=encoders.EncoderConfig(type='mobilebert'))
    pretrainer = model_builder.build_bert_pretrainer(
        pretrainer_cfg=self.pretrainer_config,
        encoder=encoder)
    encoder_network = pretrainer.encoder_network
    self.assertEqual(encoder_network, encoder)

  def test_initialization_with_mlm(self):
    """Initializes pretrainer model with an existing MLM head."""
    embedding = modeling.layers.MobileBertEmbedding(
        word_vocab_size=30522,
        word_embed_size=128,
        type_vocab_size=2,
        output_embed_size=encoders.MobileBertEncoderConfig().hidden_size)
    dummy_input = tf_keras.layers.Input(
        shape=(None,), dtype=tf.int32)
    _ = embedding(dummy_input)
    embedding_table = embedding.word_embedding.embeddings
    mlm_layer = modeling.layers.MobileBertMaskedLM(
        embedding_table=embedding_table)
    pretrainer = model_builder.build_bert_pretrainer(
        pretrainer_cfg=self.pretrainer_config,
        masked_lm=mlm_layer)
    mlm_network = pretrainer.masked_lm
    self.assertEqual(mlm_network, mlm_layer)


if __name__ == '__main__':
  tf.test.main()
