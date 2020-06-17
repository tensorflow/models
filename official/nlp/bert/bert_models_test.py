# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs
from official.nlp.modeling import networks


class BertModelsTest(tf.test.TestCase):

  def setUp(self):
    super(BertModelsTest, self).setUp()
    self._bert_test_config = bert_configs.BertConfig(
        attention_probs_dropout_prob=0.0,
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        hidden_size=16,
        initializer_range=0.02,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        type_vocab_size=2,
        vocab_size=30522)

  def test_pretrain_model(self):
    model, encoder = bert_models.pretrain_model(
        self._bert_test_config,
        seq_length=5,
        max_predictions_per_seq=2,
        initializer=None,
        use_next_sentence_label=True)
    self.assertIsInstance(model, tf.keras.Model)
    self.assertIsInstance(encoder, networks.TransformerEncoder)

    # model has one scalar output: loss value.
    self.assertEqual(model.output.shape.as_list(), [None,])

    # Expect two output from encoder: sequence and classification output.
    self.assertIsInstance(encoder.output, list)
    self.assertLen(encoder.output, 2)
    # shape should be [batch size, seq_length, hidden_size]
    self.assertEqual(encoder.output[0].shape.as_list(), [None, 5, 16])
    # shape should be [batch size, hidden_size]
    self.assertEqual(encoder.output[1].shape.as_list(), [None, 16])

  def test_squad_model(self):
    model, core_model = bert_models.squad_model(
        self._bert_test_config,
        max_seq_length=5,
        initializer=None,
        hub_module_url=None,
        hub_module_trainable=None)
    self.assertIsInstance(model, tf.keras.Model)
    self.assertIsInstance(core_model, tf.keras.Model)

    # Expect two output from model: start positions and end positions
    self.assertIsInstance(model.output, list)
    self.assertLen(model.output, 2)
    # shape should be [batch size, seq_length]
    self.assertEqual(model.output[0].shape.as_list(), [None, 5])
    # shape should be [batch size, seq_length]
    self.assertEqual(model.output[1].shape.as_list(), [None, 5])

    # Expect two output from core_model: sequence and classification output.
    self.assertIsInstance(core_model.output, list)
    self.assertLen(core_model.output, 2)
    # shape should be [batch size, seq_length, hidden_size]
    self.assertEqual(core_model.output[0].shape.as_list(), [None, 5, 16])
    # shape should be [batch size, hidden_size]
    self.assertEqual(core_model.output[1].shape.as_list(), [None, 16])

  def test_classifier_model(self):
    model, core_model = bert_models.classifier_model(
        self._bert_test_config,
        num_labels=3,
        max_seq_length=5,
        final_layer_initializer=None,
        hub_module_url=None,
        hub_module_trainable=None)
    self.assertIsInstance(model, tf.keras.Model)
    self.assertIsInstance(core_model, tf.keras.Model)

    # model has one classification output with num_labels=3.
    self.assertEqual(model.output.shape.as_list(), [None, 3])

    # Expect two output from core_model: sequence and classification output.
    self.assertIsInstance(core_model.output, list)
    self.assertLen(core_model.output, 2)
    # shape should be [batch size, 1, hidden_size]
    self.assertEqual(core_model.output[0].shape.as_list(), [None, 1, 16])
    # shape should be [batch size, hidden_size]
    self.assertEqual(core_model.output[1].shape.as_list(), [None, 16])


if __name__ == '__main__':
  tf.test.main()
