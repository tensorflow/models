# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.tasks.masked_lm."""

import tensorflow as tf, tf_keras

from official.nlp.data import pretrain_dataloader
from official.nlp.data import sentence_prediction_dataloader
from official.projects.perceiver.configs import perceiver


class PerceiverWordPiecePretrainConfigTest(tf.test.TestCase):

  def test_word_piece_pretrain_config(self):
    config = perceiver.PretrainConfig(
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            global_batch_size=512,
            use_next_sentence_label=False,
            use_v2_feature_names=True),
        validation_data=pretrain_dataloader.BertPretrainDataConfig(
            global_batch_size=512,
            is_training=False,
            use_next_sentence_label=False,
            use_v2_feature_names=True))
    self.assertIsNotNone(config)
    self.assertIsNotNone(config.model)
    self.assertFalse(config.scale_loss)


class PerceiverWordPieceSentencePredictionConfigTest(tf.test.TestCase):

  def test_word_piece_fine_tune_config(self):
    config = perceiver.SentencePredictionConfig(
        train_data=sentence_prediction_dataloader
        .SentencePredictionDataConfig(),
        validation_data=sentence_prediction_dataloader
        .SentencePredictionDataConfig())
    self.assertIsNotNone(config)
    self.assertIsNotNone(config.model)
    self.assertFalse(config.init_cls_pooler)

  def test_perceiver_sentence_prediction_returns_valid_learning_rate(self):
    experiment_cfg = perceiver.perceiver_word_piece_sentence_prediction()
    self.assertIsNotNone(experiment_cfg.trainer.optimizer_config.learning_rate)


class PerceiverWordPieceRawSentencePredictionConfigTest(tf.test.TestCase):

  def test_word_piece_raw_sentence_fine_tune_config(self):
    config = perceiver.SentencePredictionConfig(
        train_data=sentence_prediction_dataloader
        .SentencePredictionTextDataConfig(),
        validation_data=sentence_prediction_dataloader
        .SentencePredictionTextDataConfig())
    self.assertIsNotNone(config)
    self.assertIsNotNone(config.model)
    self.assertFalse(config.init_cls_pooler)

  def test_perceiver_raw_sentence_prediction_returns_valid_learning_rate(self):
    experiment_cfg = perceiver.perceiver_word_piece_raw_sentence_prediction()
    self.assertIsNotNone(experiment_cfg.trainer.optimizer_config.learning_rate)


if __name__ == "__main__":
  tf.test.main()
