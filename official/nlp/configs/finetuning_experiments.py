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

"""Finetuning experiment configurations."""
# pylint: disable=g-doc-return-or-yield,line-too-long
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.data import question_answering_dataloader
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.data import tagging_dataloader
from official.nlp.tasks import question_answering
from official.nlp.tasks import sentence_prediction
from official.nlp.tasks import tagging


@exp_factory.register_config_factory('bert/sentence_prediction')
def bert_sentence_prediction() -> cfg.ExperimentConfig:
  r"""BERT GLUE."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          train_data=sentence_prediction_dataloader
          .SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader
          .SentencePredictionDataConfig(
              is_training=False, drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 3e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('bert/sentence_prediction_text')
def bert_sentence_prediction_text() -> cfg.ExperimentConfig:
  r"""BERT sentence prediction with raw text data.

  Example: use tf.text and tfds as input with glue_mnli_text.yaml
  """
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          train_data=sentence_prediction_dataloader
          .SentencePredictionTextDataConfig(),
          validation_data=sentence_prediction_dataloader
          .SentencePredictionTextDataConfig(
              is_training=False, drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 3e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('bert/squad')
def bert_squad() -> cfg.ExperimentConfig:
  """BERT Squad V1/V2."""
  config = cfg.ExperimentConfig(
      task=question_answering.QuestionAnsweringConfig(
          train_data=question_answering_dataloader.QADataConfig(),
          validation_data=question_answering_dataloader.QADataConfig()),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 8e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('bert/tagging')
def bert_tagging() -> cfg.ExperimentConfig:
  """BERT tagging task."""
  config = cfg.ExperimentConfig(
      task=tagging.TaggingConfig(
          train_data=tagging_dataloader.TaggingDataConfig(),
          validation_data=tagging_dataloader.TaggingDataConfig(
              is_training=False, drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 8e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])
  return config
