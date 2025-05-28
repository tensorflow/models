# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Pretraining experiment configurations."""
# pylint: disable=g-doc-return-or-yield,line-too-long
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.data import pretrain_dataloader
from official.nlp.data import pretrain_dynamic_dataloader
from official.nlp.data import pretrain_text_dataloader
from official.nlp.tasks import electra_task
from official.nlp.tasks import masked_lm


_TRAINER = cfg.TrainerConfig(
    train_steps=1000000,
    optimizer_config=optimization.OptimizationConfig({
        'optimizer': {
            'type': 'adamw',
            'adamw': {
                'weight_decay_rate':
                    0.01,
                'exclude_from_weight_decay': [
                    'LayerNorm', 'layer_norm', 'bias'
                ],
            }
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {
                'initial_learning_rate': 1e-4,
                'end_learning_rate': 0.0,
            }
        },
        'warmup': {
            'type': 'polynomial'
        }
    }))


@exp_factory.register_config_factory('bert/pretraining')
def bert_pretraining() -> cfg.ExperimentConfig:
  """BERT pretraining experiment."""
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=masked_lm.MaskedLMConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('bert/pretraining_dynamic')
def bert_dynamic() -> cfg.ExperimentConfig:
  """BERT base with dynamic input sequences.

  TPU needs to run with tf.data service with round-robin behavior.
  """
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=masked_lm.MaskedLMConfig(
          train_data=pretrain_dynamic_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('bert/text_wiki_pretraining')
def bert_text_wiki_pretraining() -> cfg.ExperimentConfig:
  r"""BERT with wiki text tfds.

  Note that: only wikipedia english corpus is used. It cannot exactly reproduce
  BERT training setup because the next sentence sampling is hard to match the
  implementation with tf ops.
  """
  config = cfg.ExperimentConfig(
      task=masked_lm.MaskedLMConfig(
          train_data=pretrain_text_dataloader.BertPretrainTextDataConfig(
              tfds_name='wikipedia/20201201.en',
              tfds_split='train',
              vocab_file_path='TODO for users',
          ),
          validation_data=pretrain_text_dataloader.BertPretrainTextDataConfig(
              tfds_name='wikipedia/20201201.en',
              tfds_split='train',
              vocab_file_path='TODO for users',
              is_training=False)),
      trainer=_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('electra/pretraining')
def electra_pretrain() -> cfg.ExperimentConfig:
  """ELECTRA pretraining experiment."""
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=electra_task.ElectraPretrainConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False
          ),
      ),
      trainer=_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ],
  )
  return config
