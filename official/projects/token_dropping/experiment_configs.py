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

"""Token dropping BERT experiment configurations.

Only pretraining configs. Token dropping BERT's checkpoints can be used directly
for the regular BERT. So you can just use the regular BERT for finetuning.
"""
# pylint: disable=g-doc-return-or-yield,line-too-long
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.projects.token_dropping import encoder_config
from official.projects.token_dropping import masked_lm


@exp_factory.register_config_factory('token_drop_bert/pretraining')
def token_drop_bert_pretraining() -> cfg.ExperimentConfig:
  """BERT pretraining with token dropping."""
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=masked_lm.TokenDropMaskedLMConfig(
          model=bert.PretrainerConfig(
              encoder=encoders.EncoderConfig(
                  any=encoder_config.TokenDropBertEncoderConfig(
                      vocab_size=30522, num_layers=1, token_keep_k=64),
                  type='any')),
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=cfg.TrainerConfig(
          train_steps=1000000,
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
                      'initial_learning_rate': 1e-4,
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
