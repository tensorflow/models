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

"""FFFNER experiment configurations."""
# pylint: disable=g-doc-return-or-yield,line-too-long
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.configs import encoders
from official.projects.fffner import fffner
from official.projects.fffner import fffner_dataloader
from official.projects.fffner import fffner_prediction

AdamWeightDecay = optimization.AdamWeightDecayConfig
PolynomialLr = optimization.PolynomialLrConfig
PolynomialWarmupConfig = optimization.PolynomialWarmupConfig


@exp_factory.register_config_factory('fffner/ner')
def fffner_ner() -> cfg.ExperimentConfig:
  """Defines fffner experiments."""
  config = cfg.ExperimentConfig(
      task=fffner_prediction.FFFNerPredictionConfig(
          model=fffner_prediction.FFFNerModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=fffner.FFFNerEncoderConfig())),
          train_data=fffner_dataloader.FFFNerDataConfig(),
          validation_data=fffner_dataloader.FFFNerDataConfig(
              is_training=False, drop_remainder=False,
              include_example_id=True)),
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
                      'initial_learning_rate': 2e-5,
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
