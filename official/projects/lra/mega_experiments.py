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

"""Mega experiments."""
# pylint: disable=g-doc-return-or-yield,line-too-long
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.configs import encoders
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.tasks import sentence_prediction
from official.projects.lra import lra_dual_encoder_dataloader
from official.projects.lra import lra_dual_encoder_task
from official.projects.lra.mega import MegaEncoderConfig

AdamWeightDecay = optimization.AdamWeightDecayConfig
PolynomialLr = optimization.PolynomialLrConfig
PolynomialWarmupConfig = optimization.PolynomialWarmupConfig

_TRAINER = cfg.TrainerConfig(
    optimizer_config=optimization.OptimizationConfig({
        'optimizer': {
            'type': 'adamw',
            'adamw': {
                'weight_decay_rate': 0.01,
                'exclude_from_weight_decay': [
                    'LayerNorm',
                    'layer_norm',
                    'bias',
                ],
            },
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {
                'initial_learning_rate': 1e-7,
                'end_learning_rate': 0.0,
            },
        },
        'warmup': {'type': 'polynomial'},
    })
)


@exp_factory.register_config_factory('mega/lra_listops')
def mega_listops() -> cfg.ExperimentConfig:
  """Mega lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=MegaEncoderConfig()
              )
          ),
          train_data=sentence_prediction_dataloader.SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader.SentencePredictionDataConfig(
              is_training=False, drop_remainder=False
          ),
      ),
      trainer=_TRAINER,
  )
  return config


@exp_factory.register_config_factory('mega/lra_imdb')
def mega_imdb() -> cfg.ExperimentConfig:
  """Mega lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=MegaEncoderConfig()
              )
          ),
          train_data=sentence_prediction_dataloader.SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader.SentencePredictionDataConfig(
              is_training=False, drop_remainder=False
          ),
      ),
      trainer=_TRAINER,
  )
  return config


@exp_factory.register_config_factory('mega/lra_cifar')
def mega_cifar() -> cfg.ExperimentConfig:
  """Mega lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=MegaEncoderConfig()
              )
          ),
          train_data=sentence_prediction_dataloader.SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader.SentencePredictionDataConfig(
              is_training=False, drop_remainder=False
          ),
      ),
      trainer=_TRAINER,
  )
  return config


@exp_factory.register_config_factory('mega/lra_pathfinder')
def mega_pathfinder() -> cfg.ExperimentConfig:
  """Mega lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=MegaEncoderConfig()
              )
          ),
          train_data=sentence_prediction_dataloader.SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader.SentencePredictionDataConfig(
              is_training=False, drop_remainder=False
          ),
      ),
      trainer=_TRAINER,
  )
  return config


@exp_factory.register_config_factory('mega/lra_aan')
def mega_aan() -> cfg.ExperimentConfig:
  """Mega LRA task."""
  config = cfg.ExperimentConfig(
      task=lra_dual_encoder_task.DualEncoderConfig(
          model=lra_dual_encoder_task.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=MegaEncoderConfig()
              )
          ),
          train_data=lra_dual_encoder_dataloader.DualEncoderDataConfig(),
          validation_data=lra_dual_encoder_dataloader.DualEncoderDataConfig(
              is_training=False, drop_remainder=False
          ),
      ),
      trainer=_TRAINER,
  )
  return config
