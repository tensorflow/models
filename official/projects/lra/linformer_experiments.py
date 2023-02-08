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
"""Linformer experiments."""
# pylint: disable=g-doc-return-or-yield,line-too-long

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.configs import encoders
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.tasks import sentence_prediction
from official.projects.lra import lra_dual_encoder_dataloader
from official.projects.lra import lra_dual_encoder_task
from official.projects.lra.linformer import LinformerEncoderConfig


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
                'initial_learning_rate': 3e-5,
                'end_learning_rate': 0.0,
            },
        },
        'warmup': {'type': 'polynomial'},
    })
)


@exp_factory.register_config_factory('linformer/lra_listops')
def linformer_listops() -> cfg.ExperimentConfig:
  """Linformer lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=LinformerEncoderConfig()
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


@exp_factory.register_config_factory('linformer/lra_imdb')
def linformer_imdb() -> cfg.ExperimentConfig:
  """Linformer lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=LinformerEncoderConfig()
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


@exp_factory.register_config_factory('linformer/lra_cifar')
def linformer_cifar() -> cfg.ExperimentConfig:
  """Linformer lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=LinformerEncoderConfig()
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


@exp_factory.register_config_factory('linformer/lra_pathfinder')
def linformer_pathfinder() -> cfg.ExperimentConfig:
  """Linformer lra fine-tuning."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=LinformerEncoderConfig()
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


@exp_factory.register_config_factory('linformer/lra_aan')
def linformer_aan() -> cfg.ExperimentConfig:
  """Linformer LRA Task."""
  config = cfg.ExperimentConfig(
      task=lra_dual_encoder_task.DualEncoderConfig(
          model=lra_dual_encoder_task.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type='any', any=LinformerEncoderConfig()
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
