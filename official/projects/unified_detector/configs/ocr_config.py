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

"""OCR tasks and models configurations."""

import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization


@dataclasses.dataclass
class OcrTaskConfig(cfg.TaskConfig):
  train_data: cfg.DataConfig = cfg.DataConfig()
  model_call_needs_labels: bool = False


@exp_factory.register_config_factory('unified_detector')
def unified_detector() -> cfg.ExperimentConfig:
  """Configurations for trainer of unified detector."""
  total_train_steps = 100000
  summary_interval = steps_per_loop = 200
  checkpoint_interval = 2000
  warmup_steps = 1000
  config = cfg.ExperimentConfig(
      # Input pipeline and model are configured through Gin.
      task=OcrTaskConfig(train_data=cfg.DataConfig(is_training=True)),
      trainer=cfg.TrainerConfig(
          train_steps=total_train_steps,
          steps_per_loop=steps_per_loop,
          summary_interval=summary_interval,
          checkpoint_interval=checkpoint_interval,
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate': 0.05,
                      'include_in_weight_decay': [
                          '^((?!depthwise).)*(kernel|weights):0$',
                      ],
                      'exclude_from_weight_decay': [
                          '(^((?!kernel).)*:0)|(depthwise_kernel)',
                      ],
                      'gradient_clip_norm': 10.,
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 1e-3,
                      'decay_steps': total_train_steps - warmup_steps,
                      'alpha': 1e-2,
                      'offset': warmup_steps,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_learning_rate': 1e-5,
                      'warmup_steps': warmup_steps,
                  }
              },
          }),
      ),
  )
  return config
