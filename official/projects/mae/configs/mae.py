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

"""MAE configurations."""

import dataclasses
from typing import Tuple

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.configs import image_classification


@dataclasses.dataclass
class MAEConfig(cfg.TaskConfig):
  """The translation task config."""
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )
  masking_ratio: float = 0.75
  patch_h: int = 14
  patch_w: int = 14
  num_classes: int = 1000
  input_size: Tuple[int, int] = (224, 224)
  norm_target: bool = False


@exp_factory.register_config_factory('mae_imagenet')
def mae_imagenet() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 4096
  eval_batch_size = 4096
  imagenet_size = 1281167
  steps_per_epoch = imagenet_size // train_batch_size
  config = cfg.ExperimentConfig(
      task=MAEConfig(
          train_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='train',
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=10000,
              crop_area_range=(0.2, 1.0),
          ),
          validation_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='validation',
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
          )
      ),
      trainer=cfg.TrainerConfig(
          train_steps=800 * steps_per_epoch,
          validation_steps=24,
          steps_per_loop=1000,
          summary_interval=1000,
          checkpoint_interval=1000,
          validation_interval=1000,
          max_to_keep=5,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'beta_2': 0.95,
                      'weight_decay_rate': 0.05,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm':
                          0.0,
                      'exclude_from_weight_decay': [
                          'LayerNorm', 'layer_norm', 'bias']
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate':
                          1.5 * 1e-4 * train_batch_size / 256,
                      'decay_steps': 800 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 40 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
              })
          ),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
