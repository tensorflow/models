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

"""Pixel configurations."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.projects.pixel.data_loader import PixelDataConfig
from official.projects.pixel.tasks.classification import PixelConfig
from official.projects.pixel.tasks.classification import PixelModelConfig


@exp_factory.register_config_factory('pixel_sst2_finetune')
def pixel_sst2_finetune() -> cfg.ExperimentConfig:
  """Config to get results that matches https://github.com/xplip/pixel for sst2."""
  train_batch_size = 256
  eval_batch_size = 32
  num_train_steps = 15000

  input_size = (16, 4096)
  patch_h, patch_w = 16, 16
  num_channels = 3
  num_classes = 2

  config = cfg.ExperimentConfig(
      task=PixelConfig(
          train_data=PixelDataConfig(
              input_path=None,
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=10000,
              drop_remainder=True,
              input_size=input_size,
              patch_h=patch_h,
              patch_w=patch_w,
              num_channels=num_channels,
          ),
          validation_data=PixelDataConfig(
              input_path=None,
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=10000,
              drop_remainder=True,
              input_size=input_size,
              patch_h=patch_h,
              patch_w=patch_w,
              num_channels=num_channels,
          ),
          model=PixelModelConfig(
              filters=768,
              num_layers=12,
              mlp_dim=3072,
              num_heads=12,
              dropout_rate=0.1,
              attention_dropout_rate=0.1,
              init_stochastic_depth_rate=0.0,
          ),
          init_checkpoint=None,
          input_size=input_size,
          patch_h=patch_h,
          patch_w=patch_w,
          num_channels=num_channels,
          num_classes=num_classes,
      ),
      trainer=cfg.TrainerConfig(
          train_steps=num_train_steps,
          validation_steps=27,
          steps_per_loop=100,
          summary_interval=100,
          checkpoint_interval=100,
          validation_interval=100,
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'cycle': False,
                  'polynomial': {
                      'decay_steps': num_train_steps,
                      'end_learning_rate': 0.0,
                      'initial_learning_rate': 3.0e-05,
                      'power': 1.0,
                  },
              },
              'warmup': {
                  'type': 'polynomial',
                  'polynomial': {
                      'warmup_steps': 100,
                      'power': 1.0,
                  },
              },
          }),
      ),
      restrictions=[
          'task.train_data.is_training != None',
      ],
  )
  return config
