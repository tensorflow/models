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

"""ViT configurations."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.mae import optimization
from official.projects.mae.tasks import image_classification as vit
from official.vision.configs import common
from official.vision.configs import image_classification


vars_substr = [
    'token_layer/cls', 'dense_1/kernel', 'vi_t_classifier/dense',
    'encoder/layer_normalization', 'encoder/transformer_encoder_block/'
]

layers_idx = [0, 0, 25, 24, 1]

for i in range(1, 24):
  vars_substr.append('encoder/transformer_encoder_block_%s/' % str(i))
  layers_idx.append(i + 1)


@exp_factory.register_config_factory('vit_imagenet_mae_finetune')
def vit_imagenet_mae_finetune() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 1024
  eval_batch_size = 1024
  imagenet_size = 1281167
  steps_per_epoch = imagenet_size // train_batch_size
  config = cfg.ExperimentConfig(
      task=vit.ViTConfig(
          train_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='train',
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=10000,
              aug_type=common.Augmentation(
                  type='randaug',
                  randaug=common.RandAugment(
                      magnitude=9,
                      magnitude_std=0.5,
                      exclude_ops=['Cutout', 'Invert'],
                  ),
              ),
          ),
          validation_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='validation',
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
              aug_rand_hflip=False,
          ),
          init_stochastic_depth_rate=0.1,
          init_checkpoint='Please provide',
      ),
      trainer=cfg.TrainerConfig(
          train_steps=50 * steps_per_epoch,
          validation_steps=48,
          steps_per_loop=2000,
          summary_interval=2000,
          checkpoint_interval=2000,
          validation_interval=2000,
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'vit_adamw',
                  'vit_adamw': {
                      'weight_decay_rate': 0.05,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0,
                      'beta_2': 0.999,
                      'layer_decay': 0.75,
                      'vars_substr': vars_substr,
                      'layers_idx': layers_idx,
                      'exclude_from_weight_decay': ['cls'],
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 1e-3 * train_batch_size / 256,
                      'decay_steps': 50 * steps_per_epoch,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0,
                  },
              },
          }),
      ),
      restrictions=[
          'task.train_data.is_training != None',
      ],
  )
  return config


@exp_factory.register_config_factory('vit_imagenet_scratch')
def vit_imagenet_scratch() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 4096
  eval_batch_size = 1024
  imagenet_size = 1281167
  steps_per_epoch = imagenet_size // train_batch_size
  config = cfg.ExperimentConfig(
      task=vit.ViTConfig(
          train_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='train',
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=10000,
              aug_type=common.Augmentation(
                  type='randaug',
                  randaug=common.RandAugment(
                      magnitude=9,
                      magnitude_std=0.5,
                      exclude_ops=['Cutout', 'Invert'])
              )
          ),
          validation_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='validation',
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
              aug_rand_hflip=False,
          )
      ),
      trainer=cfg.TrainerConfig(
          train_steps=200 * steps_per_epoch,
          validation_steps=48,
          steps_per_loop=1000,
          summary_interval=1000,
          checkpoint_interval=1000,
          validation_interval=1000,
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'vit_adamw',
                  'vit_adamw': {
                      'weight_decay_rate': 0.3,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0,
                      'beta_2': 0.95,
                      'exclude_from_weight_decay': ['cls']
                  }
              },
              'ema': {
                  'average_decay': 0.9999,
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate':
                          1e-4 * train_batch_size / 256,
                      'decay_steps': 200 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 20 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
              })
          ),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
