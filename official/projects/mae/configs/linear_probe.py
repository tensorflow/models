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

"""ViT linear probing configurations."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.projects.mae.tasks import linear_probe
from official.vision.configs import image_classification


@exp_factory.register_config_factory('vit_imagenet_mae_linear_probe')
def vit_imagenet_mae_linear_probe() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 16384
  eval_batch_size = 1024
  imagenet_size = 1281167
  steps_per_epoch = imagenet_size // train_batch_size
  config = cfg.ExperimentConfig(
      task=linear_probe.ViTLinearProbeConfig(  # pylint: disable=unexpected-keyword-arg
          train_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='train',
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=20000,
          ),
          validation_data=image_classification.DataConfig(
              tfds_name='imagenet2012',
              tfds_split='validation',
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
              aug_rand_hflip=False,
          ),
          init_stochastic_depth_rate=0.0,
          init_checkpoint='Please provide',
      ),
      trainer=cfg.TrainerConfig(
          train_steps=90 * steps_per_epoch,
          validation_steps=48,
          steps_per_loop=100,
          summary_interval=100,
          checkpoint_interval=100,
          validation_interval=100,
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'lars',
                  'lars': {
                      'weight_decay_rate': 0.0,
                      'momentum': 0.9,
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.1 * train_batch_size / 256,
                      'decay_steps': 90 * steps_per_epoch,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 10 * steps_per_epoch,
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
