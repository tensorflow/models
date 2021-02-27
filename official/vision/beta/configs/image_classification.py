# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Image classification configuration definition."""
import os
from typing import List, Optional
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  aug_policy: Optional[str] = None  # None, 'autoaug', or 'randaug'
  randaug_magnitude: Optional[int] = 10
  file_type: str = 'tfrecord'


@dataclasses.dataclass
class ImageClassificationModel(hyperparams.Config):
  """The model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  dropout_rate: float = 0.0
  norm_activation: common.NormActivation = common.NormActivation(
      use_sync_bn=False)
  # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
  add_head_batch_norm: bool = False


@dataclasses.dataclass
class Losses(hyperparams.Config):
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5


@dataclasses.dataclass
class ImageClassificationTask(cfg.TaskConfig):
  """The task config."""
  model: ImageClassificationModel = ImageClassificationModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone


@exp_factory.register_config_factory('image_classification')
def image_classification() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=ImageClassificationTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


@exp_factory.register_config_factory('resnet_imagenet')
def image_classification_imagenet() -> cfg.ExperimentConfig:
  """Image classification on imagenet with resnet."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [
                          0.1 * train_batch_size / 256,
                          0.01 * train_batch_size / 256,
                          0.001 * train_batch_size / 256,
                          0.0001 * train_batch_size / 256,
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('resnet_rs_imagenet')
def image_classification_imagenet_resnetrs() -> cfg.ExperimentConfig:
  """Image classification on imagenet with resnet-rs."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[160, 160, 3],
              backbone=backbones.Backbone(
                  type='resnet',
                  resnet=backbones.ResNet(
                      model_id=50,
                      stem_type='v1',
                      resnetd_shortcut=True,
                      replace_stem_max_pool=True,
                      se_ratio=0.25,
                      stochastic_depth_drop_rate=0.0)),
              dropout_rate=0.25,
              norm_activation=common.NormActivation(
                  norm_momentum=0.0,
                  norm_epsilon=1e-5,
                  use_sync_bn=False,
                  activation='swish')),
          losses=Losses(l2_weight_decay=4e-5, label_smoothing=0.1),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_policy='randaug',
              randaug_magnitude=10),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=350 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'ema': {
                  'average_decay': 0.9999
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 1.6,
                      'decay_steps': 350 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('revnet_imagenet')
def image_classification_imagenet_revnet() -> cfg.ExperimentConfig:
  """Returns a revnet config for image classification on imagenet."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size

  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='revnet', revnet=backbones.RevNet(model_id=56)),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False),
              add_head_batch_norm=True),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [0.8, 0.08, 0.008, 0.0008]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('mobilenet_imagenet')
def image_classification_imagenet_mobilenet() -> cfg.ExperimentConfig:
  """Image classification on imagenet with mobilenet."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              dropout_rate=0.2,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='mobilenet',
                  mobilenet=backbones.MobileNet(
                      model_id='MobileNetV2', filter_size_scale=1.0)),
              norm_activation=common.NormActivation(
                  norm_momentum=0.997, norm_epsilon=1e-3, use_sync_bn=False)),
          losses=Losses(l2_weight_decay=1e-5, label_smoothing=0.1),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=500 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'rmsprop',
                  'rmsprop': {
                      'rho': 0.9,
                      'momentum': 0.9,
                      'epsilon': 0.002,
                  }
              },
              'learning_rate': {
                  'type': 'exponential',
                  'exponential': {
                      'initial_learning_rate':
                          0.008 * (train_batch_size // 128),
                      'decay_steps':
                          int(2.5 * steps_per_epoch),
                      'decay_rate':
                          0.98,
                      'staircase':
                          True
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              },
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
