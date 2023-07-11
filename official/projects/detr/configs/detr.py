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

"""DETR configurations."""

import dataclasses
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.projects.detr import optimization
from official.projects.detr.dataloaders import coco
from official.vision.configs import backbones
from official.vision.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  tfds_name: str = ''
  tfds_split: str = 'train'
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: common.DataDecoder = dataclasses.field(default_factory=common.DataDecoder)
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True


@dataclasses.dataclass
class Losses(hyperparams.Config):
  class_offset: int = 0
  lambda_cls: float = 1.0
  lambda_box: float = 5.0
  lambda_giou: float = 2.0
  background_cls_weight: float = 0.1
  l2_weight_decay: float = 1e-4


@dataclasses.dataclass
class Detr(hyperparams.Config):
  """Detr model definations."""
  num_queries: int = 100
  hidden_size: int = 256
  num_classes: int = 91  # 0: background
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = dataclasses.field(default_factory=lambda:backbones.Backbone(
      type='resnet', resnet=backbones.ResNet(model_id=50, bn_trainable=False)))
  norm_activation: common.NormActivation = dataclasses.field(default_factory=common.NormActivation)
  backbone_endpoint_name: str = '5'


@dataclasses.dataclass
class DetrTask(cfg.TaskConfig):
  model: Detr = dataclasses.field(default_factory=Detr)
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = 'all'  # all, backbone
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('detr_coco')
def detr_coco() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 64
  eval_batch_size = 64
  num_train_data = COCO_TRAIN_EXAMPLES
  num_steps_per_epoch = num_train_data // train_batch_size
  train_steps = 500 * num_steps_per_epoch  # 500 epochs
  decay_at = train_steps - 100 * num_steps_per_epoch  # 400 epochs
  config = cfg.ExperimentConfig(
      task=DetrTask(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          model=Detr(
              num_classes=81,
              input_size=[1333, 1333, 3],
              norm_activation=common.NormActivation()),
          losses=Losses(),
          train_data=coco.COCODataConfig(
              tfds_name='coco/2017',
              tfds_split='train',
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
          ),
          validation_data=coco.COCODataConfig(
              tfds_name='coco/2017',
              tfds_split='validation',
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=-1,
          steps_per_loop=10000,
          summary_interval=10000,
          checkpoint_interval=10000,
          validation_interval=10000,
          max_to_keep=1,
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'detr_adamw',
                  'detr_adamw': {
                      'weight_decay_rate': 1e-4,
                      'global_clipnorm': 0.1,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [decay_at],
                      'values': [0.0001, 1.0e-05]
                  }
              },
          })),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config


@exp_factory.register_config_factory('detr_coco_tfrecord')
def detr_coco_tfrecord() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 64
  eval_batch_size = 64
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  train_steps = 300 * steps_per_epoch  # 300 epochs
  decay_at = train_steps - 100 * steps_per_epoch  # 200 epochs
  config = cfg.ExperimentConfig(
      task=DetrTask(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=Detr(
              input_size=[1333, 1333, 3],
              norm_activation=common.NormActivation()),
          losses=Losses(),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
          )),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          validation_interval=5 * steps_per_epoch,
          max_to_keep=1,
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'detr_adamw',
                  'detr_adamw': {
                      'weight_decay_rate': 1e-4,
                      'global_clipnorm': 0.1,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [decay_at],
                      'values': [0.0001, 1.0e-05]
                  }
              },
          })),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config


@exp_factory.register_config_factory('detr_coco_tfds')
def detr_coco_tfds() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 64
  eval_batch_size = 64
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  train_steps = 300 * steps_per_epoch  # 300 epochs
  decay_at = train_steps - 100 * steps_per_epoch  # 200 epochs
  config = cfg.ExperimentConfig(
      task=DetrTask(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          model=Detr(
              num_classes=81,
              input_size=[1333, 1333, 3],
              norm_activation=common.NormActivation()),
          losses=Losses(class_offset=1),
          train_data=DataConfig(
              tfds_name='coco/2017',
              tfds_split='train',
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
          ),
          validation_data=DataConfig(
              tfds_name='coco/2017',
              tfds_split='validation',
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          validation_interval=5 * steps_per_epoch,
          max_to_keep=1,
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'detr_adamw',
                  'detr_adamw': {
                      'weight_decay_rate': 1e-4,
                      'global_clipnorm': 0.1,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [decay_at],
                      'values': [0.0001, 1.0e-05]
                  }
              },
          })),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
