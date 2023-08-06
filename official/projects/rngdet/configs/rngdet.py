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
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones


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
  background_cls_weight: float = 0.2
  l2_weight_decay: float = 1e-5


@dataclasses.dataclass
class Rngdet(hyperparams.Config):
  """Rngdet model definations."""
  num_queries: int = 10
  hidden_size: int = 256
  num_classes: int = 2  # 1: background
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  input_size: List[int] = dataclasses.field(default_factory=list)
  roi_size: int = 128
  backbone: backbones.Backbone = dataclasses.field(default_factory=lambda:backbones.Backbone(
      type='resnet', resnet=backbones.ResNet(model_id=50, bn_trainable=False)))
  decoder: decoders.Decoder = dataclasses.field(
      default_factory=lambda: decoders.Decoder(type='fpn', fpn=decoders.FPN())
  )
  min_level: int = 2
  max_level: int = 5
  norm_activation: common.NormActivation = dataclasses.field(default_factory=common.NormActivation)
  backbone_endpoint_name: str = '5'


@dataclasses.dataclass
class RngdetTask(cfg.TaskConfig):
  model: Rngdet = dataclasses.field(default_factory=Rngdet)
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = 'all'  # all, backbone
  per_category_metrics: bool = False


CITYSCALE_INPUT_PATH_BASE = '/data2/cityscale/tfrecord/'
CITYSCALE_TRAIN_EXAMPLES = 420140
CITYSCALE_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('rngdet_cityscale')
def rngdet_cityscale() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 16
  eval_batch_size = 64
  steps_per_epoch = CITYSCALE_TRAIN_EXAMPLES // train_batch_size
  train_steps = 50 * steps_per_epoch  # 50 epochs
  decay_at = train_steps - 40 * steps_per_epoch  # 40 epochs
  config = cfg.ExperimentConfig(
      task=RngdetTask(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          model=Rngdet(
              input_size=[128, 128, 3],
              roi_size=128,
              norm_activation=common.NormActivation()),
          losses=Losses(),
          train_data=DataConfig(
              input_path=os.path.join(CITYSCALE_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(CITYSCALE_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
          )),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=CITYSCALE_VAL_EXAMPLES // eval_batch_size,
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
                      'weight_decay_rate': 1e-5,
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