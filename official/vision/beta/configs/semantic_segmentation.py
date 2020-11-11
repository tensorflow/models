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
"""Semantic segmentation configuration definition."""
import os
from typing import List, Optional, Union

import dataclasses
import numpy as np

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 1000
  cycle_length: int = 10
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  drop_remainder: bool = True


@dataclasses.dataclass
class SegmentationHead(hyperparams.Config):
  level: int = 3
  num_convs: int = 2
  num_filters: int = 256
  upsample_factor: int = 1
  feature_fusion: Optional[str] = None  # None, or deeplabv3plus
  # deeplabv3plus feature fusion params
  low_level: int = 2
  low_level_num_filters: int = 48


@dataclasses.dataclass
class SemanticSegmentationModel(hyperparams.Config):
  """Semantic segmentation model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 6
  head: SegmentationHead = SegmentationHead()
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(type='identity')
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class Losses(hyperparams.Config):
  label_smoothing: float = 0.1
  ignore_label: int = 255
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 0.0
  use_groundtruth_dimension: bool = True


@dataclasses.dataclass
class SemanticSegmentationTask(cfg.TaskConfig):
  """The model config."""
  model: SemanticSegmentationModel = SemanticSegmentationModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder


@exp_factory.register_config_factory('semantic_segmentation')
def semantic_segmentation() -> cfg.ExperimentConfig:
  """Semantic segmentation general."""
  return cfg.ExperimentConfig(
      task=SemanticSegmentationModel(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

# PASCAL VOC 2012 Dataset
PASCAL_TRAIN_EXAMPLES = 10582
PASCAL_VAL_EXAMPLES = 1449
PASCAL_INPUT_PATH_BASE = 'pascal_voc_seg'


@exp_factory.register_config_factory('seg_deeplabv3_pascal')
def seg_deeplabv3_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on imagenet with resnet deeplabv3."""
  train_batch_size = 16
  eval_batch_size = 8
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  output_stride = 8
  aspp_dilation_rates = [12, 24, 36]  # [6, 12, 18] if output_stride = 16
  level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              # TODO(arashwan): test changing size to 513 to match deeplab.
              input_size=[512, 512, 3],
              backbone=backbones.Backbone(
                  type='dilated_resnet', dilated_resnet=backbones.DilatedResNet(
                      model_id=50, output_stride=output_stride)),
              decoder=decoders.Decoder(
                  type='aspp', aspp=decoders.ASPP(
                      level=level, dilation_rates=aspp_dilation_rates)),
              head=SegmentationHead(level=level, num_convs=0),
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.9997,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False),
          # resnet50
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/deeplab/deeplab_resnet50_imagenet/ckpt-62400',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=45 * steps_per_epoch,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007,
                      'decay_steps': 45 * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
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


@exp_factory.register_config_factory('seg_deeplabv3plus_pascal')
def seg_deeplabv3plus_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on imagenet with resnet deeplabv3+."""
  train_batch_size = 16
  eval_batch_size = 8
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16
  aspp_dilation_rates = [6, 12, 18]  # [12, 24, 36] if output_stride = 8
  level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[512, 512, 3],
              backbone=backbones.Backbone(
                  type='dilated_resnet', dilated_resnet=backbones.DilatedResNet(
                      model_id=50, output_stride=output_stride)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level, dilation_rates=aspp_dilation_rates)),
              head=SegmentationHead(
                  level=level,
                  num_convs=2,
                  feature_fusion='deeplabv3plus',
                  low_level=2,
                  low_level_num_filters=48),
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.9997,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False),
          # resnet50
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/deeplab/deeplab_resnet50_imagenet/ckpt-62400',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=45 * steps_per_epoch,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007,
                      'decay_steps': 45 * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
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
