# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""RetinaNet configuration definition."""

import dataclasses
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders
from official.vision.beta.configs import backbones


# pylint: disable=missing-class-docstring
# Keep for backward compatibility.
@dataclasses.dataclass
class TfExampleDecoder(common.TfExampleDecoder):
  """A simple TF Example decoder config."""


# Keep for backward compatibility.
@dataclasses.dataclass
class TfExampleDecoderLabelMap(common.TfExampleDecoderLabelMap):
  """TF Example decoder with label map config."""


# Keep for backward compatibility.
@dataclasses.dataclass
class DataDecoder(common.DataDecoder):
  """Data decoder config."""


@dataclasses.dataclass
class Parser(hyperparams.Config):
  num_channels: int = 3
  match_threshold: float = 0.5
  unmatched_threshold: float = 0.5
  aug_rand_hflip: bool = False
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_policy: Optional[str] = None
  skip_crowd_during_training: bool = True
  max_num_instances: int = 100


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: common.DataDecoder = common.DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'


@dataclasses.dataclass
class Anchor(hyperparams.Config):
  num_scales: int = 3
  aspect_ratios: List[float] = dataclasses.field(
      default_factory=lambda: [0.5, 1.0, 2.0])
  anchor_size: float = 4.0


@dataclasses.dataclass
class Losses(hyperparams.Config):
  focal_loss_alpha: float = 0.25
  focal_loss_gamma: float = 1.5
  huber_loss_delta: float = 0.1
  box_loss_weight: int = 50
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class AttributeHead(hyperparams.Config):
  name: str = ''
  type: str = 'regression'
  size: int = 1


@dataclasses.dataclass
class RetinaNetHead(hyperparams.Config):
  num_convs: int = 4
  num_filters: int = 256
  use_separable_conv: bool = False
  attribute_heads: List[AttributeHead] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DetectionGenerator(hyperparams.Config):
  apply_nms: bool = True
  pre_nms_top_k: int = 5000
  pre_nms_score_threshold: float = 0.05
  nms_iou_threshold: float = 0.5
  max_num_detections: int = 100
  nms_version: str = 'v2'  # `v2`, `v1`, `batched`.
  use_cpu_nms: bool = False


@dataclasses.dataclass
class RetinaNet(hyperparams.Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 7
  anchor: Anchor = Anchor()
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(
      type='fpn', fpn=decoders.FPN())
  head: RetinaNetHead = RetinaNetHead()
  detection_generator: DetectionGenerator = DetectionGenerator()
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class ExportConfig(hyperparams.Config):
  output_normalized_coordinates: bool = False
  cast_num_detections_to_float: bool = False
  cast_detection_classes_to_float: bool = False


@dataclasses.dataclass
class RetinaNetTask(cfg.TaskConfig):
  model: RetinaNet = RetinaNet()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False
  export_config: ExportConfig = ExportConfig()


@exp_factory.register_config_factory('retinanet')
def retinanet() -> cfg.ExperimentConfig:
  """RetinaNet general config."""
  return cfg.ExperimentConfig(
      task=RetinaNetTask(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('retinanet_resnetfpn_coco')
def retinanet_resnetfpn_coco() -> cfg.ExperimentConfig:
  """COCO object detection with RetinaNet."""
  train_batch_size = 256
  eval_batch_size = 8
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=RetinaNetTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=RetinaNet(
              num_classes=91,
              input_size=[640, 640, 3],
              norm_activation=common.NormActivation(use_sync_bn=False),
              min_level=3,
              max_level=7),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.8, aug_scale_max=1.2)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          train_steps=72 * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
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
                          57 * steps_per_epoch, 67 * steps_per_epoch
                      ],
                      'values': [
                          0.32 * train_batch_size / 256.0,
                          0.032 * train_batch_size / 256.0,
                          0.0032 * train_batch_size / 256.0
                      ],
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
                      'warmup_learning_rate': 0.0067
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('retinanet_spinenet_coco')
def retinanet_spinenet_coco() -> cfg.ExperimentConfig:
  """COCO object detection with RetinaNet using SpineNet backbone."""
  train_batch_size = 256
  eval_batch_size = 8
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  input_size = 640

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=RetinaNetTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=RetinaNet(
              backbone=backbones.Backbone(
                  type='spinenet',
                  spinenet=backbones.SpineNet(
                      model_id='49',
                      stochastic_depth_drop_rate=0.2,
                      min_level=3,
                      max_level=7)),
              decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
              anchor=Anchor(anchor_size=3),
              norm_activation=common.NormActivation(
                  use_sync_bn=True, activation='swish'),
              num_classes=91,
              input_size=[input_size, input_size, 3],
              min_level=3,
              max_level=7),
          losses=Losses(l2_weight_decay=4e-5),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.1, aug_scale_max=2.0)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          train_steps=500 * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
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
                          475 * steps_per_epoch, 490 * steps_per_epoch
                      ],
                      'values': [
                          0.32 * train_batch_size / 256.0,
                          0.032 * train_batch_size / 256.0,
                          0.0032 * train_batch_size / 256.0
                      ],
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0.0067
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.model.min_level == task.model.backbone.spinenet.min_level',
          'task.model.max_level == task.model.backbone.spinenet.max_level',
      ])

  return config


@exp_factory.register_config_factory('retinanet_mobile_coco')
def retinanet_spinenet_mobile_coco() -> cfg.ExperimentConfig:
  """COCO object detection with mobile RetinaNet."""
  train_batch_size = 256
  eval_batch_size = 8
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  input_size = 384

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=RetinaNetTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=RetinaNet(
              backbone=backbones.Backbone(
                  type='spinenet_mobile',
                  spinenet_mobile=backbones.SpineNetMobile(
                      model_id='49',
                      stochastic_depth_drop_rate=0.2,
                      min_level=3,
                      max_level=7,
                      use_keras_upsampling_2d=False)),
              decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
              head=RetinaNetHead(num_filters=48, use_separable_conv=True),
              anchor=Anchor(anchor_size=3),
              norm_activation=common.NormActivation(
                  use_sync_bn=True, activation='swish'),
              num_classes=91,
              input_size=[input_size, input_size, 3],
              min_level=3,
              max_level=7),
          losses=Losses(l2_weight_decay=3e-5),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.1, aug_scale_max=2.0)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          train_steps=600 * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
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
                          575 * steps_per_epoch, 590 * steps_per_epoch
                      ],
                      'values': [
                          0.32 * train_batch_size / 256.0,
                          0.032 * train_batch_size / 256.0,
                          0.0032 * train_batch_size / 256.0
                      ],
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0.0067
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])

  return config
