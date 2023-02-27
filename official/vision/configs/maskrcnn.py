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

"""R-CNN(-RS) configuration definition."""

import dataclasses
import os
from typing import List, Optional, Sequence, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class Parser(hyperparams.Config):
  num_channels: int = 3
  match_threshold: float = 0.5
  unmatched_threshold: float = 0.5
  aug_rand_hflip: bool = False
  aug_rand_vflip: bool = False
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_type: Optional[
      common.Augmentation] = None  # Choose from AutoAugment and RandAugment.
  skip_crowd_during_training: bool = True
  max_num_instances: int = 100
  rpn_match_threshold: float = 0.7
  rpn_unmatched_threshold: float = 0.3
  rpn_batch_size_per_im: int = 256
  rpn_fg_fraction: float = 0.5
  mask_crop_size: int = 112


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: Union[Sequence[str], str, hyperparams.Config] = ''
  weights: Optional[hyperparams.Config] = None
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: common.DataDecoder = common.DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True
  # Number of examples in the data set, it's used to create the annotation file.
  num_examples: int = -1


@dataclasses.dataclass
class Anchor(hyperparams.Config):
  num_scales: int = 1
  aspect_ratios: List[float] = dataclasses.field(
      default_factory=lambda: [0.5, 1.0, 2.0])
  anchor_size: float = 8.0


@dataclasses.dataclass
class RPNHead(hyperparams.Config):
  num_convs: int = 1
  num_filters: int = 256
  use_separable_conv: bool = False


@dataclasses.dataclass
class DetectionHead(hyperparams.Config):
  num_convs: int = 4
  num_filters: int = 256
  use_separable_conv: bool = False
  num_fcs: int = 1
  fc_dims: int = 1024
  class_agnostic_bbox_pred: bool = False  # Has to be True for Cascade RCNN.
  # If additional IoUs are passed in 'cascade_iou_thresholds'
  # then ensemble the class probabilities from all heads.
  cascade_class_ensemble: bool = False


@dataclasses.dataclass
class ROIGenerator(hyperparams.Config):
  pre_nms_top_k: int = 2000
  pre_nms_score_threshold: float = 0.0
  pre_nms_min_size_threshold: float = 0.0
  nms_iou_threshold: float = 0.7
  num_proposals: int = 1000
  test_pre_nms_top_k: int = 1000
  test_pre_nms_score_threshold: float = 0.0
  test_pre_nms_min_size_threshold: float = 0.0
  test_nms_iou_threshold: float = 0.7
  test_num_proposals: int = 1000
  use_batched_nms: bool = False


@dataclasses.dataclass
class ROISampler(hyperparams.Config):
  mix_gt_boxes: bool = True
  num_sampled_rois: int = 512
  foreground_fraction: float = 0.25
  foreground_iou_threshold: float = 0.5
  background_iou_high_threshold: float = 0.5
  background_iou_low_threshold: float = 0.0
  # IoU thresholds for additional FRCNN heads in Cascade mode.
  # `foreground_iou_threshold` is the first threshold.
  cascade_iou_thresholds: Optional[List[float]] = None


@dataclasses.dataclass
class ROIAligner(hyperparams.Config):
  crop_size: int = 7
  sample_offset: float = 0.5


@dataclasses.dataclass
class DetectionGenerator(hyperparams.Config):
  apply_nms: bool = True
  pre_nms_top_k: int = 5000
  pre_nms_score_threshold: float = 0.05
  nms_iou_threshold: float = 0.5
  max_num_detections: int = 100
  nms_version: str = 'v2'  # `v2`, `v1`, `batched`
  use_cpu_nms: bool = False
  soft_nms_sigma: Optional[float] = None  # Only works when nms_version='v1'.
  use_sigmoid_probability: bool = False


@dataclasses.dataclass
class MaskHead(hyperparams.Config):
  upsample_factor: int = 2
  num_convs: int = 4
  num_filters: int = 256
  use_separable_conv: bool = False
  class_agnostic: bool = False


@dataclasses.dataclass
class MaskSampler(hyperparams.Config):
  num_sampled_masks: int = 128


@dataclasses.dataclass
class MaskROIAligner(hyperparams.Config):
  crop_size: int = 14
  sample_offset: float = 0.5


@dataclasses.dataclass
class MaskRCNN(hyperparams.Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 2
  max_level: int = 6
  anchor: Anchor = Anchor()
  include_mask: bool = True
  outer_boxes_scale: float = 1.0
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(
      type='fpn', fpn=decoders.FPN())
  rpn_head: RPNHead = RPNHead()
  detection_head: DetectionHead = DetectionHead()
  roi_generator: ROIGenerator = ROIGenerator()
  roi_sampler: ROISampler = ROISampler()
  roi_aligner: ROIAligner = ROIAligner()
  detection_generator: DetectionGenerator = DetectionGenerator()
  mask_head: Optional[MaskHead] = MaskHead()
  mask_sampler: Optional[MaskSampler] = MaskSampler()
  mask_roi_aligner: Optional[MaskROIAligner] = MaskROIAligner()
  norm_activation: common.NormActivation = common.NormActivation(
      norm_momentum=0.997,
      norm_epsilon=0.0001,
      use_sync_bn=True)


@dataclasses.dataclass
class Losses(hyperparams.Config):
  loss_weight: float = 1.0
  rpn_huber_loss_delta: float = 1. / 9.
  frcnn_huber_loss_delta: float = 1.
  frcnn_class_use_binary_cross_entropy: bool = False
  frcnn_class_loss_top_k_percent: float = 1.
  l2_weight_decay: float = 0.0
  rpn_score_weight: float = 1.0
  rpn_box_weight: float = 1.0
  frcnn_class_weight: float = 1.0
  frcnn_box_weight: float = 1.0
  mask_weight: float = 1.0


@dataclasses.dataclass
class MaskRCNNTask(cfg.TaskConfig):
  model: MaskRCNN = MaskRCNN()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False,
                                           drop_remainder=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False
  # If set, we only use masks for the specified class IDs.
  allowed_mask_class_ids: Optional[List[int]] = None
  # If set, the COCO metrics will be computed.
  use_coco_metrics: bool = True
  # If set, the Waymo Open Dataset evaluator would be used.
  use_wod_metrics: bool = False
  # If set, use instance metrics (AP, mask AP, etc.) computed by an efficient
  # approximation algorithm with TPU compatible operations.
  use_approx_instance_metrics: bool = False

  # If set, freezes the backbone during training.
  # TODO(crisnv) Add paper link when available.
  freeze_backbone: bool = False


COCO_INPUT_PATH_BASE = 'coco'


@exp_factory.register_config_factory('fasterrcnn_resnetfpn_coco')
def fasterrcnn_resnetfpn_coco() -> cfg.ExperimentConfig:
  """COCO object detection with Faster R-CNN."""
  steps_per_epoch = 500
  coco_val_samples = 5000
  train_batch_size = 64
  eval_batch_size = 8

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=MaskRCNNTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskRCNN(
              num_classes=91,
              input_size=[1024, 1024, 3],
              include_mask=False,
              mask_head=None,
              mask_sampler=None,
              mask_roi_aligner=None),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.8, aug_scale_max=1.25)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=22500,
          validation_steps=coco_val_samples // eval_batch_size,
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
                      'boundaries': [15000, 20000],
                      'values': [0.12, 0.012, 0.0012],
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


@exp_factory.register_config_factory('maskrcnn_resnetfpn_coco')
def maskrcnn_resnetfpn_coco() -> cfg.ExperimentConfig:
  """COCO object detection with Mask R-CNN."""
  steps_per_epoch = 500
  coco_val_samples = 5000
  train_batch_size = 64
  eval_batch_size = 8

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='bfloat16', enable_xla=True),
      task=MaskRCNNTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskRCNN(
              num_classes=91, input_size=[1024, 1024, 3], include_mask=True),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.8, aug_scale_max=1.25)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=22500,
          validation_steps=coco_val_samples // eval_batch_size,
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
                      'boundaries': [15000, 20000],
                      'values': [0.12, 0.012, 0.0012],
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


@exp_factory.register_config_factory('maskrcnn_spinenet_coco')
def maskrcnn_spinenet_coco() -> cfg.ExperimentConfig:
  """COCO object detection with Mask R-CNN with SpineNet backbone."""
  steps_per_epoch = 463
  coco_val_samples = 5000
  train_batch_size = 256
  eval_batch_size = 8

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=MaskRCNNTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskRCNN(
              backbone=backbones.Backbone(
                  type='spinenet',
                  spinenet=backbones.SpineNet(
                      model_id='49',
                      min_level=3,
                      max_level=7,
                  )),
              decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
              anchor=Anchor(anchor_size=3),
              norm_activation=common.NormActivation(use_sync_bn=True),
              num_classes=91,
              input_size=[640, 640, 3],
              min_level=3,
              max_level=7,
              include_mask=True),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.5, aug_scale_max=2.0)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=steps_per_epoch * 350,
          validation_steps=coco_val_samples // eval_batch_size,
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
                          steps_per_epoch * 320, steps_per_epoch * 340
                      ],
                      'values': [0.32, 0.032, 0.0032],
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


@exp_factory.register_config_factory('cascadercnn_spinenet_coco')
def cascadercnn_spinenet_coco() -> cfg.ExperimentConfig:
  """COCO object detection with Cascade RCNN-RS with SpineNet backbone."""
  steps_per_epoch = 463
  coco_val_samples = 5000
  train_batch_size = 256
  eval_batch_size = 8

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=MaskRCNNTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskRCNN(
              backbone=backbones.Backbone(
                  type='spinenet',
                  spinenet=backbones.SpineNet(
                      model_id='49',
                      min_level=3,
                      max_level=7,
                  )),
              decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
              roi_sampler=ROISampler(cascade_iou_thresholds=[0.6, 0.7]),
              detection_head=DetectionHead(
                  class_agnostic_bbox_pred=True, cascade_class_ensemble=True),
              anchor=Anchor(anchor_size=3),
              norm_activation=common.NormActivation(
                  use_sync_bn=True, activation='swish'),
              num_classes=91,
              input_size=[640, 640, 3],
              min_level=3,
              max_level=7,
              include_mask=True),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.1, aug_scale_max=2.5)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=steps_per_epoch * 500,
          validation_steps=coco_val_samples // eval_batch_size,
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
                          steps_per_epoch * 475, steps_per_epoch * 490
                      ],
                      'values': [0.32, 0.032, 0.0032],
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


@exp_factory.register_config_factory('maskrcnn_mobilenet_coco')
def maskrcnn_mobilenet_coco() -> cfg.ExperimentConfig:
  """COCO object detection with Mask R-CNN with MobileNet backbone."""
  steps_per_epoch = 232
  coco_val_samples = 5000
  train_batch_size = 512
  eval_batch_size = 512

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=MaskRCNNTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskRCNN(
              backbone=backbones.Backbone(
                  type='mobilenet',
                  mobilenet=backbones.MobileNet(model_id='MobileNetV2')),
              decoder=decoders.Decoder(
                  type='fpn',
                  fpn=decoders.FPN(num_filters=128, use_separable_conv=True)),
              rpn_head=RPNHead(use_separable_conv=True,
                               num_filters=128),  # 1/2 of original channels.
              detection_head=DetectionHead(
                  use_separable_conv=True, num_filters=128,
                  fc_dims=512),  # 1/2 of original channels.
              mask_head=MaskHead(use_separable_conv=True,
                                 num_filters=128),  # 1/2 of original channels.
              anchor=Anchor(anchor_size=3),
              norm_activation=common.NormActivation(
                  activation='relu6',
                  norm_momentum=0.99,
                  norm_epsilon=0.001,
                  use_sync_bn=True),
              num_classes=91,
              input_size=[512, 512, 3],
              min_level=3,
              max_level=6,
              include_mask=True),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.5, aug_scale_max=2.0)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=steps_per_epoch * 350,
          validation_steps=coco_val_samples // eval_batch_size,
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
                          steps_per_epoch * 320, steps_per_epoch * 340
                      ],
                      'values': [0.32, 0.032, 0.0032],
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
