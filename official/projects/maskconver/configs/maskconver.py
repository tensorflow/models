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

"""Panoptic Mask R-CNN configuration definition."""

import dataclasses
import os
from typing import List, Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.projects.maskconver.configs import backbones
from official.projects.maskconver.configs import decoders
from official.vision.configs import common
from official.vision.configs import maskrcnn
from official.vision.configs import semantic_segmentation


_COCO_INPUT_PATH_BASE = 'coco/tfrecords'
_COCO_TRAIN_EXAMPLES = 118287
_COCO_VAL_EXAMPLES = 5000

# PASCAL VOC 2012 Dataset
_PASCAL_TRAIN_EXAMPLES = 10582
_PASCAL_VAL_EXAMPLES = 1449
_PASCAL_INPUT_PATH_BASE = 'gs://**/pascal_voc_seg'

# Cityscapes Dataset
_CITYSCAPES_TRAIN_EXAMPLES = 2975
_CITYSCAPES_VAL_EXAMPLES = 500
_CITYSCAPES_INPUT_PATH_BASE = 'cityscapes/tfrecord'


# pytype: disable=wrong-keyword-args
# pylint: disable=unexpected-keyword-arg


@dataclasses.dataclass
class CopyPaste(hyperparams.Config):
  copypaste_frequency: float = 0.5
  aug_scale_min: float = 0.1
  aug_scale_max: float = 1.9
  copypaste_aug_scale_max: float = 1.0
  copypaste_aug_scale_min: float = 0.05


@dataclasses.dataclass
class Parser(maskrcnn.Parser):
  """MaskConver parser config."""
  # If segmentation_resize_eval_groundtruth is set to False, original image
  # sizes are used for eval. In that case,
  # segmentation_groundtruth_padded_size has to be specified too to allow for
  # batching the variable input sizes of images.
  segmentation_resize_eval_groundtruth: bool = True
  segmentation_groundtruth_padded_size: List[int] = dataclasses.field(
      default_factory=list)
  segmentation_ignore_label: int = 0
  panoptic_ignore_label: int = 0
  # Setting this to true will enable parsing category_mask and instance_mask.
  include_panoptic_masks: bool = True
  gaussaian_iou: float = 0.7
  max_num_stuff_centers: int = 3
  aug_type: common.Augmentation = dataclasses.field(
      default_factory=common.Augmentation
  )
  copypaste: CopyPaste = dataclasses.field(default_factory=CopyPaste)


@dataclasses.dataclass
class TfExampleDecoder(common.TfExampleDecoder):
  """A simple TF Example decoder config."""
  # Setting this to true will enable decoding category_mask and instance_mask.
  include_panoptic_masks: bool = True
  panoptic_category_mask_key: str = 'image/panoptic/category_mask'
  panoptic_instance_mask_key: str = 'image/panoptic/instance_mask'


@dataclasses.dataclass
class DataDecoder(common.DataDecoder):
  """Data decoder config."""
  simple_decoder: TfExampleDecoder = dataclasses.field(
      default_factory=TfExampleDecoder
  )


@dataclasses.dataclass
class DataConfig(maskrcnn.DataConfig):
  """Input config for training."""
  decoder: DataDecoder = dataclasses.field(default_factory=DataDecoder)
  parser: Parser = dataclasses.field(default_factory=Parser)
  dtype: str = 'float32'
  prefetch_buffer_size: int = 8


@dataclasses.dataclass
class Anchor(hyperparams.Config):
  num_scales: int = 1
  aspect_ratios: List[float] = dataclasses.field(
      default_factory=lambda: [0.5, 1.0, 2.0])
  anchor_size: float = 8.0


@dataclasses.dataclass
class PanopticGenerator(hyperparams.Config):
  """MaskConver panoptic generator."""
  object_mask_threshold: float = 0.001
  small_area_threshold: int = 0
  overlap_threshold: float = 0.5
  rescale_predictions: bool = True
  use_hardware_optimization: bool = False


@dataclasses.dataclass
class SegmentationHead(semantic_segmentation.SegmentationHead):
  """Segmentation head config."""
  depthwise_kernel_size: int = 7
  use_layer_norm: bool = False


@dataclasses.dataclass
class MaskConver(hyperparams.Config):
  """MaskConver model config."""
  num_classes: int = 0
  num_thing_classes: int = 0
  num_instances: int = 100
  embedding_size: int = 512
  padded_output_size: List[int] = dataclasses.field(default_factory=list)
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 2
  max_level: int = 6
  num_anchors: int = 100
  panoptic_fusion_num_filters: int = 256
  anchor: Anchor = dataclasses.field(default_factory=Anchor)
  level: int = 3
  class_head: SegmentationHead = dataclasses.field(
      default_factory=SegmentationHead
  )
  mask_embedding_head: SegmentationHead = dataclasses.field(
      default_factory=SegmentationHead
  )
  per_pixel_embedding_head: SegmentationHead = dataclasses.field(
      default_factory=SegmentationHead
  )
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=backbones.Backbone
  )
  decoder: decoders.Decoder = dataclasses.field(
      default_factory=lambda: decoders.Decoder(type='identity')
  )
  mask_decoder: Optional[decoders.Decoder] = dataclasses.field(
      default_factory=lambda: decoders.Decoder(type='identity')
  )
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=common.NormActivation
  )
  panoptic_generator: PanopticGenerator = dataclasses.field(
      default_factory=PanopticGenerator
  )


@dataclasses.dataclass
class Losses(hyperparams.Config):
  """maskconver loss config."""
  l2_weight_decay: float = 0.0
  ignore_label: int = 0
  use_groundtruth_dimension: bool = True
  top_k_percent_pixels_category: float = 1.0
  top_k_percent_pixels_instance: float = 1.0
  loss_weight: float = 1.0
  mask_weight: float = 10.0
  beta: float = 4.0
  alpha: float = 2.0


@dataclasses.dataclass
class PanopticQualityEvaluator(hyperparams.Config):
  """Panoptic Quality Evaluator config."""
  num_categories: int = 2
  ignored_label: int = 0
  max_instances_per_category: int = 256
  offset: int = 256 * 256 * 256
  is_thing: List[float] = dataclasses.field(
      default_factory=list)
  rescale_predictions: bool = True
  report_per_class_metrics: bool = False

###################################
###### PANOPTIC SEGMENTATION ######
###################################


@dataclasses.dataclass
class MaskConverTask(cfg.TaskConfig):
  """MaskConverTask task config."""
  model: MaskConver = dataclasses.field(default_factory=MaskConver)
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  # pylint: disable=g-long-lambda
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(
          is_training=False, drop_remainder=False
      )
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None

  init_checkpoint_modules: Optional[List[str]] = dataclasses.field(
      default_factory=list)
  panoptic_quality_evaluator: PanopticQualityEvaluator = dataclasses.field(
      default_factory=PanopticQualityEvaluator
  )
  # pylint: enable=g-long-lambda


@exp_factory.register_config_factory('maskconver_coco')
def maskconver_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with MaskConver."""
  train_batch_size = 64
  eval_batch_size = 8
  # steps_per_epoch = _COCO_TRAIN_EXAMPLES // train_batch_size
  validation_steps = _COCO_VAL_EXAMPLES // eval_batch_size

  # coco panoptic dataset has category ids ranging from [0-200] inclusive.
  # 0 is not used and represents the background class
  # ids 1-91 represent thing categories (91)
  # ids 92-200 represent stuff categories (109)
  # for the segmentation task, we continue using id=0 for the background
  # and map all thing categories to id=1, the remaining 109 stuff categories
  # are shifted by an offset=90 given by num_thing classes - 1. This shifting
  # will make all the stuff categories begin from id=2 and end at id=110
  num_panoptic_categories = 201
  num_thing_categories = 91
  # num_semantic_segmentation_classes = 111

  is_thing = [False]
  for idx in range(1, num_panoptic_categories):
    is_thing.append(True if idx < num_thing_categories else False)

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='float32', enable_xla=False),
      task=MaskConverTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=MaskConver(
              num_classes=201, num_thing_classes=91, input_size=[512, 512, 3],
              padded_output_size=[512, 512]),
          losses=Losses(l2_weight_decay=0.0),
          train_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'train-nocrowd*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.5, aug_scale_max=1.5,
                  aug_type=common.Augmentation(
                      type='autoaug',
                      autoaug=common.AutoAugment(
                          augmentation_name='panoptic_deeplab_policy')))),
          validation_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'val-nocrowd*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              parser=Parser(
                  segmentation_resize_eval_groundtruth=True,
                  segmentation_groundtruth_padded_size=[640, 640]),
              drop_remainder=False),
          panoptic_quality_evaluator=PanopticQualityEvaluator(
              num_categories=num_panoptic_categories,
              ignored_label=0,
              is_thing=is_thing,
              rescale_predictions=True)),
      trainer=cfg.TrainerConfig(
          train_steps=200000,
          validation_steps=validation_steps,
          validation_interval=1000,
          steps_per_loop=1000,
          summary_interval=1000,
          checkpoint_interval=1000,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.08,
                      'decay_steps': 200000,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0.0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config

###################################
###### SEMANTIC SEGMENTATION ######
###################################


@dataclasses.dataclass
class SegDataConfig(cfg.DataConfig):
  """Input config for training."""
  output_size: List[int] = dataclasses.field(default_factory=list)
  # If crop_size is specified, image will be resized first to
  # output_size, then crop of size crop_size will be cropped.
  crop_size: List[int] = dataclasses.field(default_factory=list)
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 1000
  prefetch_buffer_size: int = 8
  cycle_length: int = 10
  # If resize_eval_groundtruth is set to False, original image sizes are used
  # for eval. In that case, groundtruth_padded_size has to be specified too to
  # allow for batching the variable input sizes of images.
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_hflip: bool = True
  preserve_aspect_ratio: bool = True
  aug_policy: Optional[str] = None
  drop_remainder: bool = True
  file_type: str = 'tfrecord'
  gaussaian_iou: float = 0.7
  max_num_stuff_centers: int = 3
  max_num_instances: int = 100
  aug_type: common.Augmentation = dataclasses.field(
      default_factory=common.Augmentation)


@dataclasses.dataclass
class MaskConverSegTask(cfg.TaskConfig):
  """MaskConverTask task config."""
  model: MaskConver = dataclasses.field(default_factory=MaskConver)
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: SegDataConfig(is_training=True)
  )
  # pylint: disable=g-long-lambda
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: SegDataConfig(
          is_training=False, drop_remainder=False
      )
  )
  # pylint: enable=g-long-lambda
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None

  init_checkpoint_modules: Optional[List[str]] = dataclasses.field(
      default_factory=list)


@exp_factory.register_config_factory('maskconver_seg_pascal')
def maskconver_seg_pascal() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with MaskConver."""
  train_batch_size = 64
  eval_batch_size = 8
  validation_steps = _PASCAL_VAL_EXAMPLES // eval_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='float32', enable_xla=False),
      task=MaskConverSegTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=MaskConver(
              num_classes=21, num_thing_classes=91, input_size=[512, 512, 3],
              padded_output_size=[512, 512]),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=SegDataConfig(
              input_path=os.path.join(_PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              output_size=[512, 512],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0,
              aug_type=common.Augmentation(
                  type='autoaug',
                  autoaug=common.AutoAugment(
                      augmentation_name='panoptic_deeplab_policy'))),
          validation_data=SegDataConfig(
              input_path=os.path.join(_PASCAL_INPUT_PATH_BASE, 'val*'),
              output_size=[512, 512],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=200000,
          validation_steps=validation_steps,
          validation_interval=1000,
          steps_per_loop=1000,
          summary_interval=1000,
          checkpoint_interval=1000,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.08,
                      'decay_steps': 200000,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0.0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('maskconver_seg_cityscapes')
def maskconver_seg_cityscapes() -> cfg.ExperimentConfig:
  """Cityscapes semantic segmentation with MaskConver."""
  train_batch_size = 32
  eval_batch_size = 8
  validation_steps = _CITYSCAPES_VAL_EXAMPLES // eval_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='float32', enable_xla=False),
      task=MaskConverSegTask(
          init_checkpoint='maskconver_seg_mnv3p5rf_coco_200k/43437096',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=MaskConver(
              num_classes=19, input_size=[None, None, 3],
              padded_output_size=[1024, 2048]),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=SegDataConfig(
              input_path=os.path.join(_CITYSCAPES_INPUT_PATH_BASE,
                                      'train_fine*'),
              output_size=[1024, 2048],
              crop_size=[512, 1024],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0,
              aug_type=common.Augmentation(
                  type='autoaug',
                  autoaug=common.AutoAugment(
                      augmentation_name='panoptic_deeplab_policy'))),
          validation_data=SegDataConfig(
              input_path=os.path.join(_CITYSCAPES_INPUT_PATH_BASE, 'val_fine*'),
              output_size=[1024, 2048],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[1024, 2048],
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=100000,
          validation_steps=validation_steps,
          validation_interval=185,
          steps_per_loop=185,
          summary_interval=185,
          checkpoint_interval=185,
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
                      'initial_learning_rate': 0.01,
                      'decay_steps': 100000,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 925,
                      'warmup_learning_rate': 0.0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config
