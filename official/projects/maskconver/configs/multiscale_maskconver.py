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

"""Multi-scale Maskconver configuration definition."""

import dataclasses
import os
from typing import List, Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.projects.maskconver.configs import maskconver
from official.vision.configs import common

# pylint: disable=unused-private-name
# pytype: disable=wrong-keyword-args
# pylint: disable=unexpected-keyword-arg

_COCO_INPUT_PATH_BASE = 'coco/tfrecords'
_COCO_TRAIN_EXAMPLES = 118287
_COCO_VAL_EXAMPLES = 5000

TfExampleDecoder = maskconver.TfExampleDecoder
DataDecoder = maskconver.DataDecoder
DataConfig = maskconver.DataConfig
Losses = maskconver.Losses
PanopticGenerator = maskconver.PanopticGenerator
PanopticQualityEvaluator = maskconver.PanopticQualityEvaluator


@dataclasses.dataclass
class CopyPaste(hyperparams.Config):
  copypaste_frequency: float = 1.0
  aug_scale_min: float = 0.1
  aug_scale_max: float = 1.9
  copypaste_aug_scale_max: float = 1.0
  copypaste_aug_scale_min: float = 0.05


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """MaskConver parser config."""
  aug_rand_hflip: bool = False
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
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
  max_num_instances: int = 256
  aug_type: common.Augmentation = dataclasses.field(
      default_factory=common.Augmentation)
  fpn_low_range: List[int] = dataclasses.field(default_factory=list)
  fpn_high_range: List[int] = dataclasses.field(default_factory=list)
  mask_target_level: int = 1
  copypaste: CopyPaste = dataclasses.field(default_factory=CopyPaste)


@dataclasses.dataclass
class MultiScaleMaskConverHead(hyperparams.Config):
  """Segmentation head config."""
  num_convs: int = 4
  num_filters: int = 256
  use_depthwise_convolution: bool = False
  prediction_kernel_size: int = 3
  upsample_factor: int = 1
  depthwise_kernel_size: int = 7
  use_layer_norm: bool = True


@dataclasses.dataclass
class MultiScaleMaskConver(maskconver.MaskConver):
  """Multi-scale MaskConver model config."""
  min_level: int = 3
  max_level: int = 7
  num_instances: int = 100
  class_head: MultiScaleMaskConverHead = dataclasses.field(
      default_factory=MultiScaleMaskConverHead
  )
  mask_embedding_head: MultiScaleMaskConverHead = dataclasses.field(
      default_factory=MultiScaleMaskConverHead
  )
  per_pixel_embedding_head: maskconver.SegmentationHead = dataclasses.field(
      default_factory=lambda: maskconver.SegmentationHead(use_layer_norm=True)
  )


###################################
###### PANOPTIC SEGMENTATION ######
###################################


@dataclasses.dataclass
class MultiScaleMaskConverTask(cfg.TaskConfig):
  """MaskConverTask task config."""
  model: MultiScaleMaskConver = dataclasses.field(
      default_factory=MultiScaleMaskConver
  )
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  # pylint: disable=g-long-lambda
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(
          is_training=False, drop_remainder=False
      )
  )
  # pylint: enable=g-long-lambda
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None

  init_checkpoint_modules: Optional[List[str]] = dataclasses.field(
      default_factory=list)
  panoptic_quality_evaluator: PanopticQualityEvaluator = dataclasses.field(
      default_factory=PanopticQualityEvaluator
  )


@exp_factory.register_config_factory('multiscale_maskconver_coco')
def multiscale_maskconver_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with MaskConver."""
  train_batch_size = 128
  eval_batch_size = 1
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
      task=MultiScaleMaskConverTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=MultiScaleMaskConver(
              num_classes=201,
              num_thing_classes=91,
              input_size=[640, 640, 3],
              padded_output_size=[640, 640]),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True,
                  aug_scale_min=0.1,
                  aug_scale_max=1.9,
                  fpn_low_range=[0, 40, 80, 160, 320],
                  fpn_high_range=[64, 128, 256, 512, 10000000],
                  aug_type=common.Augmentation(
                      type='autoaug',
                      autoaug=common.AutoAugment(
                          augmentation_name='panoptic_deeplab_policy')))),
          validation_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              parser=Parser(
                  segmentation_resize_eval_groundtruth=False,
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
          optimizer_config=optimization.OptimizationConfig()),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config

