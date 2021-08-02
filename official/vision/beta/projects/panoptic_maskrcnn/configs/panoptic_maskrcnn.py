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

"""Panoptic Mask R-CNN configuration definition."""

import dataclasses
import os
from typing import List, Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.beta.configs import maskrcnn
from official.vision.beta.configs import semantic_segmentation


SEGMENTATION_MODEL = semantic_segmentation.SemanticSegmentationModel
SEGMENTATION_HEAD = semantic_segmentation.SegmentationHead

_COCO_INPUT_PATH_BASE = 'coco'
_COCO_TRAIN_EXAMPLES = 118287
_COCO_VAL_EXAMPLES = 5000

# pytype: disable=wrong-keyword-args


@dataclasses.dataclass
class Parser(maskrcnn.Parser):
  """Panoptic Mask R-CNN parser config."""
  # If segmentation_resize_eval_groundtruth is set to False, original image
  # sizes are used for eval. In that case,
  # segmentation_groundtruth_padded_size has to be specified too to allow for
  # batching the variable input sizes of images.
  segmentation_resize_eval_groundtruth: bool = True
  segmentation_groundtruth_padded_size: List[int] = dataclasses.field(
      default_factory=list)
  segmentation_ignore_label: int = 255


@dataclasses.dataclass
class DataConfig(maskrcnn.DataConfig):
  """Input config for training."""
  parser: Parser = Parser()


@dataclasses.dataclass
class PanopticMaskRCNN(maskrcnn.MaskRCNN):
  """Panoptic Mask R-CNN model config."""
  segmentation_model: semantic_segmentation.SemanticSegmentationModel = (
      SEGMENTATION_MODEL(num_classes=2))
  include_mask = True
  shared_backbone: bool = True
  shared_decoder: bool = True


@dataclasses.dataclass
class Losses(maskrcnn.Losses):
  """Panoptic Mask R-CNN loss config."""
  semantic_segmentation_label_smoothing: float = 0.0
  semantic_segmentation_ignore_label: int = 255
  semantic_segmentation_class_weights: List[float] = dataclasses.field(
      default_factory=list)
  semantic_segmentation_use_groundtruth_dimension: bool = True
  semantic_segmentation_top_k_percent_pixels: float = 1.0
  semantic_segmentation_weight: float = 1.0


@dataclasses.dataclass
class PanopticMaskRCNNTask(maskrcnn.MaskRCNNTask):
  """Panoptic Mask R-CNN task config."""
  model: PanopticMaskRCNN = PanopticMaskRCNN()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False,
                                           drop_remainder=False)
  segmentation_evaluation: semantic_segmentation.Evaluation = semantic_segmentation.Evaluation()  # pylint: disable=line-too-long
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  segmentation_init_checkpoint: Optional[str] = None

  # 'init_checkpoint_modules' controls the modules that need to be initialized
  # from checkpoint paths given by 'init_checkpoint' and/or
  # 'segmentation_init_checkpoint. Supports modules:
  # 'backbone': Initialize MaskRCNN backbone
  # 'segmentation_backbone': Initialize segmentation backbone
  # 'segmentation_decoder': Initialize segmentation decoder
  # 'all': Initialize all modules
  init_checkpoint_modules: Optional[List[str]] = dataclasses.field(
      default_factory=list)


@exp_factory.register_config_factory('panoptic_maskrcnn_resnetfpn_coco')
def panoptic_maskrcnn_resnetfpn_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with Panoptic Mask R-CNN."""
  train_batch_size = 64
  eval_batch_size = 8
  steps_per_epoch = _COCO_TRAIN_EXAMPLES // train_batch_size
  validation_steps = _COCO_VAL_EXAMPLES // eval_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=PanopticMaskRCNNTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=PanopticMaskRCNN(
              num_classes=91, input_size=[1024, 1024, 3],
              segmentation_model=SEGMENTATION_MODEL(
                  num_classes=91,
                  head=SEGMENTATION_HEAD(level=3))),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.8, aug_scale_max=1.25)),
          validation_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False),
          annotation_file=os.path.join(_COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json')),
      trainer=cfg.TrainerConfig(
          train_steps=22500,
          validation_steps=validation_steps,
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
