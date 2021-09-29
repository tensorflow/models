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
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import common
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
  panoptic_ignore_label: int = 0
  # Setting this to true will enable parsing category_mask and instance_mask.
  include_panoptic_masks: bool = True


@dataclasses.dataclass
class TfExampleDecoder(common.TfExampleDecoder):
  """A simple TF Example decoder config."""
  # Setting this to true will enable decoding category_mask and instance_mask.
  include_panoptic_masks: bool = True


@dataclasses.dataclass
class DataDecoder(common.DataDecoder):
  """Data decoder config."""
  simple_decoder: TfExampleDecoder = TfExampleDecoder()


@dataclasses.dataclass
class DataConfig(maskrcnn.DataConfig):
  """Input config for training."""
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()


@dataclasses.dataclass
class PanopticSegmentationGenerator(hyperparams.Config):
  output_size: List[int] = dataclasses.field(
      default_factory=list)
  mask_binarize_threshold: float = 0.5
  score_threshold: float = 0.05
  things_class_label: int = 1
  void_class_label: int = 0
  void_instance_id: int = 0


@dataclasses.dataclass
class PanopticMaskRCNN(maskrcnn.MaskRCNN):
  """Panoptic Mask R-CNN model config."""
  segmentation_model: semantic_segmentation.SemanticSegmentationModel = (
      SEGMENTATION_MODEL(num_classes=2))
  include_mask = True
  shared_backbone: bool = True
  shared_decoder: bool = True
  stuff_classes_offset: int = 0
  generate_panoptic_masks: bool = True
  panoptic_segmentation_generator: PanopticSegmentationGenerator = PanopticSegmentationGenerator()  # pylint:disable=line-too-long


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
class PanopticQualityEvaluator(hyperparams.Config):
  """Panoptic Quality Evaluator config."""
  num_categories: int = 2
  ignored_label: int = 0
  max_instances_per_category: int = 100
  offset: int = 256 * 256 * 256
  is_thing: List[float] = dataclasses.field(
      default_factory=list)


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
  panoptic_quality_evaluator: PanopticQualityEvaluator = PanopticQualityEvaluator()  # pylint: disable=line-too-long


@exp_factory.register_config_factory('panoptic_maskrcnn_resnetfpn_coco')
def panoptic_maskrcnn_resnetfpn_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with Panoptic Mask R-CNN."""
  train_batch_size = 64
  eval_batch_size = 8
  steps_per_epoch = _COCO_TRAIN_EXAMPLES // train_batch_size
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
  num_semantic_segmentation_classes = 111

  is_thing = [False]
  for idx in range(1, num_panoptic_categories):
    is_thing.append(True if idx <= num_thing_categories else False)

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=PanopticMaskRCNNTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=PanopticMaskRCNN(
              num_classes=91, input_size=[1024, 1024, 3],
              panoptic_segmentation_generator=PanopticSegmentationGenerator(
                  output_size=[1024, 1024]),
              stuff_classes_offset=90,
              segmentation_model=SEGMENTATION_MODEL(
                  num_classes=num_semantic_segmentation_classes,
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
                                       'instances_val2017.json'),
          panoptic_quality_evaluator=PanopticQualityEvaluator(
              num_categories=num_panoptic_categories,
              ignored_label=0,
              is_thing=is_thing)),
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
