# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""YOLOv7 configuration definition."""
import dataclasses
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.projects.yolo import optimization
from official.projects.yolo.configs import backbones
from official.projects.yolo.configs import decoders
from official.projects.yolo.configs.yolo import AnchorBoxes
from official.projects.yolo.configs.yolo import DataConfig
from official.projects.yolo.configs.yolo import Mosaic
from official.projects.yolo.configs.yolo import Parser
from official.projects.yolo.configs.yolo import YoloDetectionGenerator
from official.vision.configs import common


# pytype: disable=annotation-type-mismatch

MIN_LEVEL = 3
MAX_LEVEL = 5
GLOBAL_SEED = 1000


def _build_dict(min_level, max_level, value):
  vals = {str(key): value for key in range(min_level, max_level + 1)}
  vals['all'] = None
  return lambda: vals


def _build_path_scales(min_level, max_level):
  return lambda: {str(key): 2**key for key in range(min_level, max_level + 1)}


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False
  coco91_to_80: bool = True


@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''


@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = dataclasses.field(
      default_factory=TfExampleDecoder
  )
  label_map_decoder: TfExampleDecoderLabelMap = dataclasses.field(
      default_factory=TfExampleDecoderLabelMap
  )


@dataclasses.dataclass
class YoloV7Head(hyperparams.Config):
  """Parameterization for the YOLO Head."""
  num_anchors: int = 3
  use_separable_conv: bool = False


@dataclasses.dataclass
class YoloV7Loss(hyperparams.Config):
  """Config or YOLOv7 loss."""
  alpha: float = 0.0
  gamma: float = 0.0
  box_weight: float = 0.05
  obj_weight: float = 0.7
  cls_weight: float = 0.3
  label_smoothing: float = 0.0
  anchor_threshold: float = 4.0
  iou_mix_ratio: float = 1.0
  auto_balance: bool = False
  use_ota: bool = True


@dataclasses.dataclass
class Box(hyperparams.Config):
  box: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class YoloV7(hyperparams.Config):
  input_size: Optional[List[int]] = dataclasses.field(
      default_factory=lambda: [640, 640, 3]
  )
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='yolov7', yolov7=backbones.YoloV7(model_id='yolov7')
      )
  )
  decoder: decoders.Decoder = dataclasses.field(
      default_factory=lambda: decoders.Decoder(  # pylint: disable=g-long-lambda
          type='yolov7', yolo_decoder=decoders.YoloV7(model_id='yolov7')
      )
  )
  head: YoloV7Head = dataclasses.field(default_factory=YoloV7Head)
  detection_generator: YoloDetectionGenerator = dataclasses.field(
      default_factory=lambda: YoloDetectionGenerator(  # pylint: disable=g-long-lambda
          box_type=_build_dict(MIN_LEVEL, MAX_LEVEL, 'scaled')(),
          scale_xy=_build_dict(MIN_LEVEL, MAX_LEVEL, 2.0)(),
          path_scales=_build_path_scales(MIN_LEVEL, MAX_LEVEL)(),
          nms_version='iou',
          iou_thresh=0.001,
          nms_thresh=0.7,
          max_boxes=300,
          pre_nms_points=5000,
      )
  )
  loss: YoloV7Loss = dataclasses.field(default_factory=YoloV7Loss)
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(  # pylint: disable=g-long-lambda
          activation='swish',
          use_sync_bn=True,
          norm_momentum=0.99,
          norm_epsilon=0.001,
      )
  )
  num_classes: int = 80
  min_level: int = 3
  max_level: int = 5
  anchor_boxes: AnchorBoxes = dataclasses.field(default_factory=AnchorBoxes)


@dataclasses.dataclass
class YoloV7Task(cfg.TaskConfig):
  per_category_metrics: bool = False
  smart_bias_lr: float = 0.0
  model: YoloV7 = dataclasses.field(default_factory=YoloV7)
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=False)
  )
  weight_decay: float = 0.0
  annotation_file: Optional[str] = None
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = (
      'all'  # all, backbone, and/or decoder
  )
  gradient_clip_norm: float = 0.0
  seed = GLOBAL_SEED
  # Sets maximum number of boxes to be evaluated by coco eval api.
  max_num_eval_detections: int = 100


COCO_INPUT_PATH_BASE = (
    '/readahead/200M/placer/prod/home/tensorflow-performance-data/datasets/coco'
)
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('yolov7')
def yolov7() -> cfg.ExperimentConfig:
  """YOLOv7 general config."""
  return cfg.ExperimentConfig(
      task=YoloV7Task(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ],
  )


@exp_factory.register_config_factory('coco_yolov7')
def coco_yolov7() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv7."""
  train_batch_size = 256
  eval_batch_size = 256
  train_epochs = 300
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  validation_interval = 5
  warmup_steps = 3 * steps_per_epoch

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=YoloV7Task(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          annotation_file=None,
          weight_decay=0.0,
          model=YoloV7(
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.03,
                  norm_epsilon=0.001,
                  use_sync_bn=True),
              head=YoloV7Head(),
              loss=YoloV7Loss(),
              anchor_boxes=AnchorBoxes(
                  anchors_per_scale=3,
                  boxes=[
                      Box(box=[12, 16]),
                      Box(box=[19, 36]),
                      Box(box=[40, 28]),
                      Box(box=[36, 75]),
                      Box(box=[76, 55]),
                      Box(box=[72, 146]),
                      Box(box=[142, 110]),
                      Box(box=[192, 243]),
                      Box(box=[459, 401]),
                  ],
              ),
          ),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              dtype='float32',
              parser=Parser(
                  max_num_instances=300,
                  letter_box=True,
                  random_flip=True,
                  random_pad=False,
                  jitter=0.0,
                  aug_scale_min=1.0,
                  aug_scale_max=1.0,
                  aug_rand_translate=0.2,
                  aug_rand_saturation=0.7,
                  aug_rand_brightness=0.4,
                  aug_rand_hue=0.015,
                  aug_rand_angle=0.0,
                  aug_rand_perspective=0.0,
                  use_tie_breaker=True,
                  best_match_only=True,
                  anchor_thresh=4.0,
                  area_thresh=0.0,
                  mosaic=Mosaic(
                      mosaic_frequency=1.0,
                      mosaic9_frequency=0.2,
                      mixup_frequency=0.15,
                      mosaic_crop_mode='scale',
                      mosaic_center=0.25,
                      mosaic9_center=0.33,
                      aug_scale_min=0.1,
                      aug_scale_max=1.9,
                  ),
              ),
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=True,
              dtype='float32',
              parser=Parser(
                  max_num_instances=300,
                  letter_box=True,
                  use_tie_breaker=True,
                  best_match_only=True,
                  anchor_thresh=4.0,
                  area_thresh=0.0,
              ),
          ),
          smart_bias_lr=0.1,
      ),
      trainer=cfg.TrainerConfig(
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          best_checkpoint_metric_comp='higher',
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema': {
                  'average_decay': 0.9999,
                  'trainable_weights_only': False,
                  'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'sgd_torch',
                  'sgd_torch': {
                      'momentum': 0.937,
                      'momentum_start': 0.8,
                      'nesterov': True,
                      'warmup_steps': warmup_steps,
                      # Scale up the weight decay by batch size.
                      'weight_decay': 0.0005 * train_batch_size / 64,
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.01,
                      'alpha': 0.1,
                      'decay_steps': train_epochs * steps_per_epoch,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': warmup_steps,
                      'warmup_learning_rate': 0.0,
                  },
              },
          }),
      ),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ],
  )

  return config


@exp_factory.register_config_factory('coco_yolov7tiny')
def coco_yolov7_tiny() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv7-tiny."""
  config = coco_yolov7()
  config.task.model.input_size = [416, 416, 3]
  config.task.model.backbone.yolov7.model_id = 'yolov7-tiny'
  config.task.model.decoder.yolov7.model_id = 'yolov7-tiny'
  config.task.model.norm_activation.activation = 'leaky'
  config.task.model.anchor_boxes.boxes = [
      Box(box=[10, 13]),
      Box(box=[16, 30]),
      Box(box=[33, 23]),
      Box(box=[30, 61]),
      Box(box=[62, 45]),
      Box(box=[59, 119]),
      Box(box=[116, 90]),
      Box(box=[156, 198]),
      Box(box=[373, 326]),
  ]

  config.task.model.loss.cls_weight = 0.5
  config.task.model.loss.obj_weight = 1.0
  config.task.train_data.parser.aug_rand_translate = 0.1
  config.task.train_data.parser.mosaic.mixup_frequency = 0.05
  config.task.train_data.parser.mosaic.aug_scale_min = 0.5
  config.task.train_data.parser.mosaic.aug_scale_max = 1.5
  config.trainer.optimizer_config.learning_rate.cosine.alpha = 0.01
  return config


@exp_factory.register_config_factory('coco91_yolov7tiny')
def coco91_yolov7_tiny() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv7-tiny using 91 classes."""
  config = coco_yolov7_tiny()
  config.task.model.num_classes = 91
  config.task.model.decoder.yolov7.use_separable_conv = True
  config.task.model.head.use_separable_conv = True
  config.task.train_data.coco91_to_80 = False
  config.task.validation_data.coco91_to_80 = False
  return config


@exp_factory.register_config_factory('coco_yolov7x')
def coco_yolov7x() -> cfg.ExperimentConfig:
  config = coco_yolov7()
  config.task.model.backbone.yolov7.model_id = 'yolov7x'
  config.task.model.decoder.yolov7.model_id = 'yolov7x'
  return config
