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

"""CenterNet configuration definition."""

import dataclasses
import os
from typing import List, Optional, Tuple

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.projects.centernet.configs import backbones
from official.vision.configs import common


TfExampleDecoderLabelMap = common.TfExampleDecoderLabelMap


@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False


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
class Parser(hyperparams.Config):
  """Config for parser."""
  bgr_ordering: bool = True
  aug_rand_hflip: bool = True
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_saturation: bool = False
  aug_rand_brightness: bool = False
  aug_rand_hue: bool = False
  aug_rand_contrast: bool = False
  odapi_augmentation: bool = False
  channel_means: Tuple[float, float, float] = dataclasses.field(
      default_factory=lambda: (104.01362025, 114.03422265, 119.9165958))
  channel_stds: Tuple[float, float, float] = dataclasses.field(
      default_factory=lambda: (73.6027665, 69.89082075, 70.9150767))


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 32
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = dataclasses.field(default_factory=DataDecoder)
  parser: Parser = dataclasses.field(default_factory=Parser)
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True


@dataclasses.dataclass
class DetectionLoss(hyperparams.Config):
  object_center_weight: float = 1.0
  offset_weight: float = 1.0
  scale_weight: float = 0.1


@dataclasses.dataclass
class Losses(hyperparams.Config):
  detection: DetectionLoss = dataclasses.field(default_factory=DetectionLoss)
  gaussian_iou: float = 0.7
  class_offset: int = 1


@dataclasses.dataclass
class CenterNetHead(hyperparams.Config):
  heatmap_bias: float = -2.19
  input_levels: List[str] = dataclasses.field(
      default_factory=lambda: ['2_0', '2'])


@dataclasses.dataclass
class CenterNetDetectionGenerator(hyperparams.Config):
  max_detections: int = 100
  peak_error: float = 1e-6
  peak_extract_kernel_size: int = 3
  class_offset: int = 1
  use_nms: bool = False
  nms_pre_thresh: float = 0.1
  nms_thresh: float = 0.4
  use_reduction_sum: bool = True


@dataclasses.dataclass
class CenterNetModel(hyperparams.Config):
  """Config for centernet model."""
  num_classes: int = 90
  max_num_instances: int = 128
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='hourglass', hourglass=backbones.Hourglass(model_id=52)
      )
  )
  head: CenterNetHead = dataclasses.field(default_factory=CenterNetHead)
  # pylint: disable=line-too-long
  detection_generator: CenterNetDetectionGenerator = dataclasses.field(
      default_factory=CenterNetDetectionGenerator
  )
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(  # pylint: disable=g-long-lambda
          norm_momentum=0.1, norm_epsilon=1e-5, use_sync_bn=True
      )
  )


@dataclasses.dataclass
class CenterNetDetection(hyperparams.Config):
  # use_center is the only option implemented currently.
  use_centers: bool = True


@dataclasses.dataclass
class CenterNetSubTasks(hyperparams.Config):
  detection: CenterNetDetection = dataclasses.field(
      default_factory=CenterNetDetection
  )


@dataclasses.dataclass
class CenterNetTask(cfg.TaskConfig):
  """Config for centernet task."""
  model: CenterNetModel = dataclasses.field(default_factory=CenterNetModel)
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=False)
  )
  subtasks: CenterNetSubTasks = dataclasses.field(
      default_factory=CenterNetSubTasks
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  gradient_clip_norm: float = 10.0
  per_category_metrics: bool = False
  weight_decay: float = 5e-4
  # Load checkpoints
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'
  annotation_file: Optional[str] = None

  def get_output_length_dict(self):
    task_outputs = {}
    if self.subtasks.detection and self.subtasks.detection.use_centers:
      task_outputs.update({
          'ct_heatmaps': self.model.num_classes,
          'ct_offset': 2,
          'ct_size': 2
      })
    else:
      raise ValueError('Detection with center point is only available ')
    return task_outputs


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('centernet_hourglass_coco')
def centernet_hourglass_coco() -> cfg.ExperimentConfig:
  """COCO object detection with CenterNet."""
  train_batch_size = 128
  eval_batch_size = 8
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size

  config = cfg.ExperimentConfig(
      task=CenterNetTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=CenterNetModel(),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(),
              shuffle_buffer_size=2),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=2),
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=150 * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
                  'adam': {
                      'epsilon': 1e-7
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.001,
                      'decay_steps': 150 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
