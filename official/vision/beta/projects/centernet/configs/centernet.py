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
# ==============================================================================

"""CenterNet configuration definition."""
# Import libraries
import os
import dataclasses
from typing import List, Optional

from official.vision.beta.projects.centernet.configs import backbones
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import common
from official.modeling.hyperparams import config_definitions as cfg


@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False


@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''


@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = TfExampleDecoder()
  label_map_decoder: TfExampleDecoderLabelMap = TfExampleDecoderLabelMap()


# dataset parser
@dataclasses.dataclass
class Parser(hyperparams.Config):
  max_num_instances: int = 128
  bgr_ordering: bool = True
  aug_rand_hflip: bool = True
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_saturation: bool = False
  aug_rand_brightness: bool = False
  aug_rand_zoom: bool = False
  aug_rand_hue: bool = False
  channel_means: List[int] = dataclasses.field(
      default_factory=lambda: [104.01362025, 114.03422265, 119.9165958])
  channel_stds: List[int] = dataclasses.field(
      default_factory=lambda: [73.6027665, 69.89082075, 70.9150767])


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 32
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True


@dataclasses.dataclass
class DetectionLoss(hyperparams.Config):
  detection_weight: float = 1.0
  corner_pull_weight: float = 0.1  # alpha
  corner_push_weight: float = 0.1  # beta
  offset_weight: float = 1.0  # gamma
  scale_weight: float = 0.1


@dataclasses.dataclass
class SegmentationLoss(hyperparams.Config):
  pass


@dataclasses.dataclass
class Losses(hyperparams.Config):
  detection: DetectionLoss = DetectionLoss()
  segmentation: SegmentationLoss = SegmentationLoss()
  use_gaussian_bump: bool = True
  gaussian_rad: int = -1
  gaussian_iou: float = 0.7
  class_offset: int = 1


@dataclasses.dataclass
class CenterNetHead(hyperparams.Config):
  heatmap_bias: float = -2.19
  num_inputs: int = 2


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
  num_classes: int = 90
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='hourglass', hourglass=backbones.Hourglass(model_id=52))
  head: CenterNetHead = CenterNetHead()
  detection_generator: CenterNetDetectionGenerator = CenterNetDetectionGenerator()
  norm_activation: common.NormActivation = common.NormActivation(
      norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)


@dataclasses.dataclass
class CenterNetDetection(hyperparams.Config):
  use_centers: bool = True
  # corner is not supported currently
  use_corners: bool = False
  predict_3d: bool = False


@dataclasses.dataclass
class CenterNetSubTasks(hyperparams.Config):
  detection: CenterNetDetection = CenterNetDetection()
  # Placeholder for extending for additional task
  # segmentation: bool = False
  # pose: bool = False
  # kp_detection: bool = False
  # reid: bool = False
  # temporal: bool = False


@dataclasses.dataclass
class CenterNetTask(cfg.TaskConfig):
  model: CenterNetModel = CenterNetModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  subtasks: CenterNetSubTasks = CenterNetSubTasks()
  losses: Losses = Losses()
  gradient_clip_norm: float = 0.0
  per_category_metrics: bool = False
  weight_decay: float = 5e-4
  # Load checkpoints
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'
  init_checkpoint_source: str = 'TFVision'  # ODAPI, Extremenet or TFVision
  annotation_file: Optional[str] = None
  # For checkpoints from ODAPI or Extremenet
  checkpoint_backbone_name: str = 'hourglass104_512'
  checkpoint_head_name: str = 'detection_2d'
  
  def get_output_length_dict(self):
    lengths = {}
    assert self.subtasks.detection is not None or self.subtasks.kp_detection \
           or self.subtasks.segmentation, "You must specify at least one " \
                                          "subtask to CenterNet"
    
    if self.subtasks.detection:
      # TODO: locations of the ground truths will also be passed in from the
      # data pipeline which need to be mapped accordingly
      assert (self.subtasks.detection.use_centers and
              not self.subtasks.detection.use_corners,
              "Use corner is not supported currently.")
      if self.subtasks.detection.use_centers:
        lengths.update({
            'ct_heatmaps': self.model.num_classes,
            'ct_offset': 2,
        })
      
      if self.subtasks.detection.predict_3d:
        lengths.update({
            'depth': 1,
            'orientation': 8
        })
    
    return lengths


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('centernet_hourglass_coco')
def centernet_custom() -> cfg.ExperimentConfig:
  """COCO object detection with CenterNet."""
  train_batch_size = 1
  eval_batch_size = 1
  base_default = 1200000
  num_batches = int(1200000 * 64 / train_batch_size)
  
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
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
          steps_per_loop=2000,
          summary_interval=8000,
          checkpoint_interval=10000,
          train_steps=num_batches,
          validation_steps=625,
          validation_interval=10,
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
                          int(400000 / base_default * num_batches),
                          int(450000 / base_default * num_batches)
                      ],
                      'values': [
                          0.00261 * train_batch_size / 64,
                          0.000261 * train_batch_size / 64,
                          0.0000261 * train_batch_size / 64
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1000 * 64 // num_batches,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  
  return config
