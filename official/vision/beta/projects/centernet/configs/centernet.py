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
"""CenterNet configuration definition."""
# Import libraries
import dataclasses
from typing import ClassVar, Dict, List, Optional, Tuple, Union

from official.vision.beta.projects.centernet.configs import backbones
from official.core import exp_factory
from official.modeling import hyperparams, optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common


# default param classes
@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
  
  @property
  def input_size(self):
    if self._input_size is None:
      return [None, None, 3]
    else:
      return self._input_size
  
  @input_size.setter
  def input_size(self, input_size):
    self._input_size = input_size
  
  @property
  def backbone(self):
    if isinstance(self.base, str):
      # TODO: remove the automatic activation setter
      # self.norm_activation.activation = Yolo._DEFAULTS[self.base].activation
      return CenterNet._DEFAULTS[self.base].backbone
    else:
      return self.base.backbone
  
  @backbone.setter
  def backbone(self, val):
    self.base.backbone = val
  
  @property
  def decoder(self):
    if isinstance(self.base, str):
      return CenterNet._DEFAULTS[self.base].decoder
    else:
      return self.base.decoder
  
  @decoder.setter
  def decoder(self, val):
    self.base.decoder = val
  
  @property
  def odapi_weights_file(self):
    if isinstance(self.base, str):
      return CenterNet._DEFAULTS[self.base].odapi_weights
    else:
      return self.base.odapi_weights
  
  @property
  def extremenet_weights_file(self):
    if isinstance(self.base, str):
      return CenterNet._DEFAULTS[self.base].extremenet_weights
    else:
      return self.base.extremenet_weights


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
  image_w: int = 512
  image_h: int = 512
  max_num_instances: int = 128
  bgr_ordering: bool = True
  channel_means: List[int] = dataclasses.field(
      default_factory=lambda: [104.01362025, 114.03422265, 119.9165958])
  channel_stds: List[int] = dataclasses.field(
      default_factory=lambda: [73.6027665, 69.89082075, 70.9150767])
  dtype: str = 'float32'


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = 'gs://tensorflow2/coco_records/val/2017*'
  tfds_name: str = None  # 'coco'
  tfds_split: str = None  # 'train' #'val'
  global_batch_size: int = 32
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = False
  cache: bool = False


class Loss(hyperparams.Config):
  pass


@dataclasses.dataclass
class DetectionLoss(Loss):
  detection_weight: float = 1.0
  corner_pull_weight: float = 0.1  # alpha
  corner_push_weight: float = 0.1  # beta
  offset_weight: float = 1.0  # gamma
  scale_weight: float = 0.1


@dataclasses.dataclass
class SegmentationLoss(Loss):
  pass


@dataclasses.dataclass
class Losses(hyperparams.Config):
  detection: DetectionLoss = DetectionLoss()
  segmentation: SegmentationLoss = SegmentationLoss()
  image_h: int = 512
  image_w: int = 512
  output_dims: int = 128
  max_num_instances: int = 128
  use_gaussian_bump: bool = True
  gaussian_rad: int = -1
  gaussian_iou: float = 0.7
  class_offset: int = 1
  dtype: str = 'float32'


@dataclasses.dataclass
class CenterNetDecoder(hyperparams.Config):
  heatmap_bias: float = -2.19


@dataclasses.dataclass
class CenterNetLayer(hyperparams.Config):
  max_detections: int = 100
  peak_error: float = 1e-6
  peak_extract_kernel_size: int = 3
  class_offset: int = 1
  net_down_scale: int = 4
  input_image_dims: int = 512
  use_nms: bool = False
  nms_pre_thresh: float = 0.1
  nms_thresh: float = 0.4
  use_reduction_sum: bool = True


@dataclasses.dataclass
class CenterNetDetection(hyperparams.Config):
  use_centers: bool = True
  use_corners: bool = False
  predict_3d: bool = False


@dataclasses.dataclass
class CenterNetSubTasks(hyperparams.Config):
  detection: CenterNetDetection = CenterNetDetection()
  # kp_detection: bool = False
  segmentation: bool = False
  # pose: bool = False
  # reid: bool = False
  # temporal: bool = False


@dataclasses.dataclass
class CenterNetBase(hyperparams.OneOfConfig):
  backbone: backbones.Backbone = backbones.Backbone(type='hourglass')
  decoder: CenterNetDecoder = CenterNetDecoder()
  odapi_weights: str = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'
  extremenet_weights: str = 'D:\\weights\extremenet'
  backbone_name: str = 'hourglass104_512'
  decoder_name: str = 'detection_2d'


@dataclasses.dataclass
class CenterNet(ModelConfig):
  base: Union[str, CenterNetBase] = CenterNetBase()
  num_classes: int = 90
  _input_size: Optional[List[int]] = None
  filter: CenterNetLayer = CenterNetLayer()


@dataclasses.dataclass
class CenterNetTask(cfg.TaskConfig):
  model: CenterNet = CenterNet()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  subtasks: CenterNetSubTasks = CenterNetSubTasks()
  losses: Losses = Losses()
  
  per_category_metrics: bool = False
  weight_decay: float = 5e-4
  
  init_checkpoint: str = None
  annotation_file: Optional[str] = None
  gradient_clip_norm: float = 0.0
  load_odapi_weights: bool = True
  load_extremenet_weights: bool = False
  
  def _get_output_length_dict(self):
    lengths = {}
    assert self.subtasks.detection is not None or self.subtasks.kp_detection \
           or self.subtasks.segmentation, "You must specify at least one " \
                                          "subtask to CenterNet"
    
    if self.subtasks.detection:
      # TODO: locations of the ground truths will also be passed in from the
      # data pipeline which need to be mapped accordingly
      assert self.subtasks.detection.use_centers or \
             self.subtasks.detection.use_corners, "Cannot use CenterNet without " \
                                                  "heatmaps"
      if self.subtasks.detection.use_centers:
        lengths.update({
            'ct_heatmaps': self.model.num_classes,
            'ct_offset': 2,
        })
        if not self.subtasks.detection.use_corners:
          lengths['ct_size'] = 2
      
      if self.subtasks.detection.use_corners:
        lengths.update({
            'tl_heatmaps': self.model.num_classes,
            'tl_offset': 2,
            'br_heatmaps': self.model.num_classes,
            'br_offset': 2
        })
      
      if self.subtasks.detection.predict_3d:
        lengths.update({
            'depth': 1,
            'orientation': 8
        })
    
    if self.subtasks.segmentation:
      lengths['seg_heatmaps'] = self.model.num_classes
    
    # if self.subtasks.pose:
    #   lengths.update({
    #     'pose_heatmaps': 17,
    #     'joint_locs': 17 * 2,
    #     'joint_offset': 2
    #   })
    
    return lengths


@exp_factory.register_config_factory('centernet_custom')
def centernet_custom() -> cfg.ExperimentConfig:
  """COCO object detection with CenterNet."""
  train_batch_size = 1
  eval_batch_size = 1
  base_default = 1200000
  num_batches = 1200000 * 64 / train_batch_size
  
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          #            mixed_precision_dtype='float16',
          #            loss_scale='dynamic',
          num_gpus=2),
      task=CenterNetTask(
          model=CenterNet(),
          train_data=DataConfig(  # input_path=os.path.join(
              # COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(),
              shuffle_buffer_size=2),
          validation_data=DataConfig(
              # input_path=os.path.join(COCO_INPUT_PATH_BASE,
              #                        'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=2)),
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


@exp_factory.register_config_factory('centernet_tpu')
def centernet_tpu() -> cfg.ExperimentConfig:
  """COCO object detection with CenterNet."""
  train_batch_size = 1
  eval_batch_size = 8
  base_default = 1200000
  num_batches = 1200000 * 64 / train_batch_size
  
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=CenterNetTask(
          model=CenterNet(),
          train_data=DataConfig(  # input_path=os.path.join(
              # COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(),
              shuffle_buffer_size=10000),
          validation_data=DataConfig(
              # input_path=os.path.join(COCO_INPUT_PATH_BASE,
              #                        'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=100)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=2000,
          summary_interval=8000,
          checkpoint_interval=10000,
          train_steps=num_batches,
          validation_steps=625,
          validation_interval=1,
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
