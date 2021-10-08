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
"""Semantic segmentation configuration definition."""

import dataclasses
from typing import List, Optional, Union
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.projects.volumetric_models.configs import backbones
from official.projects.volumetric_models.configs import decoders
from official.vision.beta.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  output_size: List[int] = dataclasses.field(default_factory=list)
  input_size: List[int] = dataclasses.field(default_factory=list)
  num_classes: int = 0
  num_channels: int = 1
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  label_dtype: str = 'float32'
  image_field_key: str = 'image/encoded'
  label_field_key: str = 'image/class/label'
  shuffle_buffer_size: int = 1000
  cycle_length: int = 10
  drop_remainder: bool = False
  file_type: str = 'tfrecord'


@dataclasses.dataclass
class SegmentationHead3D(hyperparams.Config):
  """Segmentation head config."""
  num_classes: int = 0
  level: int = 1
  num_convs: int = 0
  num_filters: int = 256
  upsample_factor: int = 1
  output_logits: bool = True
  use_batch_normalization: bool = True


@dataclasses.dataclass
class SemanticSegmentationModel3D(hyperparams.Config):
  """Semantic segmentation model config."""
  num_classes: int = 0
  num_channels: int = 1
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 6
  head: SegmentationHead3D = SegmentationHead3D()
  backbone: backbones.Backbone = backbones.Backbone(
      type='unet_3d', unet_3d=backbones.UNet3D())
  decoder: decoders.Decoder = decoders.Decoder(
      type='unet_3d_decoder', unet_3d_decoder=decoders.UNet3DDecoder())
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class Losses(hyperparams.Config):
  # Supported `loss_type` are `adaptive` and `generalized`.
  loss_type: str = 'adaptive'
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  report_per_class_metric: bool = False  # Whether to report per-class metrics.


@dataclasses.dataclass
class SemanticSegmentation3DTask(cfg.TaskConfig):
  """The model config."""
  model: SemanticSegmentationModel3D = SemanticSegmentationModel3D()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  evaluation: Evaluation = Evaluation()
  train_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  eval_input_partition_dims: List[int] = dataclasses.field(default_factory=list)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder


@exp_factory.register_config_factory('seg_unet3d_test')
def seg_unet3d_test() -> cfg.ExperimentConfig:
  """Image segmentation on a dummy dataset with 3D UNet for testing purpose."""
  train_batch_size = 2
  eval_batch_size = 2
  steps_per_epoch = 10
  config = cfg.ExperimentConfig(
      task=SemanticSegmentation3DTask(
          model=SemanticSegmentationModel3D(
              num_classes=2,
              input_size=[32, 32, 32],
              num_channels=2,
              backbone=backbones.Backbone(
                  type='unet_3d', unet_3d=backbones.UNet3D(model_id=2)),
              decoder=decoders.Decoder(
                  type='unet_3d_decoder',
                  unet_3d_decoder=decoders.UNet3DDecoder(model_id=2)),
              head=SegmentationHead3D(num_convs=0, num_classes=2),
              norm_activation=common.NormActivation(
                  activation='relu', use_sync_bn=False)),
          train_data=DataConfig(
              input_path='train.tfrecord',
              num_classes=2,
              input_size=[32, 32, 32],
              num_channels=2,
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              input_path='val.tfrecord',
              num_classes=2,
              input_size=[32, 32, 32],
              num_channels=2,
              is_training=False,
              global_batch_size=eval_batch_size),
          losses=Losses(loss_type='adaptive')),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=10,
          validation_steps=10,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
              },
              'learning_rate': {
                  'type': 'constant',
                  'constant': {
                      'learning_rate': 0.000001
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
