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
"""BASNet configuration definition."""
import os
from typing import List, Union, Optional
import dataclasses
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.projects.basnet.configs import backbones
from official.vision.beta.projects.basnet.configs import common
from official.vision.beta.projects.basnet.configs import decoders


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 1000
  cycle_length: int = 10
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BASNetModel(hyperparams.Config):
  """BASNet model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='basnet_en', basnet_en=backbones.BASNet_En())
  decoder: decoders.Decoder = decoders.Decoder(type='basnet_de')
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class Losses(hyperparams.Config):
  label_smoothing: float = 0.1
  ignore_label: int = 255
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 0.0
  use_groundtruth_dimension: bool = True


@dataclasses.dataclass
class BASNetTask(cfg.TaskConfig):
  """The model config."""
  model: BASNetModel = BASNetModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  gradient_clip_norm: float = 0.0
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'backbone'  # all, backbone, and/or decoder


@exp_factory.register_config_factory('basnet')
def basnet() -> cfg.ExperimentConfig:
  """BASNet general."""
  return cfg.ExperimentConfig(
      task=BASNetModel(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

# DUTS Dataset
DUTS_TRAIN_EXAMPLES = 21106
DUTS_VAL_EXAMPLES = 5019
DUTS_INPUT_PATH_BASE_TR = '/home/datasets/DUTS/DUTS_TR_hflip_TFRecords/'
DUTS_INPUT_PATH_BASE_VAL = '/home/datasets/DUTS/DUTS_TE_TFRecords/'



@exp_factory.register_config_factory('basnet_duts')
def basnet_duts() -> cfg.ExperimentConfig:
  """Image segmentation on duts with basnet."""
  train_batch_size = 16
  eval_batch_size = 16
  steps_per_epoch = DUTS_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=BASNetTask(
          model=BASNetModel(
              input_size=[224, 224, 3],   # Resize to 256, 256
              backbone=backbones.Backbone(
                  type='basnet_en', basnet_en=backbones.BASNet_En(
                      )),
              decoder=decoders.Decoder(
                  type='basnet_de', basnet_de=decoders.BASNet_De(
                      )),
              norm_activation=common.NormActivation(
                  activation='relu',
                  norm_momentum=0.99,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=0),
          train_data=DataConfig(
              input_path=os.path.join(DUTS_INPUT_PATH_BASE_TR, 'DUTS-TR-*'),
              is_training=True,
              global_batch_size=train_batch_size,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(DUTS_INPUT_PATH_BASE_VAL, 'DUTS-TE-*'),
              is_training=False,
              global_batch_size=eval_batch_size,
          ),
          #init_checkpoint='',
          #init_checkpoint_modules='backbone'
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=300 * steps_per_epoch,
          validation_steps=DUTS_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
                  'adam': {
                      'beta_1': 0.9,
                      'beta_2': 0.999,
                      'epsilon': 1e-8,
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {'boundaries': [200*steps_per_epoch],
                               'values': [0.001, 0.001]}
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config
