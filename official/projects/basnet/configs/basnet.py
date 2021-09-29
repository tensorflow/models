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

"""BASNet configuration definition."""
import dataclasses
import os
from typing import List, Optional, Union

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
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
  cycle_length: int = 10
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_rand_hflip: bool = True
  file_type: str = 'tfrecord'


@dataclasses.dataclass
class BASNetModel(hyperparams.Config):
  """BASNet model config."""
  input_size: List[int] = dataclasses.field(default_factory=list)
  use_bias: bool = False
  norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class Losses(hyperparams.Config):
  label_smoothing: float = 0.1
  ignore_label: int = 0  # will be treated as background
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
DUTS_TRAIN_EXAMPLES = 10553
DUTS_VAL_EXAMPLES = 5019
DUTS_INPUT_PATH_BASE_TR = 'DUTS_DATASET'
DUTS_INPUT_PATH_BASE_VAL = 'DUTS_DATASET'


@exp_factory.register_config_factory('basnet_duts')
def basnet_duts() -> cfg.ExperimentConfig:
  """Image segmentation on duts with basnet."""
  train_batch_size = 64
  eval_batch_size = 16
  steps_per_epoch = DUTS_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=BASNetTask(
          model=BASNetModel(
              input_size=[None, None, 3],
              use_bias=True,
              norm_activation=common.NormActivation(
                  activation='relu',
                  norm_momentum=0.99,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=0),
          train_data=DataConfig(
              input_path=os.path.join(DUTS_INPUT_PATH_BASE_TR,
                                      'tf_record_train'),
              file_type='tfrecord',
              crop_size=[224, 224],
              output_size=[256, 256],
              is_training=True,
              global_batch_size=train_batch_size,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(DUTS_INPUT_PATH_BASE_VAL,
                                      'tf_record_test'),
              file_type='tfrecord',
              output_size=[256, 256],
              is_training=False,
              global_batch_size=eval_batch_size,
          ),
          init_checkpoint='gs://cloud-basnet-checkpoints/basnet_encoder_imagenet/ckpt-340306',
          init_checkpoint_modules='backbone'
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
                  'type': 'constant',
                  'constant': {'learning_rate': 0.001}
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config
