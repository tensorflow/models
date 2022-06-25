# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""DETR configurations."""

import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.detr import optimization
import os
from official.vision.configs import common


# pylint: disable=missing-class-docstring
# Keep for backward compatibility.
@dataclasses.dataclass
class TfExampleDecoder(common.TfExampleDecoder):
  """A simple TF Example decoder config."""


# Keep for backward compatibility.
@dataclasses.dataclass
class TfExampleDecoderLabelMap(common.TfExampleDecoderLabelMap):
  """TF Example decoder with label map config."""


# Keep for backward compatibility.
@dataclasses.dataclass
class DataDecoder(common.DataDecoder):
  """Data decoder config."""

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: common.DataDecoder = common.DataDecoder()
  #parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'

@dataclasses.dataclass
class DetectionConfig(cfg.TaskConfig):
  """The translation task config."""
  annotation_file: str = ''
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()
  lambda_cls: float = 1.0
  lambda_box: float = 5.0
  lambda_giou: float = 2.0

  #init_ckpt: str = ''
  init_checkpoint: str = 'gs://ghpark-imagenet-tfrecord/ckpt/resnet50_imagenet'
  init_checkpoint_modules: str = 'backbone'
  #num_classes: int = 81  # 0: background
  num_classes: int = 91  # 0: background
  background_cls_weight: float = 0.1
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6

  # Make DETRConfig.
  num_queries: int = 100
  num_hidden: int = 256
  per_category_metrics: bool = False

COCO_INPUT_PATH_BASE = 'gs://ghpark-tfrecords/coco'
#COCO_TRAIN_EXAMPLES = 118287
COCO_TRAIN_EXAMPLES = 960
COCO_VAL_EXAMPLES = 5000

@exp_factory.register_config_factory('detr_coco')
def detr_coco() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 32
  eval_batch_size = 64
  num_train_data = 118287
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  train_steps = 300 * steps_per_epoch  # 500 epochs
  decay_at = train_steps - 100 * steps_per_epoch  # 400 epochs
  config = cfg.ExperimentConfig(
      task=DetectionConfig(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
          )
      ),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          validation_interval=5*steps_per_epoch,
          max_to_keep=1,
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'detr_adamw',
                  'detr_adamw': {
                      'weight_decay_rate': 1e-4,
                      'global_clipnorm': 0.1,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [decay_at],
                      'values': [0.0001, 1.0e-05]
                  }
              },
              })
          ),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
