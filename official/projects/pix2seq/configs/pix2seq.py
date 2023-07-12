# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Pix2Seq configurations."""

import dataclasses
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import backbones
from official.vision.configs import common


# Vocab.

# A shared vocab among tasks and its structure -
# Special tokens: [0, 99).
# Class tokens: [100, coord_vocab_shift).
# Coordinate tokens: [coord_vocab_shift, text_vocab_shift).

PADDING_TOKEN = 0

# 10-29 reserved for task id.

FAKE_CLASS_TOKEN = 30
FAKE_TEXT_TOKEN = 30  # Same token to represent fake class and fake text.
SEPARATOR_TOKEN = 40
INVISIBLE_TOKEN = 41

BASE_VOCAB_SHIFT = 100

# Floats used to represent padding and separator in the flat list of polygon
# coords, and invisibility in the key points.
PADDING_FLOAT = -1.0
SEPARATOR_FLOAT = -2.0
INVISIBLE_FLOAT = -3.0
FLOATS = [PADDING_FLOAT, SEPARATOR_FLOAT, INVISIBLE_FLOAT]
TOKENS = [PADDING_TOKEN, SEPARATOR_TOKEN, INVISIBLE_TOKEN]
FLOAT_TO_TOKEN = dict(zip(FLOATS, TOKENS))
TOKEN_TO_FLOAT = dict(zip(TOKENS, FLOATS))

OD_ID = 10


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""

  input_path: str = ''
  tfds_name: str = ''
  tfds_split: str = 'train'
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'float32'
  decoder: common.DataDecoder = dataclasses.field(
      default_factory=common.DataDecoder
  )
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_color_jitter_strength: float = 0.0
  label_shift: int = 0


@dataclasses.dataclass
class Losses(hyperparams.Config):
  noise_bbox_weight: float = 1.0
  eos_token_weight: float = 0.1
  l2_weight_decay: float = 1e-4


@dataclasses.dataclass
class Pix2Seq(hyperparams.Config):
  """Pix2Seq model definations."""

  max_num_instances: int = 100
  hidden_size: int = 256
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  vocab_size: int = 3000
  use_cls_token: bool = False
  shared_decoder_embedding: bool = True
  decoder_output_bias: bool = True
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='resnet',
          resnet=backbones.ResNet(model_id=50, bn_trainable=False),
      )
  )
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=common.NormActivation
  )
  backbone_endpoint_name: str = '5'
  drop_path: float = 0.1
  drop_units: float = 0.1
  drop_att: float = 0.0
  norm_first: bool = True


@dataclasses.dataclass
class Pix2SeqTask(cfg.TaskConfig):
  model: Pix2Seq = dataclasses.field(default_factory=Pix2Seq)
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = 'all'  # all, backbone
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False
  coord_vocab_shift: int = 1000
  quantization_bins: int = 1000


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('pix2seq_r50_coco')
def pix2seq_r50_coco() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 128
  eval_batch_size = 16
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  train_steps = 80 * steps_per_epoch
  config = cfg.ExperimentConfig(
      task=Pix2SeqTask(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(
              COCO_INPUT_PATH_BASE, 'instances_val2017.json'
          ),
          model=Pix2Seq(
              input_size=[640, 640, 3],
              norm_activation=common.NormActivation(
                  norm_momentum=0.9,
                  norm_epsilon=1e-5,
                  use_sync_bn=True),
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)
              ),
          ),
          losses=Losses(l2_weight_decay=0.0),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=train_batch_size * 10,
              aug_scale_min=0.3,
              aug_scale_max=2.0,
              aug_color_jitter_strength=0.0
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=True,
          ),
      ),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=5 * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          max_to_keep=10,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw_experimental',
                  'adamw_experimental': {
                      'epsilon': 1.0e-08,
                      'weight_decay': 0.05,
                      'beta_1': 0.9,
                      'beta_2': 0.95,
                      'global_clipnorm': -1.0,
                  },
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.0001,
                      'end_learning_rate': 0.000001,
                      'offset': 0,
                      'power': 1.0,
                      'decay_steps': 80 * steps_per_epoch,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2 * steps_per_epoch,
                      'warmup_learning_rate': 0,
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
