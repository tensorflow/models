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

# pylint: disable=line-too-long
# type: ignore
"""Semantic segmentation configuration definition for AutoML built models."""

import dataclasses
import os
from typing import Any, List, Mapping, Optional

# Import libraries

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import backbones
from official.vision.beta.configs import semantic_segmentation as base_cfg

# ADE 20K Dataset
ADE20K_TRAIN_EXAMPLES = 20210
ADE20K_VAL_EXAMPLES = 2000
ADE20K_INPUT_PATH_BASE = 'gs://**/ADE20K'

PRETRAINED_CKPT_PATH_BASE = 'gs://**/placeholder_for_edgetpu_models/pretrained_checkpoints'

BACKBONE_PRETRAINED_CHECKPOINT = {
    'autoseg_edgetpu_backbone_xs':
        PRETRAINED_CKPT_PATH_BASE +
        '/autoseg_edgetpu_backbone_xs/ckpt-171600',
    'autoseg_edgetpu_backbone_s':
        PRETRAINED_CKPT_PATH_BASE +
        '/autoseg_edgetpu_backbone_s/ckpt-171600',
    'autoseg_edgetpu_backbone_m':
        PRETRAINED_CKPT_PATH_BASE +
        '/autoseg_edgetpu_backbone_m/ckpt-171600',
}


@dataclasses.dataclass
class BiFPNHeadConfig(hyperparams.Config):
  """BiFPN-based segmentation head config."""
  min_level: int = 3
  max_level: int = 8
  fpn_num_filters: int = 96


@dataclasses.dataclass
class Losses(hyperparams.Config):
  label_smoothing: float = 0.0
  ignore_label: int = 255
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 0.0
  use_groundtruth_dimension: bool = True
  top_k_percent_pixels: float = 1.0


@dataclasses.dataclass
class AutosegEdgeTPUModelConfig(hyperparams.Config):
  """Autoseg-EdgeTPU segmentation model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone()
  head: BiFPNHeadConfig = BiFPNHeadConfig()
  model_params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {  # pylint: disable=g-long-lambda
          'model_name': 'autoseg_edgetpu_backbone_s',
          'checkpoint_format': 'tf_checkpoint',
          'overrides': {
              'batch_norm': 'tpu',
              'rescale_input': False,
              'backbone_only': True,
              'resolution': 512
          }
      })


@dataclasses.dataclass
class AutosegEdgeTPUTaskConfig(base_cfg.SemanticSegmentationTask):
  """The task config inherited from the base segmentation task."""

  model: AutosegEdgeTPUModelConfig = AutosegEdgeTPUModelConfig()
  train_data: base_cfg.DataConfig = base_cfg.DataConfig(is_training=True)
  validation_data: base_cfg.DataConfig = base_cfg.DataConfig(is_training=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'backbone'  # all or backbone
  model_output_keys: Optional[List[int]] = dataclasses.field(
      default_factory=list)


def autoseg_edgetpu_experiment_config(backbone_name: str,
                                      init_backbone: bool = True
                                     ) -> cfg.ExperimentConfig:
  """Experiment using the semantic segmenatation searched model.

  Args:
    backbone_name: Name of the backbone used for this model
    init_backbone: Whether to initialize backbone from a pretrained checkpoint
  Returns:
    ExperimentConfig
  """
  epochs = 300
  train_batch_size = 64
  eval_batch_size = 32
  image_size = 512
  steps_per_epoch = ADE20K_TRAIN_EXAMPLES // train_batch_size
  train_steps = epochs * steps_per_epoch
  model_config = AutosegEdgeTPUModelConfig(
      num_classes=32, input_size=[image_size, image_size, 3])
  model_config.model_params.model_name = backbone_name
  if init_backbone:
    model_config.model_params.model_weights_path = (
        BACKBONE_PRETRAINED_CHECKPOINT[backbone_name])
  model_config.model_params.overrides.resolution = image_size
  config = cfg.ExperimentConfig(
      task=AutosegEdgeTPUTaskConfig(
          model=model_config,
          train_data=base_cfg.DataConfig(
              input_path=os.path.join(ADE20K_INPUT_PATH_BASE, 'train-*'),
              output_size=[image_size, image_size],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=base_cfg.DataConfig(
              input_path=os.path.join(ADE20K_INPUT_PATH_BASE, 'val-*'),
              output_size=[image_size, image_size],
              is_training=False,
              resize_eval_groundtruth=True,
              drop_remainder=True,
              global_batch_size=eval_batch_size),
          evaluation=base_cfg.Evaluation(report_train_mean_iou=False)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch * 5,
          max_to_keep=10,
          train_steps=train_steps,
          validation_steps=ADE20K_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'nesterov': True,
                      'momentum': 0.9,
                  }
              },
              'ema': {
                  'average_decay': 0.9998,
                  'trainable_weights_only': False,
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.12,
                      'decay_steps': train_steps
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              },
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


# Registration for searched segmentation model.
@exp_factory.register_config_factory('autoseg_edgetpu_xs')
def autoseg_edgetpu_xs() -> cfg.ExperimentConfig:
  return autoseg_edgetpu_experiment_config('autoseg_edgetpu_backbone_xs')


# Registration for searched segmentation model.
@exp_factory.register_config_factory('autoseg_edgetpu_s')
def autoseg_edgetpu_s() -> cfg.ExperimentConfig:
  return autoseg_edgetpu_experiment_config('autoseg_edgetpu_backbone_s')


# Registration for searched segmentation model.
@exp_factory.register_config_factory('autoseg_edgetpu_m')
def autoseg_edgetpu_m() -> cfg.ExperimentConfig:
  return autoseg_edgetpu_experiment_config('autoseg_edgetpu_backbone_m')
