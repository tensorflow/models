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
"""Semantic segmentation configuration definition.

The segmentation model is built using the mobilenet edgetpu v2 backbone and
deeplab v3 segmentation head.
"""
import dataclasses
import os
from typing import Optional

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders
from official.vision.beta.configs import semantic_segmentation as base_cfg


@dataclasses.dataclass
class MobileNetEdgeTPU(hyperparams.Config):
  """MobileNetEdgeTPU config."""
  model_id: str = 'mobilenet_edgetpu_v2_s'
  freeze_large_filters: Optional[int] = None
  pretrained_checkpoint_path: Optional[str] = None


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, on the of fields below.
    spinenet_seg: spinenet-seg backbone config.
  """
  type: Optional[str] = None
  mobilenet_edgetpu: MobileNetEdgeTPU = MobileNetEdgeTPU()


@dataclasses.dataclass
class CustomSemanticSegmentationTaskConfig(base_cfg.SemanticSegmentationTask):
  """Same config for custom taks."""

  model: Optional[base_cfg.SemanticSegmentationModel] = None
  train_data: base_cfg.DataConfig = base_cfg.DataConfig(is_training=True)
  validation_data: base_cfg.DataConfig = base_cfg.DataConfig(is_training=False)
  evaluation: base_cfg.Evaluation = base_cfg.Evaluation()


# ADE 20K Dataset
ADE20K_TRAIN_EXAMPLES = 20210
ADE20K_VAL_EXAMPLES = 2000
ADE20K_INPUT_PATH_BASE = 'gs://**/ADE20K'

PRETRAINED_CKPT_PATH_BASE = 'gs://**/placeholder_for_edgetpu_models'

BACKBONE_PRETRAINED_CHECKPOINT = {
    'mobilenet_edgetpu_v2_l':
        PRETRAINED_CKPT_PATH_BASE +
        '/pretrained_checkpoints/mobilenet_edgetpu_v2_l/ckpt-171600',
    'mobilenet_edgetpu_v2_m':
        PRETRAINED_CKPT_PATH_BASE +
        '/pretrained_checkpoints/mobilenet_edgetpu_v2_m/ckpt-171600',
    'mobilenet_edgetpu_v2_s':
        PRETRAINED_CKPT_PATH_BASE +
        '/pretrained_checkpoints/mobilenet_edgetpu_v2_s/ckpt-171600',
    'mobilenet_edgetpu_v2_xs':
        PRETRAINED_CKPT_PATH_BASE +
        '/pretrained_checkpoints/mobilenet_edgetpu_v2_xs/ckpt-171600',
}

BACKBONE_HEADPOINT = {
    'mobilenet_edgetpu_v2_l': 4,
    'mobilenet_edgetpu_v2_m': 4,
    'mobilenet_edgetpu_v2_s': 4,
    'mobilenet_edgetpu_v2_xs': 4,
}

BACKBONE_LOWER_FEATURES = {
    'mobilenet_edgetpu_v2_l': 3,
    'mobilenet_edgetpu_v2_m': 3,
    'mobilenet_edgetpu_v2_s': 3,
    'mobilenet_edgetpu_v2_xs': 3,
}


def seg_deeplabv3plus_ade20k_32(backbone: str,
                                init_backbone: bool = True
                               ) -> cfg.ExperimentConfig:
  """Semantic segmentation on ADE20K dataset with deeplabv3+."""
  epochs = 200
  train_batch_size = 128
  eval_batch_size = 32
  image_size = 512
  steps_per_epoch = ADE20K_TRAIN_EXAMPLES // train_batch_size
  aspp_dilation_rates = [5, 10, 15]
  pretrained_checkpoint_path = BACKBONE_PRETRAINED_CHECKPOINT[
      backbone] if init_backbone else None
  config = cfg.ExperimentConfig(
      task=CustomSemanticSegmentationTaskConfig(
          model=base_cfg.SemanticSegmentationModel(
              # ADE20K uses only 32 semantic classes for train/evaluation.
              # The void (background) class is ignored in train and evaluation.
              num_classes=32,
              input_size=[None, None, 3],
              backbone=Backbone(
                  type='mobilenet_edgetpu',
                  mobilenet_edgetpu=MobileNetEdgeTPU(
                      model_id=backbone,
                      pretrained_checkpoint_path=pretrained_checkpoint_path,
                      freeze_large_filters=500,
                  )),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=BACKBONE_HEADPOINT[backbone],
                      use_depthwise_convolution=True,
                      dilation_rates=aspp_dilation_rates,
                      pool_kernel_size=[256, 256],
                      num_filters=128,
                      dropout_rate=0.3,
                  )),
              head=base_cfg.SegmentationHead(
                  level=BACKBONE_HEADPOINT[backbone],
                  num_convs=2,
                  num_filters=256,
                  use_depthwise_convolution=True,
                  feature_fusion='deeplabv3plus',
                  low_level=BACKBONE_LOWER_FEATURES[backbone],
                  low_level_num_filters=48),
              norm_activation=common.NormActivation(
                  activation='relu',
                  norm_momentum=0.99,
                  norm_epsilon=2e-3,
                  use_sync_bn=False)),
          train_data=base_cfg.DataConfig(
              input_path=os.path.join(ADE20K_INPUT_PATH_BASE, 'train-*'),
              output_size=[image_size, image_size],
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=base_cfg.DataConfig(
              input_path=os.path.join(ADE20K_INPUT_PATH_BASE, 'val-*'),
              output_size=[image_size, image_size],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=True,
              drop_remainder=False),
          evaluation=base_cfg.Evaluation(report_train_mean_iou=False),
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=epochs * steps_per_epoch,
          validation_steps=ADE20K_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.0001,
                      'decay_steps': epochs * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 4 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


def seg_deeplabv3plus_ade20k(backbone: str):
  config = seg_deeplabv3plus_ade20k_32(backbone)
  config.task.model.num_classes = 151
  config.trainer.optimizer_config.learning_rate.polynomial.power = 1.1
  config.task.model.decoder.aspp.num_filters = 160
  config.task.model.head.low_level_num_filters = 64
  return config


# Experiment configs for 32 output classes
@exp_factory.register_config_factory(
    'deeplabv3plus_mobilenet_edgetpuv2_m_ade20k_32')
def deeplabv3plus_mobilenet_edgetpuv2_m_ade20k_32() -> cfg.ExperimentConfig:
  return seg_deeplabv3plus_ade20k_32('mobilenet_edgetpu_v2_m')


@exp_factory.register_config_factory(
    'deeplabv3plus_mobilenet_edgetpuv2_s_ade20k_32')
def deeplabv3plus_mobilenet_edgetpuv2_s_ade20k_32() -> cfg.ExperimentConfig:
  return seg_deeplabv3plus_ade20k_32('mobilenet_edgetpu_v2_s')


@exp_factory.register_config_factory(
    'deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k_32')
def deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k_32() -> cfg.ExperimentConfig:
  return seg_deeplabv3plus_ade20k_32('mobilenet_edgetpu_v2_xs')


# Experiment configs for 151 output classes
@exp_factory.register_config_factory(
    'deeplabv3plus_mobilenet_edgetpuv2_m_ade20k')
def deeplabv3plus_mobilenet_edgetpuv2_m_ade20k() -> cfg.ExperimentConfig:
  config = seg_deeplabv3plus_ade20k('mobilenet_edgetpu_v2_m')
  return config


@exp_factory.register_config_factory(
    'deeplabv3plus_mobilenet_edgetpuv2_s_ade20k')
def deeplabv3plus_mobilenet_edgetpuv2_s_ade20k() -> cfg.ExperimentConfig:
  config = seg_deeplabv3plus_ade20k('mobilenet_edgetpu_v2_s')
  return config


@exp_factory.register_config_factory(
    'deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k')
def deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k() -> cfg.ExperimentConfig:
  config = seg_deeplabv3plus_ade20k('mobilenet_edgetpu_v2_xs')
  return config
