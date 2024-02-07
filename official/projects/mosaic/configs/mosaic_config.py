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

"""Configuration definition for Semantic Segmentation with MOSAIC."""
import dataclasses
import os
from typing import List, Optional, Union

import numpy as np

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import backbones
from official.vision.configs import common
from official.vision.configs import semantic_segmentation as seg_cfg


@dataclasses.dataclass
class MosaicDecoderHead(hyperparams.Config):
  """MOSAIC decoder head config for Segmentation."""
  num_classes: int = 19
  decoder_input_levels: List[str] = dataclasses.field(default_factory=list)
  decoder_stage_merge_styles: List[str] = dataclasses.field(
      default_factory=list)
  decoder_filters: List[int] = dataclasses.field(default_factory=list)
  decoder_projected_filters: List[int] = dataclasses.field(default_factory=list)
  encoder_end_level: int = 4
  use_additional_classifier_layer: bool = False
  classifier_kernel_size: int = 1
  activation: str = 'relu'
  kernel_initializer: str = 'glorot_uniform'
  interpolation: str = 'bilinear'


@dataclasses.dataclass
class MosaicEncoderNeck(hyperparams.Config):
  """MOSAIC encoder neck config for segmentation."""
  encoder_input_level: Union[str, int] = '4'
  branch_filter_depths: List[int] = dataclasses.field(default_factory=list)
  conv_kernel_sizes: List[int] = dataclasses.field(default_factory=list)
  pyramid_pool_bin_nums: List[int] = dataclasses.field(default_factory=list)
  activation: str = 'relu'
  dropout_rate: float = 0.1
  kernel_initializer: str = 'glorot_uniform'
  interpolation: str = 'bilinear'
  use_depthwise_convolution: bool = True


@dataclasses.dataclass
class MosaicSemanticSegmentationModel(hyperparams.Config):
  """MOSAIC semantic segmentation model config."""
  num_classes: int = 19
  input_size: List[int] = dataclasses.field(default_factory=list)
  head: MosaicDecoderHead = dataclasses.field(default_factory=MosaicDecoderHead)
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='mobilenet', mobilenet=backbones.MobileNet()
      )
  )
  neck: MosaicEncoderNeck = dataclasses.field(default_factory=MosaicEncoderNeck)
  mask_scoring_head: Optional[seg_cfg.MaskScoringHead] = None
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(  # pylint: disable=g-long-lambda
          use_sync_bn=True, norm_momentum=0.99, norm_epsilon=0.001
      )
  )


@dataclasses.dataclass
class MosaicSemanticSegmentationTask(seg_cfg.SemanticSegmentationTask):
  """The config for MOSAIC segmentation task."""
  model: MosaicSemanticSegmentationModel = dataclasses.field(
      default_factory=MosaicSemanticSegmentationModel
  )
  train_data: seg_cfg.DataConfig = dataclasses.field(
      default_factory=lambda: seg_cfg.DataConfig(is_training=True)
  )
  validation_data: seg_cfg.DataConfig = dataclasses.field(
      default_factory=lambda: seg_cfg.DataConfig(is_training=False)
  )
  losses: seg_cfg.Losses = dataclasses.field(default_factory=seg_cfg.Losses)
  evaluation: seg_cfg.Evaluation = dataclasses.field(
      default_factory=seg_cfg.Evaluation
  )
  train_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  eval_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or neck.
  export_config: seg_cfg.ExportConfig = dataclasses.field(
      default_factory=seg_cfg.ExportConfig
  )


# Cityscapes Dataset (Download and process the dataset yourself)
CITYSCAPES_TRAIN_EXAMPLES = 2975
CITYSCAPES_VAL_EXAMPLES = 500
CITYSCAPES_INPUT_PATH_BASE = 'cityscapes/tfrecord'


@exp_factory.register_config_factory('mosaic_mnv35_cityscapes')
def mosaic_mnv35_cityscapes() -> cfg.ExperimentConfig:
  """Instantiates an experiment configuration of image segmentation task.

  This image segmentation experiment is conducted on Cityscapes dataset. The
  model architecture is a MOSAIC encoder-decoer. The default backbone network is
  a mobilenet variant called Mobilenet_v3.5-MultiAvg on top of which the MOSAIC
  encoder-decoder can be deployed. All detailed configurations can be overridden
  by a .yaml file provided by the user to launch the experiments. Please refer
  to .yaml examples in the path of ../configs/experiments/.

  Returns:
    A particular instance of cfg.ExperimentConfig for MOSAIC model based
    image semantic segmentation task.
  """
  train_batch_size = 16
  eval_batch_size = 16
  steps_per_epoch = CITYSCAPES_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16

  backbone_output_level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=MosaicSemanticSegmentationTask(
          model=MosaicSemanticSegmentationModel(
              # Cityscapes uses only 19 semantic classes for train/evaluation.
              # The void (background) class is ignored in train and evaluation.
              num_classes=19,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='mobilenet',
                  mobilenet=backbones.MobileNet(
                      model_id='MobileNetMultiAVGSeg',
                      output_intermediate_endpoints=True,
                      output_stride=output_stride)),
              neck=MosaicEncoderNeck(
                  encoder_input_level=backbone_output_level,
                  branch_filter_depths=[64, 64],
                  conv_kernel_sizes=[3, 5],
                  pyramid_pool_bin_nums=[1, 4, 8, 16],  # paper default
                  activation='relu',
                  dropout_rate=0.1,
                  kernel_initializer='glorot_uniform',
                  interpolation='bilinear',
                  use_depthwise_convolution=True),
              head=MosaicDecoderHead(
                  num_classes=19,
                  decoder_input_levels=['3/depthwise', '2/depthwise'],
                  decoder_stage_merge_styles=['concat_merge', 'sum_merge'],
                  decoder_filters=[64, 64],
                  decoder_projected_filters=[19, 19],
                  encoder_end_level=backbone_output_level,
                  use_additional_classifier_layer=False,
                  classifier_kernel_size=1,
                  activation='relu',
                  kernel_initializer='glorot_uniform',
                  interpolation='bilinear'),
              norm_activation=common.NormActivation(
                  activation='relu',
                  norm_momentum=0.99,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=seg_cfg.Losses(l2_weight_decay=4e-5),
          train_data=seg_cfg.DataConfig(
              input_path=os.path.join(CITYSCAPES_INPUT_PATH_BASE,
                                      'train_fine**'),
              crop_size=[1024, 2048],
              output_size=[1024, 2048],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=seg_cfg.DataConfig(
              input_path=os.path.join(CITYSCAPES_INPUT_PATH_BASE, 'val_fine*'),
              output_size=[1024, 2048],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=True,
              drop_remainder=False),
          # Imagenet pre-trained Mobilenet_v3.5-MultiAvg checkpoint.
          init_checkpoint='gs://tf_model_garden/vision/mobilenet/v3.5multiavg_seg_float/',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=100000,
          validation_steps=CITYSCAPES_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          best_checkpoint_eval_metric='mean_iou',
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_metric_comp='higher',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.1,
                      'decay_steps': 100000,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
