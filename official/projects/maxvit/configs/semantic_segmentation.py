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

"""Semantic segmentation configuration definition."""
import os

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.projects.maxvit.configs import backbones
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import semantic_segmentation

DataConfig = semantic_segmentation.DataConfig
Losses = semantic_segmentation.Losses
Evaluation = semantic_segmentation.Evaluation
SegmentationHead = semantic_segmentation.SegmentationHead
SemanticSegmentationModel = semantic_segmentation.SemanticSegmentationModel
SemanticSegmentationTask = semantic_segmentation.SemanticSegmentationTask


# PASCAL VOC 2012 Dataset
PASCAL_TRAIN_EXAMPLES = 10582
PASCAL_VAL_EXAMPLES = 1449
PASCAL_INPUT_PATH_BASE = 'gs://**/pascal_voc_seg'


@exp_factory.register_config_factory('maxvit_seg_pascal')
def maxvit_seg_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on Pascal VOC with MaxViT."""
  train_batch_size = 32
  eval_batch_size = 32
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[512, 512, 3],
              min_level=3,
              max_level=7,
              backbone=backbones.Backbone(
                  type='maxvit',
                  maxvit=backbones.MaxViT(
                      model_name='maxvit-tiny',
                      window_size=16,
                      grid_size=16,
                      scale_ratio='16/7',
                  ),
              ),
              decoder=decoders.Decoder(type='fpn', fpn=decoders.FPN()),
              head=SegmentationHead(level=3, num_convs=3),
              norm_activation=common.NormActivation(
                  use_sync_bn=True,
                  activation='relu',
                  norm_epsilon=0.001,
                  norm_momentum=0.99,
              ),
          ),
          losses=Losses(l2_weight_decay=1e-5, top_k_percent_pixels=1.0),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              output_size=[512, 512],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_rand_hflip=True,
              aug_scale_min=0.2,
              aug_scale_max=1.5,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              output_size=[512, 512],
              is_training=True,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=True,
              groundtruth_padded_size=[512, 512],
              drop_remainder=True,
          ),
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=20000,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema': {
                  'average_decay': 0.9998,
                  'trainable_weights_only': False,
              },
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'beta_1': 0.9,
                      'beta_2': 0.999,
                      'weight_decay_rate': 0.0001,
                      'include_in_weight_decay': r'.*(kernel|weight):0$',
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.0005,
                      'decay_steps': 20000,
                      'alpha': 0.03,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
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


# COCO segmentation.
COCO_TRAIN_EXAMPLES = 25600
COCO_VAL_EXAMPLES = 5000
COCO_INPUT_PATH_BASE = 'mscoco'


@exp_factory.register_config_factory('maxvit_seg_coco')
def maxvit_seg_coco() -> cfg.ExperimentConfig:
  """Image segmentation on COCO with MaxViT."""
  train_batch_size = 32
  eval_batch_size = 32
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=91,
              input_size=[640, 640, 3],
              backbone=backbones.Backbone(
                  type='maxvit',
                  maxvit=backbones.MaxViT(
                      model_name='maxvit-tiny',
                      window_size=20,
                      grid_size=20,
                      scale_ratio='20/7',
                  ),
              ),
              decoder=decoders.Decoder(type='fpn', fpn=decoders.FPN()),
              head=SegmentationHead(level=3, num_convs=3),
              norm_activation=common.NormActivation(
                  use_sync_bn=True,
                  activation='relu',
                  norm_epsilon=0.001,
                  norm_momentum=0.99,
              ),
          ),
          losses=Losses(l2_weight_decay=1e-5, top_k_percent_pixels=1.0),
          train_data=DataConfig(
              input_path=os.path.join(
                  COCO_INPUT_PATH_BASE,
                  'mscoco_alltasks_trainvalminusminival2014*',
              ),
              output_size=[640, 640],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_rand_hflip=True,
              aug_scale_min=0.2,
              aug_scale_max=2.0,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(
                  COCO_INPUT_PATH_BASE, 'mscoco_alltasks_minival2014*'
              ),
              output_size=[640, 640],
              is_training=True,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=True,
              groundtruth_padded_size=[640, 640],
              drop_remainder=True,
          ),
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=64000,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema': {
                  'average_decay': 0.9998,
                  'trainable_weights_only': False,
              },
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'beta_1': 0.9,
                      'beta_2': 0.999,
                      'weight_decay_rate': 0.00001,
                      'include_in_weight_decay': r'.*(kernel|weight):0$',
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.00005,
                      'decay_steps': 64000,
                      'alpha': 0.03,
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1600,
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
