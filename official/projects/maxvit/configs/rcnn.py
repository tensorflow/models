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

"""Mask R-CNN configuration definition."""
import os

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling.optimization.configs import optimization_config
from official.projects.maxvit.configs import backbones
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import maskrcnn


Parser = maskrcnn.Parser
Anchor = maskrcnn.Anchor
Losses = maskrcnn.Losses
ROISampler = maskrcnn.ROISampler
DetectionHead = maskrcnn.DetectionHead
DataConfig = maskrcnn.DataConfig
MaskRCNN = maskrcnn.MaskRCNN
MaskRCNNTask = maskrcnn.MaskRCNNTask


COCO_INPUT_PATH_BASE = (
    '/readahead/200M/placer/prod/home/tensorflow-performance-data/datasets/coco'
)


@exp_factory.register_config_factory('rcnn_maxvit_coco')
def rcnn_maxvit_coco() -> cfg.ExperimentConfig:
  """COCO object detection with MaxViT and Cascade R-CNN."""
  steps_per_epoch = 1848  # based on 463 steps @ bs=256
  train_batch_size = 256
  coco_val_samples = 5000
  eval_batch_size = 64

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=MaskRCNNTask(
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskRCNN(
              anchor=Anchor(num_scales=3, anchor_size=3.0),
              backbone=backbones.Backbone(
                  type='maxvit',
                  maxvit=backbones.MaxViT(model_name='maxvit-base')
              ),
              decoder=decoders.Decoder(type='fpn', fpn=decoders.FPN()),
              num_classes=91,
              input_size=[640, 640, 3],
              include_mask=True,
              roi_sampler=ROISampler(
                  cascade_iou_thresholds=[0.7], foreground_iou_threshold=0.6),
              detection_head=DetectionHead(
                  cascade_class_ensemble=True, class_agnostic_bbox_pred=True),
              norm_activation=common.NormActivation(
                  use_sync_bn=True,
                  activation='relu',
                  norm_epsilon=0.001,
                  norm_momentum=0.99),
              min_level=3,
              max_level=7,
          ),
          losses=Losses(l2_weight_decay=0.0),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True, aug_scale_min=0.1, aug_scale_max=2.5)),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=True)),
      trainer=cfg.TrainerConfig(
          train_steps=90000,
          validation_steps=coco_val_samples // eval_batch_size,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          checkpoint_interval=steps_per_epoch * 4,
          optimizer_config=optimization_config.OptimizationConfig({
              'ema': {
                  'average_decay': 0.9998,
                  'trainable_weights_only': False,
              },
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate': 0.0001,
                      'beta_1': 0.9,
                      'beta_2': 0.999,
                      'include_in_weight_decay': r'.*(kernel|weight):0$',
                  },
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'decay_steps': 90000,
                      'initial_learning_rate': 0.0001,
                      'alpha': 0.03,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 6000,
                      'warmup_learning_rate': 0.,
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config
