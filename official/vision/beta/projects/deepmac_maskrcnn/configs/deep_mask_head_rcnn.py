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

"""Configuration for Mask R-CNN with deep mask heads."""

import os
from typing import Optional

import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.beta.configs import maskrcnn as maskrcnn_config
from official.vision.beta.configs import retinanet as retinanet_config


@dataclasses.dataclass
class DeepMaskHead(maskrcnn_config.MaskHead):
  convnet_variant: str = 'default'


@dataclasses.dataclass
class DeepMaskHeadRCNN(maskrcnn_config.MaskRCNN):
  mask_head: Optional[DeepMaskHead] = DeepMaskHead()
  use_gt_boxes_for_masks: bool = False


@dataclasses.dataclass
class DeepMaskHeadRCNNTask(maskrcnn_config.MaskRCNNTask):
  """Configuration for the deep mask head R-CNN task."""
  model: DeepMaskHeadRCNN = DeepMaskHeadRCNN()


@exp_factory.register_config_factory('deep_mask_head_rcnn_resnetfpn_coco')
def deep_mask_head_rcnn_resnetfpn_coco() -> cfg.ExperimentConfig:
  """COCO object detection with Mask R-CNN with deep mask heads."""
  global_batch_size = 64
  steps_per_epoch = int(retinanet_config.COCO_TRAIN_EXAMPLES /
                        global_batch_size)
  coco_val_samples = 5000

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=DeepMaskHeadRCNNTask(
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(maskrcnn_config.COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=DeepMaskHeadRCNN(
              num_classes=91,
              input_size=[1024, 1024, 3],
              include_mask=True),  # pytype: disable=wrong-keyword-args
          losses=maskrcnn_config.Losses(l2_weight_decay=0.00004),
          train_data=maskrcnn_config.DataConfig(
              input_path=os.path.join(
                  maskrcnn_config.COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=global_batch_size,
              parser=maskrcnn_config.Parser(
                  aug_rand_hflip=True, aug_scale_min=0.8, aug_scale_max=1.25)),
          validation_data=maskrcnn_config.DataConfig(
              input_path=os.path.join(
                  maskrcnn_config.COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=8)),  # pytype: disable=wrong-keyword-args
      trainer=cfg.TrainerConfig(
          train_steps=22500,
          validation_steps=coco_val_samples // 8,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [15000, 20000],
                      'values': [0.12, 0.012, 0.0012],
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
                      'warmup_learning_rate': 0.0067
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
