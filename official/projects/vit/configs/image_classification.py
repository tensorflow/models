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

"""Image classification configuration definition."""
import dataclasses
import os
from typing import Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.core import task_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.projects.vit.configs import backbones
from official.vision.configs import common
from official.vision.configs import image_classification as img_cls_cfg
from official.vision.tasks import image_classification

# pytype: disable=wrong-keyword-args

DataConfig = img_cls_cfg.DataConfig


@dataclasses.dataclass
class ImageClassificationModel(img_cls_cfg.ImageClassificationModel):
  """The model config."""
  backbone: backbones.Backbone = backbones.Backbone(
      type='vit', vit=backbones.VisionTransformer())


@dataclasses.dataclass
class Losses(hyperparams.Config):
  loss_weight: float = 1.0
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0
  soft_labels: bool = False


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5


@dataclasses.dataclass
class ImageClassificationTask(cfg.TaskConfig):
  """The task config. Same as the classification task for convnets."""
  model: ImageClassificationModel = ImageClassificationModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone


IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'

# TODO(b/177942984): integrate the experiments to TF-vision.
task_factory.register_task_cls(ImageClassificationTask)(
    image_classification.ImageClassificationTask)


@exp_factory.register_config_factory('deit_imagenet_pretrain')
def image_classification_imagenet_deit_pretrain() -> cfg.ExperimentConfig:
  """Image classification on imagenet with vision transformer."""
  train_batch_size = 4096  # originally was 1024 but 4096 better for tpu v3-32
  eval_batch_size = 4096  # originally was 1024 but 4096 better for tpu v3-32
  num_classes = 1001
  label_smoothing = 0.1
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=num_classes,
              input_size=[224, 224, 3],
              kernel_initializer='zeros',
              backbone=backbones.Backbone(
                  type='vit',
                  vit=backbones.VisionTransformer(
                      model_name='vit-b16',
                      representation_size=768,
                      init_stochastic_depth_rate=0.1,
                      original_init=False,
                      transformer=backbones.Transformer(
                          dropout_rate=0.0, attention_dropout_rate=0.0)))),
          losses=Losses(
              l2_weight_decay=0.0,
              label_smoothing=label_smoothing,
              one_hot=False,
              soft_labels=True),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_type=common.Augmentation(
                  type='randaug',
                  randaug=common.RandAugment(
                      magnitude=9, exclude_ops=['Cutout'])),
              mixup_and_cutmix=common.MixupAndCutmix(
                  label_smoothing=label_smoothing)),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=300 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate': 0.05,
                      'include_in_weight_decay': r'.*(kernel|weight):0$',
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.0005 * train_batch_size / 512,
                      'decay_steps': 300 * steps_per_epoch,
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


@exp_factory.register_config_factory('vit_imagenet_pretrain')
def image_classification_imagenet_vit_pretrain() -> cfg.ExperimentConfig:
  """Image classification on imagenet with vision transformer."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              kernel_initializer='zeros',
              backbone=backbones.Backbone(
                  type='vit',
                  vit=backbones.VisionTransformer(
                      model_name='vit-b16', representation_size=768))),
          losses=Losses(l2_weight_decay=0.0),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=300 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate': 0.3,
                      'include_in_weight_decay': r'.*(kernel|weight):0$',
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.003 * train_batch_size / 4096,
                      'decay_steps': 300 * steps_per_epoch,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 10000,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('vit_imagenet_finetune')
def image_classification_imagenet_vit_finetune() -> cfg.ExperimentConfig:
  """Image classification on imagenet with vision transformer."""
  train_batch_size = 512
  eval_batch_size = 512
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[384, 384, 3],
              backbone=backbones.Backbone(
                  type='vit',
                  vit=backbones.VisionTransformer(model_name='vit-b16'))),
          losses=Losses(l2_weight_decay=0.0),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=20000,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9,
                      'global_clipnorm': 1.0,
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.003,
                      'decay_steps': 20000,
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
