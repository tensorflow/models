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

"""Image classification configuration definition."""
import dataclasses
import os
from typing import List, Optional, Tuple, Union, Sequence

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common
from official.vision.configs import backbones


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: Union[Sequence[str], str, hyperparams.Config] = ''
  weights: Optional[hyperparams.base_config.Config] = None
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  is_multilabel: bool = False
  aug_rand_hflip: bool = True
  aug_crop: Optional[bool] = True
  crop_area_range: Optional[Tuple[float, float]] = (0.08, 1.0)
  aug_type: Optional[
      common.Augmentation] = None  # Choose from AutoAugment and RandAugment.
  three_augment: bool = False
  color_jitter: float = 0.
  random_erasing: Optional[common.RandomErasing] = None
  file_type: str = 'tfrecord'
  image_field_key: str = 'image/encoded'
  label_field_key: str = 'image/class/label'
  decode_jpeg_only: bool = True
  mixup_and_cutmix: Optional[common.MixupAndCutmix] = None
  decoder: Optional[common.DataDecoder] = dataclasses.field(
      default_factory=common.DataDecoder
  )

  # Keep for backward compatibility.
  aug_policy: Optional[str] = None  # None, 'autoaug', or 'randaug'.
  randaug_magnitude: Optional[int] = 10
  # Determines ratio between the side of the cropped image and the short side of
  # the original image.
  center_crop_fraction: Optional[float] = 0.875
  # Interpolation method for resizing image in Parser for both training and eval
  tf_resize_method: str = 'bilinear'
  # Repeat augmentation puts multiple augmentations of the same image in a batch
  # https://arxiv.org/abs/1902.05509
  repeated_augment: Optional[int] = None


@dataclasses.dataclass
class ImageClassificationModel(hyperparams.Config):
  """The model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='resnet', resnet=backbones.ResNet()
      )
  )
  dropout_rate: float = 0.0
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(use_sync_bn=False)
  )
  # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
  add_head_batch_norm: bool = False
  kernel_initializer: str = 'random_uniform'
  # Whether to output softmax results instead of logits.
  output_softmax: bool = False


@dataclasses.dataclass
class Losses(hyperparams.Config):
  loss_weight: float = 1.0
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0
  soft_labels: bool = False
  # Converts multi-class classification to multi-label classification. Weights
  # each object class equally in the loss function, ignoring their size.
  use_binary_cross_entropy: bool = False


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5
  precision_and_recall_thresholds: Optional[List[float]] = None
  report_per_class_precision_and_recall: bool = False


@dataclasses.dataclass
class ImageClassificationTask(cfg.TaskConfig):
  """The task config."""
  model: ImageClassificationModel = dataclasses.field(
      default_factory=ImageClassificationModel
  )
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=False)
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  evaluation: Evaluation = dataclasses.field(default_factory=Evaluation)
  train_input_partition_dims: Optional[List[int]] = dataclasses.field(
      default_factory=list)
  eval_input_partition_dims: Optional[List[int]] = dataclasses.field(
      default_factory=list)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone
  model_output_keys: Optional[List[int]] = dataclasses.field(
      default_factory=list)
  freeze_backbone: bool = False


@exp_factory.register_config_factory('image_classification')
def image_classification() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=ImageClassificationTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


@exp_factory.register_config_factory('resnet_imagenet')
def image_classification_imagenet() -> cfg.ExperimentConfig:
  """Image classification on imagenet with resnet."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
          losses=Losses(l2_weight_decay=1e-4),
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
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
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
                      'boundaries': [
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [
                          0.1 * train_batch_size / 256,
                          0.01 * train_batch_size / 256,
                          0.001 * train_batch_size / 256,
                          0.0001 * train_batch_size / 256,
                      ]
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


@exp_factory.register_config_factory('resnet_rs_imagenet')
def image_classification_imagenet_resnetrs() -> cfg.ExperimentConfig:
  """Image classification on imagenet with resnet-rs."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[160, 160, 3],
              backbone=backbones.Backbone(
                  type='resnet',
                  resnet=backbones.ResNet(
                      model_id=50,
                      stem_type='v1',
                      resnetd_shortcut=True,
                      replace_stem_max_pool=True,
                      se_ratio=0.25,
                      stochastic_depth_drop_rate=0.0)),
              dropout_rate=0.25,
              norm_activation=common.NormActivation(
                  norm_momentum=0.0,
                  norm_epsilon=1e-5,
                  use_sync_bn=False,
                  activation='swish')),
          losses=Losses(l2_weight_decay=4e-5, label_smoothing=0.1),
          train_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_type=common.Augmentation(
                  type='randaug', randaug=common.RandAugment(magnitude=10))),
          validation_data=DataConfig(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=350 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'ema': {
                  'average_decay': 0.9999,
                  'trainable_weights_only': False,
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 1.6,
                      'decay_steps': 350 * steps_per_epoch
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


@exp_factory.register_config_factory('revnet_imagenet')
def image_classification_imagenet_revnet() -> cfg.ExperimentConfig:
  """Returns a revnet config for image classification on imagenet."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size

  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='revnet', revnet=backbones.RevNet(model_id=56)),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False),
              add_head_batch_norm=True),
          losses=Losses(l2_weight_decay=1e-4),
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
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
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
                      'boundaries': [
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [0.8, 0.08, 0.008, 0.0008]
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


@exp_factory.register_config_factory('mobilenet_imagenet')
def image_classification_imagenet_mobilenet() -> cfg.ExperimentConfig:
  """Image classification on imagenet with mobilenet."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              dropout_rate=0.2,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='mobilenet',
                  mobilenet=backbones.MobileNet(
                      model_id='MobileNetV2', filter_size_scale=1.0)),
              norm_activation=common.NormActivation(
                  norm_momentum=0.997, norm_epsilon=1e-3, use_sync_bn=False)),
          losses=Losses(l2_weight_decay=1e-5, label_smoothing=0.1),
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
          train_steps=500 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'rmsprop',
                  'rmsprop': {
                      'rho': 0.9,
                      'momentum': 0.9,
                      'epsilon': 0.002,
                  }
              },
              'learning_rate': {
                  'type': 'exponential',
                  'exponential': {
                      'initial_learning_rate':
                          0.008 * (train_batch_size // 128),
                      'decay_steps':
                          int(2.5 * steps_per_epoch),
                      'decay_rate':
                          0.98,
                      'staircase':
                          True
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


@exp_factory.register_config_factory('deit_imagenet_pretrain')
def image_classification_imagenet_deit_pretrain() -> cfg.ExperimentConfig:
  """Image classification on imagenet with vision transformer."""
  train_batch_size = 4096  # originally was 1024 but 4096 better for tpu v3-32
  eval_batch_size = 4096  # originally was 1024 but 4096 better for tpu v3-32
  label_smoothing = 0.1
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
