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

"""Video classification configuration definition."""
import dataclasses
from typing import Optional, Tuple, Union
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import backbones_3d
from official.vision.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """The base configuration for building datasets."""
  name: Optional[str] = None
  file_type: Optional[str] = 'tfrecord'
  compressed_input: bool = False
  split: str = 'train'
  variant_name: Optional[str] = None
  feature_shape: Tuple[int, ...] = (64, 224, 224, 3)
  temporal_stride: int = 1
  random_stride_range: int = 0
  num_test_clips: int = 1
  num_test_crops: int = 1
  num_classes: int = -1
  num_examples: int = -1
  global_batch_size: int = 128
  data_format: str = 'channels_last'
  dtype: str = 'float32'
  label_dtype: str = 'int32'
  one_hot: bool = True
  shuffle_buffer_size: int = 64
  cache: bool = False
  input_path: Union[str, cfg.base_config.Config] = ''
  is_training: bool = True
  cycle_length: int = 10
  drop_remainder: bool = True
  min_image_size: int = 256
  zero_centering_image: bool = False
  is_multilabel: bool = False
  output_audio: bool = False
  audio_feature: str = ''
  audio_feature_shape: Tuple[int, ...] = (-1,)
  aug_min_aspect_ratio: float = 0.5
  aug_max_aspect_ratio: float = 2.0
  aug_min_area_ratio: float = 0.49
  aug_max_area_ratio: float = 1.0
  aug_type: Optional[
      common.Augmentation] = None  # AutoAugment and RandAugment.
  mixup_and_cutmix: Optional[common.MixupAndCutmix] = None
  image_field_key: str = 'image/encoded'
  label_field_key: str = 'clip/label/index'


def kinetics400(is_training):
  """Generated Kinetics 400 dataset configs."""
  return DataConfig(
      name='kinetics400',
      num_classes=400,
      is_training=is_training,
      split='train' if is_training else 'valid',
      drop_remainder=is_training,
      num_examples=215570 if is_training else 17706,
      feature_shape=(64, 224, 224, 3) if is_training else (250, 224, 224, 3))


def kinetics600(is_training):
  """Generated Kinetics 600 dataset configs."""
  return DataConfig(
      name='kinetics600',
      num_classes=600,
      is_training=is_training,
      split='train' if is_training else 'valid',
      drop_remainder=is_training,
      num_examples=366016 if is_training else 27780,
      feature_shape=(64, 224, 224, 3) if is_training else (250, 224, 224, 3))


def kinetics700(is_training):
  """Generated Kinetics 600 dataset configs."""
  return DataConfig(
      name='kinetics700',
      num_classes=700,
      is_training=is_training,
      split='train' if is_training else 'valid',
      drop_remainder=is_training,
      num_examples=522883 if is_training else 33441,
      feature_shape=(64, 224, 224, 3) if is_training else (250, 224, 224, 3))


def kinetics700_2020(is_training):
  """Generated Kinetics 600 dataset configs."""
  return DataConfig(
      name='kinetics700',
      num_classes=700,
      is_training=is_training,
      split='train' if is_training else 'valid',
      drop_remainder=is_training,
      num_examples=535982 if is_training else 33640,
      feature_shape=(64, 224, 224, 3) if is_training else (250, 224, 224, 3))


@dataclasses.dataclass
class VideoClassificationModel(hyperparams.Config):
  """The model config."""
  model_type: str = 'video_classification'
  backbone: backbones_3d.Backbone3D = backbones_3d.Backbone3D(
      type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50())
  norm_activation: common.NormActivation = common.NormActivation(
      use_sync_bn=False)
  dropout_rate: float = 0.2
  aggregate_endpoints: bool = False
  require_endpoints: Optional[Tuple[str, ...]] = None


@dataclasses.dataclass
class Losses(hyperparams.Config):
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Metrics(hyperparams.Config):
  use_per_class_recall: bool = False


@dataclasses.dataclass
class VideoClassificationTask(cfg.TaskConfig):
  """The task config."""
  model: VideoClassificationModel = VideoClassificationModel()
  train_data: DataConfig = DataConfig(is_training=True, drop_remainder=True)
  validation_data: DataConfig = DataConfig(
      is_training=False, drop_remainder=False)
  losses: Losses = Losses()
  metrics: Metrics = Metrics()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone
  freeze_backbone: bool = False
  # Spatial Partitioning fields.
  train_input_partition_dims: Optional[Tuple[int, ...]] = None
  eval_input_partition_dims: Optional[Tuple[int, ...]] = None


def add_trainer(experiment: cfg.ExperimentConfig,
                train_batch_size: int,
                eval_batch_size: int,
                learning_rate: float = 1.6,
                train_epochs: int = 44,
                warmup_epochs: int = 5):
  """Add and config a trainer to the experiment config."""
  if experiment.task.train_data.num_examples <= 0:
    raise ValueError('Wrong train dataset size {!r}'.format(
        experiment.task.train_data))
  if experiment.task.validation_data.num_examples <= 0:
    raise ValueError('Wrong validation dataset size {!r}'.format(
        experiment.task.validation_data))
  experiment.task.train_data.global_batch_size = train_batch_size
  experiment.task.validation_data.global_batch_size = eval_batch_size
  steps_per_epoch = experiment.task.train_data.num_examples // train_batch_size
  experiment.trainer = cfg.TrainerConfig(
      steps_per_loop=steps_per_epoch,
      summary_interval=steps_per_epoch,
      checkpoint_interval=steps_per_epoch,
      train_steps=train_epochs * steps_per_epoch,
      validation_steps=experiment.task.validation_data.num_examples //
      eval_batch_size,
      validation_interval=steps_per_epoch,
      optimizer_config=optimization.OptimizationConfig({
          'optimizer': {
              'type': 'sgd',
              'sgd': {
                  'momentum': 0.9,
                  'nesterov': True,
              }
          },
          'learning_rate': {
              'type': 'cosine',
              'cosine': {
                  'initial_learning_rate': learning_rate,
                  'decay_steps': train_epochs * steps_per_epoch,
              }
          },
          'warmup': {
              'type': 'linear',
              'linear': {
                  'warmup_steps': warmup_epochs * steps_per_epoch,
                  'warmup_learning_rate': 0
              }
          }
      }))
  return experiment


@exp_factory.register_config_factory('video_classification')
def video_classification() -> cfg.ExperimentConfig:
  """Video classification general."""
  return cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=VideoClassificationTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])


@exp_factory.register_config_factory('video_classification_ucf101')
def video_classification_ucf101() -> cfg.ExperimentConfig:
  """Video classification on UCF-101 with resnet."""
  train_dataset = DataConfig(
      name='ucf101',
      num_classes=101,
      is_training=True,
      split='train',
      drop_remainder=True,
      num_examples=9537,
      temporal_stride=2,
      feature_shape=(32, 224, 224, 3))
  train_dataset.tfds_name = 'ucf101'
  train_dataset.tfds_split = 'train'
  validation_dataset = DataConfig(
      name='ucf101',
      num_classes=101,
      is_training=True,
      split='test',
      drop_remainder=False,
      num_examples=3783,
      temporal_stride=2,
      feature_shape=(32, 224, 224, 3))
  validation_dataset.tfds_name = 'ucf101'
  validation_dataset.tfds_split = 'test'
  task = VideoClassificationTask(
      model=VideoClassificationModel(
          backbone=backbones_3d.Backbone3D(
              type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50()),
          norm_activation=common.NormActivation(
              norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
      losses=Losses(l2_weight_decay=1e-4),
      train_data=train_dataset,
      validation_data=validation_dataset)
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=task,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])
  add_trainer(
      config,
      train_batch_size=64,
      eval_batch_size=16,
      learning_rate=0.8,
      train_epochs=100)
  return config


@exp_factory.register_config_factory('video_classification_kinetics400')
def video_classification_kinetics400() -> cfg.ExperimentConfig:
  """Video classification on Kinetics 400 with resnet."""
  train_dataset = kinetics400(is_training=True)
  validation_dataset = kinetics400(is_training=False)
  task = VideoClassificationTask(
      model=VideoClassificationModel(
          backbone=backbones_3d.Backbone3D(
              type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50()),
          norm_activation=common.NormActivation(
              norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
      losses=Losses(l2_weight_decay=1e-4),
      train_data=train_dataset,
      validation_data=validation_dataset)
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=task,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])
  add_trainer(config, train_batch_size=1024, eval_batch_size=64)
  return config


@exp_factory.register_config_factory('video_classification_kinetics600')
def video_classification_kinetics600() -> cfg.ExperimentConfig:
  """Video classification on Kinetics 600 with resnet."""
  train_dataset = kinetics600(is_training=True)
  validation_dataset = kinetics600(is_training=False)
  task = VideoClassificationTask(
      model=VideoClassificationModel(
          backbone=backbones_3d.Backbone3D(
              type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50()),
          norm_activation=common.NormActivation(
              norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
      losses=Losses(l2_weight_decay=1e-4),
      train_data=train_dataset,
      validation_data=validation_dataset)
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=task,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])
  add_trainer(config, train_batch_size=1024, eval_batch_size=64)
  return config


@exp_factory.register_config_factory('video_classification_kinetics700')
def video_classification_kinetics700() -> cfg.ExperimentConfig:
  """Video classification on Kinetics 700 with resnet."""
  train_dataset = kinetics700(is_training=True)
  validation_dataset = kinetics700(is_training=False)
  task = VideoClassificationTask(
      model=VideoClassificationModel(
          backbone=backbones_3d.Backbone3D(
              type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50()),
          norm_activation=common.NormActivation(
              norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
      losses=Losses(l2_weight_decay=1e-4),
      train_data=train_dataset,
      validation_data=validation_dataset)
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=task,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])
  add_trainer(config, train_batch_size=1024, eval_batch_size=64)
  return config


@exp_factory.register_config_factory('video_classification_kinetics700_2020')
def video_classification_kinetics700_2020() -> cfg.ExperimentConfig:
  """Video classification on Kinetics 700 2020 with resnet."""
  train_dataset = kinetics700_2020(is_training=True)
  validation_dataset = kinetics700_2020(is_training=False)
  task = VideoClassificationTask(
      model=VideoClassificationModel(
          backbone=backbones_3d.Backbone3D(
              type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50()),
          norm_activation=common.NormActivation(
              norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
      losses=Losses(l2_weight_decay=1e-4),
      train_data=train_dataset,
      validation_data=validation_dataset)
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=task,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])
  add_trainer(config, train_batch_size=1024, eval_batch_size=64)
  return config
