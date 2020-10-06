# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Video classification configuration definition."""
from typing import Optional, Tuple
import dataclasses
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import backbones_3d
from official.vision.beta.configs import common


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """The base configuration for building datasets."""
  name: Optional[str] = None
  file_type: Optional[str] = 'tfrecord'
  compressed_input: bool = False
  split: str = 'train'
  feature_shape: Tuple[int, ...] = (64, 224, 224, 3)
  temporal_stride: int = 1
  num_test_clips: int = 1
  num_classes: int = -1
  num_channels: int = 3
  num_examples: int = -1
  global_batch_size: int = 128
  num_devices: int = 1
  data_format: str = 'channels_last'
  dtype: str = 'float32'
  one_hot: bool = True
  shuffle_buffer_size: int = 64
  cache: bool = False
  input_path: str = ''
  is_training: bool = True
  cycle_length: int = 10
  min_image_size: int = 256


def kinetics600(is_training):
  """Generated Kinectics 600 dataset configs."""
  return DataConfig(
      name='kinetics600',
      num_classes=600,
      is_training=is_training,
      split='train' if is_training else 'valid',
      num_examples=366016 if is_training else 27780,
      feature_shape=(64, 224, 224, 3) if is_training else (250, 224, 224, 3))


@dataclasses.dataclass
class VideoClassificationModel(hyperparams.Config):
  """The model config."""
  model_type: str = 'video_classification'
  backbone: backbones_3d.Backbone3D = backbones_3d.Backbone3D(
      type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50())
  norm_activation: common.NormActivation = common.NormActivation()
  dropout_rate: float = 0.2
  add_head_batch_norm: bool = False


@dataclasses.dataclass
class Losses(hyperparams.Config):
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class VideoClassificationTask(cfg.TaskConfig):
  """The task config."""
  model: VideoClassificationModel = VideoClassificationModel()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  losses: Losses = Losses()
  gradient_clip_norm: float = -1.0


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


@exp_factory.register_config_factory('video_classification_kinetics600')
def video_classification_kinetics600() -> cfg.ExperimentConfig:
  """Video classification on Videonet with resnet."""
  train_dataset = kinetics600(is_training=True)
  validation_dataset = kinetics600(is_training=False)
  task = VideoClassificationTask(
      model=VideoClassificationModel(
          backbone=backbones_3d.Backbone3D(
              type='resnet_3d', resnet_3d=backbones_3d.ResNet3D50()),
          norm_activation=common.NormActivation(
              norm_momentum=0.9, norm_epsilon=1e-5)),
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
