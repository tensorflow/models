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
"""Configuration utils for image classification experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataclasses

from official.vision.image_classification import dataset_factory
from official.vision.image_classification.configs import base_configs
from official.vision.image_classification.efficientnet import efficientnet_config
from official.vision.image_classification.resnet import resnet_config


@dataclasses.dataclass
class EfficientNetImageNetConfig(base_configs.ExperimentConfig):
  """Base configuration to train efficientnet-b0 on ImageNet.

  Attributes:
    export: An `ExportConfig` instance
    runtime: A `RuntimeConfig` instance.
    dataset: A `DatasetConfig` instance.
    train: A `TrainConfig` instance.
    evaluation: An `EvalConfig` instance.
    model: A `ModelConfig` instance.
  """
  export: base_configs.ExportConfig = base_configs.ExportConfig()
  runtime: base_configs.RuntimeConfig = base_configs.RuntimeConfig()
  train_dataset: dataset_factory.DatasetConfig = \
      dataset_factory.ImageNetConfig(split='train')
  validation_dataset: dataset_factory.DatasetConfig = \
      dataset_factory.ImageNetConfig(split='validation')
  train: base_configs.TrainConfig = base_configs.TrainConfig(
      resume_checkpoint=True,
      epochs=500,
      steps=None,
      callbacks=base_configs.CallbacksConfig(
          enable_checkpoint_and_export=True, enable_tensorboard=True),
      metrics=['accuracy', 'top_5'],
      time_history=base_configs.TimeHistoryConfig(log_steps=100),
      tensorboard=base_configs.TensorBoardConfig(
          track_lr=True, write_model_weights=False),
      set_epoch_loop=False)
  evaluation: base_configs.EvalConfig = base_configs.EvalConfig(
      epochs_between_evals=1, steps=None)
  model: base_configs.ModelConfig = \
    efficientnet_config.EfficientNetModelConfig()


@dataclasses.dataclass
class ResNetImagenetConfig(base_configs.ExperimentConfig):
  """Base configuration to train resnet-50 on ImageNet."""
  export: base_configs.ExportConfig = base_configs.ExportConfig()
  runtime: base_configs.RuntimeConfig = base_configs.RuntimeConfig()
  train_dataset: dataset_factory.DatasetConfig = \
      dataset_factory.ImageNetConfig(split='train',
                                     one_hot=False,
                                     mean_subtract=True,
                                     standardize=True)
  validation_dataset: dataset_factory.DatasetConfig = \
      dataset_factory.ImageNetConfig(split='validation',
                                     one_hot=False,
                                     mean_subtract=True,
                                     standardize=True)
  train: base_configs.TrainConfig = base_configs.TrainConfig(
      resume_checkpoint=True,
      epochs=90,
      steps=None,
      callbacks=base_configs.CallbacksConfig(
          enable_checkpoint_and_export=True, enable_tensorboard=True),
      metrics=['accuracy', 'top_5'],
      time_history=base_configs.TimeHistoryConfig(log_steps=100),
      tensorboard=base_configs.TensorBoardConfig(
          track_lr=True, write_model_weights=False),
      set_epoch_loop=False)
  evaluation: base_configs.EvalConfig = base_configs.EvalConfig(
      epochs_between_evals=1, steps=None)
  model: base_configs.ModelConfig = resnet_config.ResNetModelConfig()


def get_config(model: str, dataset: str) -> base_configs.ExperimentConfig:
  """Given model and dataset names, return the ExperimentConfig."""
  dataset_model_config_map = {
      'imagenet': {
          'efficientnet': EfficientNetImageNetConfig(),
          'resnet': ResNetImagenetConfig(),
      }
  }
  try:
    return dataset_model_config_map[dataset][model]
  except KeyError:
    if dataset not in dataset_model_config_map:
      raise KeyError('Invalid dataset received. Received: {}. Supported '
                     'datasets include: {}'.format(
                         dataset, ', '.join(dataset_model_config_map.keys())))
    raise KeyError('Invalid model received. Received: {}. Supported models for'
                   '{} include: {}'.format(
                       model, dataset,
                       ', '.join(dataset_model_config_map[dataset].keys())))
