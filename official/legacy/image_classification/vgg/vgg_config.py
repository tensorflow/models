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
"""Configuration definitions for VGG losses, learning rates, and optimizers."""

import dataclasses
from official.legacy.image_classification.configs import base_configs
from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class VGGModelConfig(base_configs.ModelConfig):
  """Configuration for the VGG model."""
  name: str = 'VGG'
  num_classes: int = 1000
  model_params: base_config.Config = dataclasses.field(default_factory=lambda: {   # pylint:disable=g-long-lambda
      'num_classes': 1000,
      'batch_size': None,
      'use_l2_regularizer': True
  })
  loss: base_configs.LossConfig = base_configs.LossConfig(
      name='sparse_categorical_crossentropy')
  optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
      name='momentum', epsilon=0.001, momentum=0.9, moving_average_decay=None)
  learning_rate: base_configs.LearningRateConfig = (
      base_configs.LearningRateConfig(
          name='stepwise',
          initial_lr=0.01,
          examples_per_epoch=1281167,
          boundaries=[30, 60],
          warmup_epochs=0,
          scale_by_batch_size=1. / 256.,
          multipliers=[0.01 / 256, 0.001 / 256, 0.0001 / 256]))
