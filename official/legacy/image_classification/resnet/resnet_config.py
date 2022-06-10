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

"""Configuration definitions for ResNet losses, learning rates, and optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataclasses
from official.legacy.image_classification.configs import base_configs
from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class ResNetModelConfig(base_configs.ModelConfig):
  """Configuration for the ResNet model."""
  name: str = 'ResNet'
  num_classes: int = 1000
  model_params: base_config.Config = dataclasses.field(
      # pylint: disable=g-long-lambda
      default_factory=lambda: {
          'num_classes': 1000,
          'batch_size': None,
          'use_l2_regularizer': True,
          'rescale_inputs': False,
      })
  # pylint: enable=g-long-lambda
  loss: base_configs.LossConfig = base_configs.LossConfig(
      name='sparse_categorical_crossentropy')
  optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
      name='momentum',
      decay=0.9,
      epsilon=0.001,
      momentum=0.9,
      moving_average_decay=None)
  learning_rate: base_configs.LearningRateConfig = (
      base_configs.LearningRateConfig(
          name='stepwise',
          initial_lr=0.1,
          examples_per_epoch=1281167,
          boundaries=[30, 60, 80],
          warmup_epochs=5,
          scale_by_batch_size=1. / 256.,
          multipliers=[0.1 / 256, 0.01 / 256, 0.001 / 256, 0.0001 / 256]))
