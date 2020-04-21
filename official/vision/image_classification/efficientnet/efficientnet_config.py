# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Configuration definitions for EfficientNet losses, learning rates, and optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping

import dataclasses

from official.modeling.hyperparams import base_config
from official.vision.image_classification.configs import base_configs


@dataclasses.dataclass
class EfficientNetModelConfig(base_configs.ModelConfig):
  """Configuration for the EfficientNet model.

  This configuration will default to settings used for training efficientnet-b0
  on a v3-8 TPU on ImageNet.

  Attributes:
    name: The name of the model. Defaults to 'EfficientNet'.
    num_classes: The number of classes in the model.
    model_params: A dictionary that represents the parameters of the
      EfficientNet model. These will be passed in to the "from_name" function.
    loss: The configuration for loss. Defaults to a categorical cross entropy
      implementation.
    optimizer: The configuration for optimizations. Defaults to an RMSProp
      configuration.
    learning_rate: The configuration for learning rate. Defaults to an
      exponential configuration.
  """
  name: str = 'EfficientNet'
  num_classes: int = 1000
  model_params: base_config.Config = dataclasses.field(
      default_factory=lambda: {
          'model_name': 'efficientnet-b0',
          'model_weights_path': '',
          'weights_format': 'saved_model',
          'overrides': {
              'batch_norm': 'default',
              'rescale_input': True,
              'num_classes': 1000,
              'activation': 'swish',
              'dtype': 'float32',
          }
      })
  loss: base_configs.LossConfig = base_configs.LossConfig(
      name='categorical_crossentropy', label_smoothing=0.1)
  optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
      name='rmsprop',
      decay=0.9,
      epsilon=0.001,
      momentum=0.9,
      moving_average_decay=None)
  learning_rate: base_configs.LearningRateConfig = base_configs.LearningRateConfig(  # pylint: disable=line-too-long
      name='exponential',
      initial_lr=0.008,
      decay_epochs=2.4,
      decay_rate=0.97,
      warmup_epochs=5,
      scale_by_batch_size=1. / 128.,
      staircase=True)
