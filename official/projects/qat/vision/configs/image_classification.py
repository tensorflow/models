# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
from typing import Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.qat.vision.configs import common
from official.vision.configs import image_classification


@dataclasses.dataclass
class ImageClassificationTask(image_classification.ImageClassificationTask):
  quantization: Optional[common.Quantization] = None


@exp_factory.register_config_factory('resnet_imagenet_qat')
def image_classification_imagenet() -> cfg.ExperimentConfig:
  """Builds an image classification config for the resnet with QAT."""
  config = image_classification.image_classification_imagenet()
  task = ImageClassificationTask.from_args(
      quantization=common.Quantization(), **config.task.as_dict())
  config.task = task
  runtime = cfg.RuntimeConfig(enable_xla=False)
  config.runtime = runtime

  return config


@exp_factory.register_config_factory('mobilenet_imagenet_qat')
def image_classification_imagenet_mobilenet() -> cfg.ExperimentConfig:
  """Builds an image classification config for the mobilenetV2 with QAT."""
  config = image_classification.image_classification_imagenet_mobilenet()
  task = ImageClassificationTask.from_args(
      quantization=common.Quantization(), **config.task.as_dict())
  config.task = task

  return config
