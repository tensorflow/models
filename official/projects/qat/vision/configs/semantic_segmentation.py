# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""RetinaNet configuration definition."""
import dataclasses
from typing import Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.qat.vision.configs import common
from official.vision.configs import semantic_segmentation


@dataclasses.dataclass
class SemanticSegmentationTask(semantic_segmentation.SemanticSegmentationTask):
  quantization: Optional[common.Quantization] = None


@exp_factory.register_config_factory('mnv2_deeplabv3_pascal_qat')
def mnv2_deeplabv3_pascal() -> cfg.ExperimentConfig:
  """Generates a config for MobileNet v2 + deeplab v3 with QAT."""
  config = semantic_segmentation.mnv2_deeplabv3_pascal()
  task = SemanticSegmentationTask.from_args(
      quantization=common.Quantization(), **config.task.as_dict())
  config.task = task
  return config


@exp_factory.register_config_factory('mnv2_deeplabv3_cityscapes_qat')
def mnv2_deeplabv3_cityscapes() -> cfg.ExperimentConfig:
  """Generates a config for MobileNet v2 + deeplab v3 with QAT."""
  config = semantic_segmentation.mnv2_deeplabv3_cityscapes()
  task = SemanticSegmentationTask.from_args(
      quantization=common.Quantization(), **config.task.as_dict())
  config.task = task
  return config


@exp_factory.register_config_factory('mnv2_deeplabv3plus_cityscapes_qat')
def mnv2_deeplabv3plus_cityscapes() -> cfg.ExperimentConfig:
  """Generates a config for MobileNet v2 + deeplab v3+ with QAT."""
  config = semantic_segmentation.mnv2_deeplabv3plus_cityscapes()
  task = SemanticSegmentationTask.from_args(
      quantization=common.Quantization(), **config.task.as_dict())
  config.task = task
  return config
