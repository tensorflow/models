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

"""Mosaic configuration definition."""
import dataclasses
from typing import Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.projects.mosaic.configs import mosaic_config
from official.projects.qat.vision.configs import common


@dataclasses.dataclass
class MosaicSemanticSegmentationTask(
    mosaic_config.MosaicSemanticSegmentationTask):
  quantization: Optional[common.Quantization] = None


@exp_factory.register_config_factory('mosaic_mnv35_cityscapes_qat')
def mosaic_mnv35_cityscapes() -> cfg.ExperimentConfig:
  """Experiment configuration of image segmentation task with QAT."""
  config = mosaic_config.mosaic_mnv35_cityscapes()
  task = MosaicSemanticSegmentationTask.from_args(
      quantization=common.Quantization(), **config.task.as_dict())
  config.task = task
  return config
