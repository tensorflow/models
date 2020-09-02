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
"""Backbones configurations."""
from typing import Optional

# Import libraries
import dataclasses

from official.modeling import hyperparams


@dataclasses.dataclass
class ResNet(hyperparams.Config):
  """ResNet config."""
  model_id: int = 50


@dataclasses.dataclass
class EfficientNet(hyperparams.Config):
  """EfficientNet config."""
  model_id: str = 'b0'
  stochastic_depth_drop_rate: float = 0.0
  se_ratio: float = 0.0


@dataclasses.dataclass
class SpineNet(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'


@dataclasses.dataclass
class RevNet(hyperparams.Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: int = 56


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one the of fields below.
    resnet: resnet backbone config.
    revnet: revnet backbone config.
    efficientnet: efficientnet backbone config.
    spinenet: spinenet backbone config.
  """
  type: Optional[str] = None
  resnet: ResNet = ResNet()
  revnet: RevNet = RevNet()
  efficientnet: EfficientNet = EfficientNet()
  spinenet: SpineNet = SpineNet()
