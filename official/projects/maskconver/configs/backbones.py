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

"""Backbones configurations."""
import dataclasses

from typing import List, Optional
from official.modeling import hyperparams
from official.vision.configs.google import backbones


@dataclasses.dataclass
class ResNetUNet(hyperparams.Config):
  """ResNetUNet config."""
  model_id: int = 50
  depth_multiplier: float = 1.0
  stem_type: str = 'v0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0
  scale_stem: bool = True
  resnetd_shortcut: bool = False
  replace_stem_max_pool: bool = False
  bn_trainable: bool = True
  classification_output: bool = False
  upsample_kernel_sizes: Optional[List[int]] = None
  upsample_repeats: Optional[List[int]] = None
  upsample_filters: Optional[List[int]] = None


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  resnet_unet: ResNetUNet = dataclasses.field(default_factory=ResNetUNet)
