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

"""Decoders configurations."""
import dataclasses

from official.modeling import hyperparams
from official.vision.configs import decoders


@dataclasses.dataclass
class MaskConverFPN(hyperparams.Config):
  """FPN config."""
  num_filters: int = 256
  fusion_type: str = 'sum'
  use_separable_conv: bool = False
  use_keras_layer: bool = False
  use_layer_norm: bool = True
  depthwise_kernel_size: int = 7


@dataclasses.dataclass
class Decoder(decoders.Decoder):
  maskconver_fpn: MaskConverFPN = dataclasses.field(
      default_factory=MaskConverFPN)
