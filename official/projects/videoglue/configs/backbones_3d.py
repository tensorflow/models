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

"""3D Backbones configurations."""
import dataclasses
from typing import Tuple, Union, Optional

from official.vision.configs import backbones
from official.vision.configs import backbones_3d


@dataclasses.dataclass
class VisionTransformer3D(backbones.VisionTransformer):
  """VisionTransformer3D config."""
  variant: str = 'native'
  temporal_patch_size: int = 4
  pos_embed_shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None


@dataclasses.dataclass
class Backbone3D(backbones_3d.Backbone3D):
  """Configuration for backbones.

  Attributes:
    type: type of backbone be used, one of the fields below.
    vit_3d: vit_3d backbone config.
  """
  type: str = 'vit_3d'
  vit_3d: VisionTransformer3D = dataclasses.field(
      default_factory=VisionTransformer3D)
