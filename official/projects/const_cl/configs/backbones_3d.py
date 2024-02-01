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
from typing import Tuple

from official.vision.configs import backbones_3d


ResNet3DBlock = backbones_3d.ResNet3DBlock


@dataclasses.dataclass
class ResNet3DY(backbones_3d.ResNet3D):
  pass


@dataclasses.dataclass
class ResNet3DY50(ResNet3DY):
  """Block specifications of the Resnet50 (3DY) model."""
  model_id: int = 50
  block_specs: Tuple[
      ResNet3DBlock, ResNet3DBlock, ResNet3DBlock, ResNet3DBlock] = (
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(3, 3, 3),
                        use_self_gating=True),
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(3, 1, 3, 1),
                        use_self_gating=True),
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(3, 1, 3, 1, 3, 1),
                        use_self_gating=True),
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(1, 3, 1),
                        use_self_gating=True))


@dataclasses.dataclass
class Backbone3D(backbones_3d.Backbone3D):
  """Configuration for backbones.

  Attributes:
    type: type of backbone be used, one of the fields below.
    resnet_3dy: resnet_3dy backbone config.
  """
  type: str = 'resnet_3dy'
  resnet_3dy: ResNet3DY = dataclasses.field(default_factory=ResNet3DY50)
