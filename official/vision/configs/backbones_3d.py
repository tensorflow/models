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
from typing import Optional, Tuple

# Import libraries
import dataclasses

from official.modeling import hyperparams


@dataclasses.dataclass
class ResNet3DBlock(hyperparams.Config):
  """Configuration of a ResNet 3D block."""
  temporal_strides: int = 1
  temporal_kernel_sizes: Tuple[int, ...] = ()
  use_self_gating: bool = False


@dataclasses.dataclass
class ResNet3D(hyperparams.Config):
  """ResNet config."""
  model_id: int = 50
  stem_type: str = 'v0'
  stem_conv_temporal_kernel_size: int = 5
  stem_conv_temporal_stride: int = 2
  stem_pool_temporal_stride: int = 2
  block_specs: Tuple[ResNet3DBlock, ...] = ()
  stochastic_depth_drop_rate: float = 0.0
  se_ratio: float = 0.0


@dataclasses.dataclass
class ResNet3D50(ResNet3D):
  """Block specifications of the Resnet50 (3D) model."""
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
class ResNet3DRS(ResNet3D):
  """Block specifications of the ResNet-RS (3D) model."""
  model_id: int = 50
  stem_type: str = 'v1'
  stem_conv_temporal_kernel_size: int = 5
  stem_conv_temporal_stride: int = 2
  stem_pool_temporal_stride: int = 2
  stochastic_depth_drop_rate: float = 0.1
  se_ratio: float = 0.2
  block_specs: Tuple[
      ResNet3DBlock, ResNet3DBlock, ResNet3DBlock, ResNet3DBlock] = (
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(1,),
                        use_self_gating=True),
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(1,),
                        use_self_gating=True),
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(3,),
                        use_self_gating=True),
          ResNet3DBlock(temporal_strides=1,
                        temporal_kernel_sizes=(3,),
                        use_self_gating=True))


@dataclasses.dataclass
class Backbone3D(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    resnet_3d: resnet3d backbone config.
    resnet_3d_rs: resnet3d-rs backbone config.
  """
  type: Optional[str] = None
  resnet_3d: ResNet3D = dataclasses.field(default_factory=ResNet3D50)
  resnet_3d_rs: ResNet3D = dataclasses.field(default_factory=ResNet3DRS)
