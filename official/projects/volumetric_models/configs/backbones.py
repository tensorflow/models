# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from typing import Optional, Sequence

from official.modeling import hyperparams


@dataclasses.dataclass
class UNet3D(hyperparams.Config):
  """UNet3D config."""
  model_id: int = 4
  pool_size: Sequence[int] = (2, 2, 2)
  kernel_size: Sequence[int] = (3, 3, 3)
  base_filters: int = 32
  use_batch_normalization: bool = True


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one the of fields below.
    unet_3d: UNet3D backbone config.
  """
  type: Optional[str] = None
  unet_3d: UNet3D = UNet3D()
