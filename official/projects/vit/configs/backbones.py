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
from typing import Optional, Tuple

from official.modeling import hyperparams


@dataclasses.dataclass
class Transformer(hyperparams.Config):
  """Transformer config."""
  mlp_dim: int = 1
  num_heads: int = 1
  num_layers: int = 1
  attention_dropout_rate: float = 0.0
  dropout_rate: float = 0.1


@dataclasses.dataclass
class VisionTransformer(hyperparams.Config):
  """VisionTransformer config."""
  model_name: str = 'vit-b16'
  # pylint: disable=line-too-long
  pooler: str = 'token'  # 'token', 'gap' or 'none'. If set to 'token', an extra classification token is added to sequence.
  # pylint: enable=line-too-long
  representation_size: int = 0
  hidden_size: int = 1
  patch_size: int = 16
  transformer: Transformer = Transformer()
  init_stochastic_depth_rate: float = 0.0
  original_init: bool = True
  pos_embed_shape: Optional[Tuple[int, int]] = None


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one the of fields below.
    vit: vit backbone config.
  """
  type: Optional[str] = None
  vit: VisionTransformer = VisionTransformer()
