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

"""Multi-head BERT encoder network with classification heads.

Includes configurations and instantiation methods.
"""
from typing import List, Optional, Text

import dataclasses

from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders


@dataclasses.dataclass
class ClsHeadConfig(base_config.Config):
  inner_dim: int = 0
  num_classes: int = 2
  activation: Optional[Text] = "tanh"
  dropout_rate: float = 0.0
  cls_token_idx: int = 0
  name: Optional[Text] = None


@dataclasses.dataclass
class PretrainerConfig(base_config.Config):
  """Pretrainer configuration."""
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  cls_heads: List[ClsHeadConfig] = dataclasses.field(default_factory=list)
  mlm_activation: str = "gelu"
  mlm_initializer_range: float = 0.02
  # Currently only used for mobile bert.
  mlm_output_weights_use_proj: bool = False
