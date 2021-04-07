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
class BASNet_En(hyperparams.Config):
  """BASNet Encoder config."""
  model_id: str = 'BASNet_En'


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one the of fields below.
    basnet_en: basnet encoder config.
  """
  type: Optional[str] = None
  basnet_en: BASNet_En = BASNet_En()
