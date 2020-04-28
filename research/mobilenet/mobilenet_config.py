# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Configuration definitions for MobileNet."""

from dataclasses import dataclass
from official.modeling.hyperparams import base_config


@dataclass
class MobileNetV1Config(base_config.Config):
  """Configuration for the MobileNetV1 model."""
  pass


@dataclass
class MobileNetV2Config(base_config.Config):
  """Configuration for the MobileNetV2 model."""
  pass


@dataclass
class MobileNetV3Config(base_config.Config):
  """Configuration for the MobileNetV3 model."""
  pass
