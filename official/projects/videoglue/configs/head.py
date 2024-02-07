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

"""Configs for different model heads."""

import dataclasses
from typing import Optional

from official.modeling import hyperparams


@dataclasses.dataclass
class MLP(hyperparams.Config):
  """Config for the MLP head."""
  normalize_inputs: bool = True
  num_hidden_channels: int = 2048
  num_hidden_layers: int = 3
  num_output_channels: int = 128
  use_sync_bn: bool = False
  norm_momentum: float = 0.997
  norm_epsilon: float = 1e-5
  activation: Optional[str] = 'relu'


@dataclasses.dataclass
class ActionTransformer(hyperparams.Config):
  """Config for the action transformer head."""
  # parameters for classifier
  num_hidden_layers: int = 0
  num_hidden_channels: int = 0
  use_sync_bn: bool = True
  activation: str = 'relu'
  dropout_rate: float = 0.0
  # parameters for RoiAligner
  crop_size: int = 4
  sample_offset: float = 0.5
  # parameters for TxDecoder
  num_tx_channels: int = 128
  num_tx_layers: int = 3
  num_tx_heads: int = 3
  use_bias: bool = True
  tx_activation: Optional[str] = 'gelu'
  attention_dropout_rate: float = 0.0
  layer_norm_epsilon: float = 1e-12
  use_positional_embedding: bool = True
