# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Dataclasses for optimizer configs."""
from typing import List, Optional

import dataclasses
from official.modeling.hyperparams import base_config
from official.modeling.optimization.configs.optimizer_config import BaseOptimizerConfig


@dataclasses.dataclass
class BaseOptimizerConfig(base_config.Config):
  """Base optimizer config.

  Attributes:
    clipnorm: float >= 0 or None. If not None, Gradients will be clipped when
      their L2 norm exceeds this value.
    clipvalue: float >= 0 or None. If not None, Gradients will be clipped when
      their absolute value exceeds this value.
    global_clipnorm: float >= 0 or None. If not None, gradient of all weights is
        clipped so that their global norm is no higher than this value
  """
  clipnorm: Optional[float] = None
  clipvalue: Optional[float] = None
  global_clipnorm: Optional[float] = None


@dataclasses.dataclass
class SGDTorchConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the arguments of tf.keras.optimizer.SGD.

  Attributes:
    name: name of the optimizer.
    decay: decay rate for SGD optimizer.
    nesterov: nesterov for SGD optimizer.
    momentum_start: momentum starting point for SGD optimizer.
    momentum: momentum for SGD optimizer.
  """
  name: str = "SGD"
  decay: float = 0.0
  nesterov: bool = False
  momentum_start: float = 0.0
  momentum: float = 0.9
  warmup_steps: int = 1000
  weight_decay: float = 0.0
  sim_torch: bool = False
