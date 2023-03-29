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

"""Dataclasses for optimization configs.

This file define the dataclass for optimization configs (OptimizationConfig).
It also has two helper functions get_optimizer_config, and get_lr_config from
an OptimizationConfig class.
"""
import dataclasses
from typing import Optional

from official.modeling.optimization.configs import optimization_config as optimization_cfg
from official.projects.yolo.optimization.configs import optimizer_config as opt_cfg


@dataclasses.dataclass
class OptimizerConfig(optimization_cfg.OptimizerConfig):
  """Configuration for optimizer.

  Attributes:
    type: 'str', type of optimizer to be used, on the of fields below.
    sgd: sgd optimizer config.
    adam: adam optimizer config.
    adamw: adam with weight decay.
    lamb: lamb optimizer.
    rmsprop: rmsprop optimizer.
  """
  type: Optional[str] = None
  sgd_torch: opt_cfg.SGDTorchConfig = opt_cfg.SGDTorchConfig()


@dataclasses.dataclass
class OptimizationConfig(optimization_cfg.OptimizationConfig):
  """Configuration for optimizer and learning rate schedule.

  Attributes:
    optimizer: optimizer oneof config.
    ema: optional exponential moving average optimizer config, if specified, ema
      optimizer will be used.
    learning_rate: learning rate oneof config.
    warmup: warmup oneof config.
  """
  type: Optional[str] = None
  optimizer: OptimizerConfig = OptimizerConfig()
