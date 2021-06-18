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

"""Dataclasses for optimization configs.

This file define the dataclass for optimization configs (OptimizationConfig).
It also has two helper functions get_optimizer_config, and get_lr_config from
an OptimizationConfig class.
"""
from typing import Optional

import dataclasses

from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import oneof
from official.modeling.optimization.configs import learning_rate_config as lr_cfg
from official.modeling.optimization.configs import optimizer_config as opt_cfg


@dataclasses.dataclass
class OptimizerConfig(oneof.OneOfConfig):
  """Configuration for optimizer.

  Attributes:
    type: 'str', type of optimizer to be used, on the of fields below.
    sgd: sgd optimizer config.
    adam: adam optimizer config.
    adamw: adam with weight decay.
    lamb: lamb optimizer.
    rmsprop: rmsprop optimizer.
    lars: lars optimizer.
    adagrad: adagrad optimizer.
    slide: slide optimizer.
  """
  type: Optional[str] = None
  sgd: opt_cfg.SGDConfig = opt_cfg.SGDConfig()
  adam: opt_cfg.AdamConfig = opt_cfg.AdamConfig()
  adamw: opt_cfg.AdamWeightDecayConfig = opt_cfg.AdamWeightDecayConfig()
  lamb: opt_cfg.LAMBConfig = opt_cfg.LAMBConfig()
  rmsprop: opt_cfg.RMSPropConfig = opt_cfg.RMSPropConfig()
  lars: opt_cfg.LARSConfig = opt_cfg.LARSConfig()
  adagrad: opt_cfg.AdagradConfig = opt_cfg.AdagradConfig()
  slide: opt_cfg.SLIDEConfig = opt_cfg.SLIDEConfig()


@dataclasses.dataclass
class LrConfig(oneof.OneOfConfig):
  """Configuration for lr schedule.

  Attributes:
    type: 'str', type of lr schedule to be used, one of the fields below.
    constant: constant learning rate config.
    stepwise: stepwise learning rate config.
    exponential: exponential learning rate config.
    polynomial: polynomial learning rate config.
    cosine: cosine learning rate config.
    power: step^power learning rate config.
    power_linear: learning rate config of step^power followed by
      step^power*linear.
    power_with_offset: power decay with a step offset.
  """
  type: Optional[str] = None
  constant: lr_cfg.ConstantLrConfig = lr_cfg.ConstantLrConfig()
  stepwise: lr_cfg.StepwiseLrConfig = lr_cfg.StepwiseLrConfig()
  exponential: lr_cfg.ExponentialLrConfig = lr_cfg.ExponentialLrConfig()
  polynomial: lr_cfg.PolynomialLrConfig = lr_cfg.PolynomialLrConfig()
  cosine: lr_cfg.CosineLrConfig = lr_cfg.CosineLrConfig()
  power: lr_cfg.DirectPowerLrConfig = lr_cfg.DirectPowerLrConfig()
  power_linear: lr_cfg.PowerAndLinearDecayLrConfig = (
      lr_cfg.PowerAndLinearDecayLrConfig())
  power_with_offset: lr_cfg.PowerDecayWithOffsetLrConfig = (
      lr_cfg.PowerDecayWithOffsetLrConfig())


@dataclasses.dataclass
class WarmupConfig(oneof.OneOfConfig):
  """Configuration for lr schedule.

  Attributes:
    type: 'str', type of warmup schedule to be used, one of the fields below.
    linear: linear warmup config.
    polynomial: polynomial warmup config.
  """
  type: Optional[str] = None
  linear: lr_cfg.LinearWarmupConfig = lr_cfg.LinearWarmupConfig()
  polynomial: lr_cfg.PolynomialWarmupConfig = lr_cfg.PolynomialWarmupConfig()


@dataclasses.dataclass
class OptimizationConfig(base_config.Config):
  """Configuration for optimizer and learning rate schedule.

  Attributes:
    optimizer: optimizer oneof config.
    ema: optional exponential moving average optimizer config, if specified, ema
      optimizer will be used.
    learning_rate: learning rate oneof config.
    warmup: warmup oneof config.
  """
  optimizer: OptimizerConfig = OptimizerConfig()
  ema: Optional[opt_cfg.EMAConfig] = None
  learning_rate: LrConfig = LrConfig()
  warmup: WarmupConfig = WarmupConfig()
