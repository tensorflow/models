# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Dataclasses for learning rate schedule config."""
from typing import List, Optional

import dataclasses
from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class ConstantLrConfig(base_config.Config):
  """Configuration for constant learning rate.

  This class is a containers for the constant learning rate decay configs.

  Attributes:
    name: The name of the learning rate schedule. Defaults to Constant.
    learning_rate: A float. The learning rate. Defaults to 0.1.
  """
  name: str = 'Constant'
  learning_rate: float = 0.1


@dataclasses.dataclass
class StepwiseLrConfig(base_config.Config):
  """Configuration for stepwise learning rate decay.

  This class is a container for the piecewise constant learning rate scheduling
  configs. It will configure an instance of PiecewiseConstantDecay keras
  learning rate schedule.

  An example (from keras docs): use a learning rate that's 1.0 for the first
  100001 steps, 0.5 for the next 10000 steps, and 0.1 for any additional steps.
    ```python
    boundaries: [100000, 110000]
    values: [1.0, 0.5, 0.1]

  Attributes:
    name: The name of the learning rate schedule. Defaults to PiecewiseConstant.
    boundaries: A list of ints of strictly increasing entries.
    Defaults to None.
    values: A list of floats that specifies the values for the intervals defined
            by `boundaries`. It should have one more element than `boundaries`.
            The learning rate is computed as follows:
              [0, boundaries[0]]                 -> values[0]
              [boundaries[0], boundaries[1]]     -> values[1]
              [boundaries[n-1], boundaries[n]]   -> values[n]
              [boundaries[n], end]               -> values[n+1]
            Defaults to None.
  """
  name: str = 'PiecewiseConstantDecay'
  boundaries: Optional[List[int]] = None
  values: Optional[List[float]] = None


@dataclasses.dataclass
class ExponentialLrConfig(base_config.Config):
  """Configuration for exponential learning rate decay.

  This class is a containers for the exponential learning rate decay configs.

  Attributes:
    name: The name of the learning rate schedule. Defaults to ExponentialDecay.
    initial_learning_rate: A float. The initial learning rate. Defaults to
                           None.
    decay_steps: A positive integer that is used for decay computation.
                 Defaults to None.
    decay_rate: A float. Defaults to None.
    staircase: A boolean, if true, learning rate is decreased at discreate
               intervals. Defaults to False.
  """
  name: str = 'ExponentialDecay'
  initial_learning_rate: Optional[float] = None
  decay_steps: Optional[int] = None
  decay_rate: Optional[float] = None
  staircase: Optional[bool] = None


@dataclasses.dataclass
class PolynomialLrConfig(base_config.Config):
  """Configuration for polynomial learning rate decay.

  This class is a containers for the polynomial learning rate decay configs.

  Attributes:
    name: The name of the learning rate schedule. Defaults to PolynomialDecay.
    initial_learning_rate: A float. The initial learning rate. Defaults to
                           None.
    decay_steps: A positive integer that is used for decay computation.
                 Defaults to None.
    end_learning_rate: A float.  The minimal end learning rate.
    power: A float.  The power of the polynomial. Defaults to linear, 1.0.
    cycle: A boolean, whether or not it should cycle beyond decay_steps.
           Defaults to False.
  """
  name: str = 'PolynomialDecay'
  initial_learning_rate: Optional[float] = None
  decay_steps: Optional[int] = None
  end_learning_rate: float = 0.0001
  power: float = 1.0
  cycle: bool = False


@dataclasses.dataclass
class CosineLrConfig(base_config.Config):
  """Configuration for Cosine learning rate decay.

  This class is a containers for the cosine learning rate decay configs,
  tf.keras.experimental.CosineDecay.

  Attributes:
    name: The name of the learning rate schedule. Defaults to CosineDecay.
    initial_learning_rate: A float. The initial learning rate. Defaults to
                           None.
    decay_steps: A positive integer that is used for decay computation.
                 Defaults to None.
    alpha: A float.  Minimum learning rate value as a fraction of
                     initial_learning_rate.
  """
  name: str = 'CosineDecay'
  initial_learning_rate: Optional[float] = None
  decay_steps: Optional[int] = None
  alpha: float = 0.0


@dataclasses.dataclass
class LinearWarmupConfig(base_config.Config):
  """Configuration for linear warmup schedule config.

  This class is a container for the linear warmup schedule configs.
  Warmup_learning_rate is the initial learning rate, the final learning rate of
  the warmup period is the learning_rate of the optimizer in use. The learning
  rate at each step linearly increased according to the following formula:
    warmup_learning_rate = warmup_learning_rate +
    step / warmup_steps * (final_learning_rate - warmup_learning_rate).
  Using warmup overrides the learning rate schedule by the number of warmup
  steps.

  Attributes:
    name: The name of warmup schedule. Defaults to linear.
    warmup_learning_rate: Initial learning rate for the warmup. Defaults to 0.
    warmup_steps: Warmup steps. Defaults to None.
  """
  name: str = 'linear'
  warmup_learning_rate: float = 0
  warmup_steps: Optional[int] = None


@dataclasses.dataclass
class PolynomialWarmupConfig(base_config.Config):
  """Configuration for linear warmup schedule config.

  This class is a container for the polynomial warmup schedule configs.

  Attributes:
    name: The name of warmup schedule. Defaults to Polynomial.
    power: Polynomial power. Defaults to 1.
    warmup_steps: Warmup steps. Defaults to None.
  """
  name: str = 'polynomial'
  power: float = 1
  warmup_steps: Optional[int] = None

