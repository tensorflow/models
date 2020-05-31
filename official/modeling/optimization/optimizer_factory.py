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
"""Optimizer factory class."""

from typing import Union

import tensorflow as tf

import tensorflow_addons.optimizers as tfa_optimizers

from official.modeling.optimization import lr_schedule
from official.modeling.optimization.configs import optimization_config as opt_cfg
from official.nlp import optimization as nlp_optimization

OPTIMIZERS_CLS = {
    'sgd': tf.keras.optimizers.SGD,
    'adam': tf.keras.optimizers.Adam,
    'adamw': nlp_optimization.AdamWeightDecay,
    'lamb': tfa_optimizers.LAMB
}

LR_CLS = {
    'stepwise': tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    'polynomial': tf.keras.optimizers.schedules.PolynomialDecay,
    'exponential': tf.keras.optimizers.schedules.ExponentialDecay,
}

WARMUP_CLS = {
    'linear': lr_schedule.LinearWarmup
}


class OptimizerFactory(object):
  """Optimizer factory class.

  This class builds learning rate and optimizer based on an optimization config.
  To use this class, you need to do the following:
  (1) Define optimization config, this includes optimizer, and learning rate
      schedule.
  (2) Initialize the class using the optimization config.
  (3) Build learning rate.
  (4) Build optimizer.

  This is a typical example for using this class:
  params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'learning_rate': 0.1, 'momentum': 0.9}
        },
        'learning_rate': {
            'type': 'stepwise',
            'stepwise': {'boundaries': [10000, 20000],
                         'values': [0.1, 0.01, 0.001]}
        },
        'warmup': {
            'type': 'linear',
            'linear': {'warmup_steps': 500, 'warmup_learning_rate': 0.01}
        }
    }
  opt_config = OptimizationConfig(params)
  opt_factory = OptimizerFactory(opt_config)
  lr = opt_factory.build_learning_rate()
  optimizer = opt_factory.build_optimizer(lr)
  """

  def __init__(self, config: opt_cfg.OptimizationConfig):
    """Initializing OptimizerFactory.

    Args:
      config: OptimizationConfig instance contain optimization config.
    """
    self._config = config
    self._optimizer_config = config.optimizer.get()
    self._optimizer_type = config.optimizer.type

    if self._optimizer_config is None:
      raise ValueError('Optimizer type must be specified')

    self._lr_config = config.learning_rate.get()
    self._lr_type = config.learning_rate.type

    self._warmup_config = config.warmup.get()
    self._warmup_type = config.warmup.type

  def build_learning_rate(self):
    """Build learning rate.

    Builds learning rate from config. Learning rate schedule is built according
    to the learning rate config. If there is no learning rate config, optimizer
    learning rate is returned.

    Returns:
      tf.keras.optimizers.schedules.LearningRateSchedule instance. If no
      learning rate schedule defined, optimizer_config.learning_rate is
      returned.
    """

    if not self._lr_config:
      lr = self._optimizer_config.learning_rate
    else:
      lr = LR_CLS[self._lr_type](**self._lr_config.as_dict())

    if self._warmup_config:
      lr = WARMUP_CLS[self._warmup_type](lr, **self._warmup_config.as_dict())

    return lr

  def build_optimizer(
      self, lr: Union[tf.keras.optimizers.schedules.LearningRateSchedule,
                      float]):
    """Build optimizer.

    Builds optimizer from config. It takes learning rate as input, and builds
    the optimizer according to the optimizer config. Typically, the learning
    rate built using self.build_lr() is passed as an argument to this method.

    Args:
      lr: A floating point value, or
          a tf.keras.optimizers.schedules.LearningRateSchedule instance.
    Returns:
      tf.keras.optimizers.Optimizer instance.
    """

    optimizer_dict = self._optimizer_config.as_dict()
    optimizer_dict['learning_rate'] = lr

    optimizer = OPTIMIZERS_CLS[self._optimizer_type](**optimizer_dict)
    return optimizer

