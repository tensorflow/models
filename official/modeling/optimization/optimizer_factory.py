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

"""Optimizer factory class."""
from typing import Callable, Optional, Union, List, Tuple

import gin
import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers

from official.modeling.optimization import slide_optimizer
from official.modeling.optimization import adafactor_optimizer
from official.modeling.optimization import ema_optimizer
from official.modeling.optimization import lars_optimizer
from official.modeling.optimization import legacy_adamw
from official.modeling.optimization import lr_schedule
from official.modeling.optimization.configs import optimization_config as opt_cfg

# Optimizer CLS to be used in both legacy and new path.
SHARED_OPTIMIZERS = {
    'sgd_experimental': tf.keras.optimizers.experimental.SGD,
    'adam_experimental': tf.keras.optimizers.experimental.Adam,
    'adamw': legacy_adamw.AdamWeightDecay,
    'adamw_experimental': tf.keras.optimizers.experimental.AdamW,
    'lamb': tfa_optimizers.LAMB,
    'lars': lars_optimizer.LARS,
    'slide': slide_optimizer.SLIDE,
    'adafactor': adafactor_optimizer.Adafactor,
}

LEGACY_OPTIMIZERS_CLS = {
    'sgd': tf.keras.optimizers.legacy.SGD,
    'adam': tf.keras.optimizers.legacy.Adam,
    'rmsprop': tf.keras.optimizers.legacy.RMSprop,
    'adagrad': tf.keras.optimizers.legacy.Adagrad,
}
LEGACY_OPTIMIZERS_CLS.update(SHARED_OPTIMIZERS)

NEW_OPTIMIZERS_CLS = {
    'sgd': tf.keras.optimizers.experimental.SGD,
    'adam': tf.keras.optimizers.experimental.Adam,
    'rmsprop': tf.keras.optimizers.experimental.RMSprop,
    'adagrad': tf.keras.optimizers.experimental.Adagrad,
}
NEW_OPTIMIZERS_CLS.update(SHARED_OPTIMIZERS)

LR_CLS = {
    'stepwise': lr_schedule.PiecewiseConstantDecayWithOffset,
    'polynomial': lr_schedule.PolynomialDecayWithOffset,
    'exponential': lr_schedule.ExponentialDecayWithOffset,
    'cosine': lr_schedule.CosineDecayWithOffset,
    'power': lr_schedule.DirectPowerDecay,
    'power_linear': lr_schedule.PowerAndLinearDecay,
    'power_with_offset': lr_schedule.PowerDecayWithOffset,
    'step_cosine_with_offset': lr_schedule.StepCosineDecayWithOffset,
}

WARMUP_CLS = {
    'linear': lr_schedule.LinearWarmup,
    'polynomial': lr_schedule.PolynomialWarmUp
}


def register_optimizer_cls(key: str,
                           optimizer_config_cls: Union[
                               tf.keras.optimizers.Optimizer,
                               tf.keras.optimizers.legacy.Optimizer,
                               tf.keras.optimizers.experimental.Optimizer
                           ],
                           use_legacy_optimizer: bool = True):
  """Register customize optimizer cls.

  The user will still need to subclass data classes in
  configs.optimization_config to be used with OptimizerFactory.

  Args:
    key: A string to that the optimizer_config_cls is registered with.
    optimizer_config_cls: A class which inherits tf.keras.optimizers.Optimizer.
    use_legacy_optimizer: A boolean that indicates if using legacy optimizers.
  """
  if use_legacy_optimizer:
    if key in LEGACY_OPTIMIZERS_CLS:
      raise ValueError('%s already registered in LEGACY_OPTIMIZERS_CLS.' % key)
    LEGACY_OPTIMIZERS_CLS[key] = optimizer_config_cls
  else:
    if key in NEW_OPTIMIZERS_CLS:
      raise ValueError('%s already registered in NEW_OPTIMIZERS_CLS.' % key)
    NEW_OPTIMIZERS_CLS[key] = optimizer_config_cls


class OptimizerFactory:
  """Optimizer factory class.

  This class builds learning rate and optimizer based on an optimization config.
  To use this class, you need to do the following:
  (1) Define optimization config, this includes optimizer, and learning rate
      schedule.
  (2) Initialize the class using the optimization config.
  (3) Build learning rate.
  (4) Build optimizer.

  This is a typical example for using this class:

  ```
  params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'momentum': 0.9}
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
  ```
  """

  def __init__(self, config: opt_cfg.OptimizationConfig):
    """Initializing OptimizerFactory.

    Args:
      config: OptimizationConfig instance contain optimization config.
    """
    self._config = config
    self._optimizer_config = config.optimizer.get()
    self._optimizer_type = config.optimizer.type

    self._use_ema = config.ema is not None
    self._ema_config = config.ema

    if self._optimizer_config is None:
      raise ValueError('Optimizer type must be specified')

    self._lr_config = config.learning_rate.get()
    self._lr_type = config.learning_rate.type

    if self._lr_type is None:
      raise ValueError('Learning rate type must be specified')

    self._warmup_config = config.warmup.get()
    self._warmup_type = config.warmup.type

  def build_learning_rate(self):
    """Build learning rate.

    Builds learning rate from config. Learning rate schedule is built according
    to the learning rate config. If learning rate type is consant,
    lr_config.learning_rate is returned.

    Returns:
      tf.keras.optimizers.schedules.LearningRateSchedule instance. If
      learning rate type is consant, lr_config.learning_rate is returned.
    """
    if self._lr_type == 'constant':
      lr = self._lr_config.learning_rate
    else:
      lr = LR_CLS[self._lr_type](**self._lr_config.as_dict())

    if self._warmup_config:
      lr = WARMUP_CLS[self._warmup_type](lr, **self._warmup_config.as_dict())

    return lr

  @gin.configurable
  def build_optimizer(
      self,
      lr: Union[tf.keras.optimizers.schedules.LearningRateSchedule, float],
      gradient_aggregator: Optional[Callable[
          [List[Tuple[tf.Tensor, tf.Tensor]]], List[Tuple[tf.Tensor,
                                                          tf.Tensor]]]] = None,
      gradient_transformers: Optional[List[Callable[
          [List[Tuple[tf.Tensor, tf.Tensor]]], List[Tuple[tf.Tensor,
                                                          tf.Tensor]]]]] = None,
      postprocessor: Optional[Callable[[tf.keras.optimizers.Optimizer],
                                       tf.keras.optimizers.Optimizer]] = None,
      use_legacy_optimizer: bool = True):
    """Build optimizer.

    Builds optimizer from config. It takes learning rate as input, and builds
    the optimizer according to the optimizer config. Typically, the learning
    rate built using self.build_lr() is passed as an argument to this method.

    Args:
      lr: A floating point value, or a
        tf.keras.optimizers.schedules.LearningRateSchedule instance.
      gradient_aggregator: Optional function to overwrite gradient aggregation.
      gradient_transformers: Optional list of functions to use to transform
        gradients before applying updates to Variables. The functions are
        applied after gradient_aggregator. The functions should accept and
        return a list of (gradient, variable) tuples. clipvalue, clipnorm,
        global_clipnorm should not be set when gradient_transformers is passed.
      postprocessor: An optional function for postprocessing the optimizer. It
        takes an optimizer and returns an optimizer.
      use_legacy_optimizer: A boolean that indicates if using legacy optimizers.

    Returns:
      `tf.keras.optimizers.legacy.Optimizer` or
      `tf.keras.optimizers.experimental.Optimizer` instance.
    """

    optimizer_dict = self._optimizer_config.as_dict()
    ## Delete clipnorm, clipvalue, global_clipnorm if None
    if optimizer_dict['clipnorm'] is None:
      del optimizer_dict['clipnorm']
    if optimizer_dict['clipvalue'] is None:
      del optimizer_dict['clipvalue']
    if optimizer_dict['global_clipnorm'] is None:
      del optimizer_dict['global_clipnorm']

    optimizer_dict['learning_rate'] = lr
    if gradient_aggregator is not None:
      optimizer_dict['gradient_aggregator'] = gradient_aggregator
    if gradient_transformers is not None:
      optimizer_dict['gradient_transformers'] = gradient_transformers

    if use_legacy_optimizer:
      optimizer = LEGACY_OPTIMIZERS_CLS[self._optimizer_type](**optimizer_dict)
    else:
      if 'decay' in optimizer_dict:
        raise ValueError(
            '`decay` is deprecated in new Keras optimizer, please reflect the '
            'decay logic in `lr` or set `use_legacy_optimizer=True` to use the '
            'legacy optimizer.')
      optimizer = NEW_OPTIMIZERS_CLS[self._optimizer_type](**optimizer_dict)

    if self._use_ema:
      if not use_legacy_optimizer:
        raise ValueError(
            'EMA can only work with the legacy optimizer, please set '
            '`use_legacy_optimizer=True`.')
      optimizer = ema_optimizer.ExponentialMovingAverage(
          optimizer, **self._ema_config.as_dict())
    if postprocessor:
      optimizer = postprocessor(optimizer)
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
      return optimizer
    # The following check makes sure the function won't break in older TF
    # version because of missing the experimental/legacy package.
    if hasattr(tf.keras.optimizers, 'experimental'):
      if isinstance(optimizer, tf.keras.optimizers.experimental.Optimizer):
        return optimizer
    if hasattr(tf.keras.optimizers, 'legacy'):
      if isinstance(optimizer, tf.keras.optimizers.legacy.Optimizer):
        return optimizer
    raise TypeError('OptimizerFactory.build_optimizer returning a '
                    'non-optimizer object: {}'.format(optimizer))
