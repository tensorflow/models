# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import gin

from official.modeling.optimization import ema_optimizer
from official.modeling.optimization import optimizer_factory
from official.projects.yolo.optimization import sgd_torch

optimizer_factory.LEGACY_OPTIMIZERS_CLS.update({
    'sgd_torch': sgd_torch.SGDTorch,
})

OPTIMIZERS_CLS = optimizer_factory.LEGACY_OPTIMIZERS_CLS
LR_CLS = optimizer_factory.LR_CLS
WARMUP_CLS = optimizer_factory.WARMUP_CLS


class OptimizerFactory(optimizer_factory.OptimizerFactory):
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
  """

  def get_bias_lr_schedule(self, bias_lr):
    """Build learning rate.

    Builds learning rate from config. Learning rate schedule is built according
    to the learning rate config. If learning rate type is consant,
    lr_config.learning_rate is returned.

    Args:
      bias_lr: learning rate config.

    Returns:
      tf_keras.optimizers.schedules.LearningRateSchedule instance. If
      learning rate type is consant, lr_config.learning_rate is returned.
    """
    if self._lr_type == 'constant':
      lr = self._lr_config.learning_rate
    else:
      lr = LR_CLS[self._lr_type](**self._lr_config.as_dict())

    if self._warmup_config:
      if self._warmup_type != 'linear':
        raise ValueError('Smart Bias is only supported currently with a'
                         'linear warm up.')
      warm_up_cfg = self._warmup_config.as_dict()
      warm_up_cfg['warmup_learning_rate'] = bias_lr
      lr = WARMUP_CLS['linear'](lr, **warm_up_cfg)
    return lr

  @gin.configurable
  def add_ema(self, optimizer):
    """Add EMA to the optimizer independently of the build optimizer method."""
    if self._use_ema:
      optimizer = ema_optimizer.ExponentialMovingAverage(
          optimizer, **self._ema_config.as_dict())
    return optimizer
