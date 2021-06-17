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

"""Learning rate schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf
from official.modeling.hyperparams import params_dict


class StepLearningRateWithLinearWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self, total_steps, params):
    """Creates the step learning rate tensor with linear warmup."""
    super(StepLearningRateWithLinearWarmup, self).__init__()
    self._total_steps = total_steps
    assert isinstance(params, (dict, params_dict.ParamsDict))
    if isinstance(params, dict):
      params = params_dict.ParamsDict(params)
    self._params = params

  def __call__(self, global_step):
    warmup_lr = self._params.warmup_learning_rate
    warmup_steps = self._params.warmup_steps
    init_lr = self._params.init_learning_rate
    lr_levels = self._params.learning_rate_levels
    lr_steps = self._params.learning_rate_steps
    linear_warmup = (
        warmup_lr + tf.cast(global_step, dtype=tf.float32) / warmup_steps *
        (init_lr - warmup_lr))
    learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)

    for next_learning_rate, start_step in zip(lr_levels, lr_steps):
      learning_rate = tf.where(global_step >= start_step, next_learning_rate,
                               learning_rate)
    return learning_rate

  def get_config(self):
    return {'_params': self._params.as_dict()}


class CosineLearningRateWithLinearWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self, total_steps, params):
    """Creates the consine learning rate tensor with linear warmup."""
    super(CosineLearningRateWithLinearWarmup, self).__init__()
    self._total_steps = total_steps
    assert isinstance(params, (dict, params_dict.ParamsDict))
    if isinstance(params, dict):
      params = params_dict.ParamsDict(params)
    self._params = params

  def __call__(self, global_step):
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_lr = self._params.warmup_learning_rate
    warmup_steps = self._params.warmup_steps
    init_lr = self._params.init_learning_rate
    total_steps = self._total_steps
    linear_warmup = (
        warmup_lr + global_step / warmup_steps * (init_lr - warmup_lr))
    cosine_learning_rate = (
        init_lr * (tf.cos(np.pi * (global_step - warmup_steps) /
                          (total_steps - warmup_steps)) + 1.0) / 2.0)
    learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
                             cosine_learning_rate)
    return learning_rate

  def get_config(self):
    return {'_params': self._params.as_dict()}


def learning_rate_generator(total_steps, params):
  """The learning rate function generator."""
  if params.type == 'step':
    return StepLearningRateWithLinearWarmup(total_steps, params)
  elif params.type == 'cosine':
    return CosineLearningRateWithLinearWarmup(total_steps, params)
  else:
    raise ValueError('Unsupported learning rate type: {}.'.format(params.type))
