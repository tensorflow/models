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

"""Learning rate utilities for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping, Optional

import numpy as np
import tensorflow as tf, tf_keras

BASE_LEARNING_RATE = 0.1


class WarmupDecaySchedule(tf_keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(self,
               lr_schedule: tf_keras.optimizers.schedules.LearningRateSchedule,
               warmup_steps: int,
               warmup_lr: Optional[float] = None):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps
      warmup_lr: an optional field for the final warmup learning rate. This
        should be provided if the base `lr_schedule` does not contain this
        field.
    """
    super(WarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr

  def __call__(self, step: int):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      if self._warmup_lr is not None:
        initial_learning_rate = tf.convert_to_tensor(
            self._warmup_lr, name="initial_learning_rate")
      else:
        initial_learning_rate = tf.convert_to_tensor(
            self._lr_schedule.initial_learning_rate,
            name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps, lambda: warmup_lr,
                   lambda: lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    config = self._lr_schedule.get_config()
    config.update({
        "warmup_steps": self._warmup_steps,
        "warmup_lr": self._warmup_lr,
    })
    return config


class CosineDecayWithWarmup(tf_keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self, batch_size: int, total_steps: int, warmup_steps: int):
    """Creates the cosine learning rate tensor with linear warmup.

    Args:
      batch_size: The training batch size used in the experiment.
      total_steps: Total training steps.
      warmup_steps: Steps for the warm up period.
    """
    super(CosineDecayWithWarmup, self).__init__()
    base_lr_batch_size = 256
    self._total_steps = total_steps
    self._init_learning_rate = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self._warmup_steps = warmup_steps

  def __call__(self, global_step: int):
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_steps = self._warmup_steps
    init_lr = self._init_learning_rate
    total_steps = self._total_steps

    linear_warmup = global_step / warmup_steps * init_lr

    cosine_learning_rate = init_lr * (tf.cos(np.pi *
                                             (global_step - warmup_steps) /
                                             (total_steps - warmup_steps)) +
                                      1.0) / 2.0

    learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
                             cosine_learning_rate)
    return learning_rate

  def get_config(self):
    return {
        "total_steps": self._total_steps,
        "warmup_learning_rate": self._warmup_learning_rate,
        "warmup_steps": self._warmup_steps,
        "init_learning_rate": self._init_learning_rate,
    }
