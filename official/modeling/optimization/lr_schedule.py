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
"""Learning rate schedule classes."""

from typing import Mapping, Any, Union, Optional

import tensorflow as tf


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Linear warmup schedule."""

  def __init__(self, after_warmup_lr_sched: Union[
      tf.keras.optimizers.schedules.LearningRateSchedule, float],
               warmup_steps: int, warmup_learning_rate: float,
               name: Optional[str] = None):
    """Add linear warmup schedule to a learning rate schedule.

    warmup_lr is the initial learning rate, the final learning rate of the
    init_warmup period is the initial learning rate of lr_schedule in use.
    The learning rate at each step linearly increased according to the following
    formula:
      learning_rate = warmup_lr + step / warmup_steps
                    * (final_warmup_lr - warmup_lr).
    Using warmup overrides the learning rate schedule by the number of warmup
    steps.

    Args:
      after_warmup_lr_sched: tf.keras.optimizers.schedules
                                .LearningRateSchedule or a constant.
      warmup_steps: int. number of the warmup steps.
      warmup_learning_rate: floating point number. Initial learning rate for the
                      warmup.
      name: Optional, name of warmup schedule.
    """
    super(LinearWarmup, self).__init__()
    self._name = name
    self._after_warmup_lr_sched = after_warmup_lr_sched
    self._warmup_steps = warmup_steps
    self._init_warmup_lr = warmup_learning_rate
    if isinstance(after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      self._final_warmup_lr = after_warmup_lr_sched(warmup_steps)
    else:
      self._final_warmup_lr = tf.cast(
          after_warmup_lr_sched, dtype=tf.float32)

  def __call__(self, step: int):

    global_step = tf.cast(step, dtype=tf.float32)

    linear_warmup_lr = (
        self._init_warmup_lr + global_step / self._warmup_steps *
        (self._final_warmup_lr - self._init_warmup_lr))

    if isinstance(self._after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      after_warmup_lr = self._after_warmup_lr_sched(step)
    else:
      after_warmup_lr = tf.cast(self._after_warmup_lr_sched, dtype=tf.float32)

    lr = tf.cond(global_step < self._warmup_steps,
                 lambda: linear_warmup_lr,
                 lambda: after_warmup_lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    if isinstance(self._after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      config = {
          "after_warmup_lr_sched": self._after_warmup_lr_sched.get_config()}  # pytype: disable=attribute-error
    else:
      config = {"after_warmup_lr_sched": self._after_warmup_lr_sched}  # pytype: disable=attribute-error

    config.update({
        "warmup_steps": self._warmup_steps,
        "warmup_learning_rate": self._init_warmup_lr,
        "name": self._name
    })
    return config


class PolynomialWarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies polynomial warmup schedule on a given learning rate decay schedule.
  """

  def __init__(self,
               after_warmup_lr_sched: Union[
                   tf.keras.optimizers.schedules.LearningRateSchedule, float],
               warmup_steps: int,
               power: float = 1.0,
               name: str = "PolynomialWarmup"):
    super(PolynomialWarmUp, self).__init__()
    if isinstance(after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      self._initial_learning_rate = after_warmup_lr_sched(warmup_steps)
    else:
      self._initial_learning_rate = tf.cast(
          after_warmup_lr_sched, dtype=tf.float32)

    self._warmup_steps = warmup_steps
    self._power = power
    self._after_warmup_lr_sched = after_warmup_lr_sched
    self._name = name

  def __call__(self, step):
    with tf.name_scope(self._name or "PolynomialWarmUp") as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self._warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self._initial_learning_rate *
          tf.math.pow(warmup_percent_done, self._power))

      if isinstance(self._after_warmup_lr_sched,
                    tf.keras.optimizers.schedules.LearningRateSchedule):
        after_warmup_lr = self._after_warmup_lr_sched(step)
      else:
        after_warmup_lr = tf.cast(self._after_warmup_lr_sched, dtype=tf.float32)

      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: after_warmup_lr,
          name=name)

  def get_config(self) -> Mapping[str, Any]:
    if isinstance(self._after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      config = {
          "after_warmup_lr_sched": self._after_warmup_lr_sched.get_config()}  # pytype: disable=attribute-error
    else:
      config = {"after_warmup_lr_sched": self._after_warmup_lr_sched}  # pytype: disable=attribute-error

    config.update({
        "warmup_steps": self._warmup_steps,
        "power": self._power,
        "name": self._name
    })
    return config


class DirectPowerDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule follows lr * (step)^power."""

  def __init__(self,
               initial_learning_rate: float,
               power: float = 1.0,
               name: str = "DirectPowerDecay"):
    """Initialize configuration of the learning rate schedule.

    Args:
      initial_learning_rate: A float, the initial learning rate.
      power: A float, the number of steps required for linear warmup.
      name: Optional, name of warmup schedule.
    """
    super(DirectPowerDecay, self).__init__()
    self._initial_learning_rate = initial_learning_rate
    self._power = power
    self._name = name

  def __call__(self, step):
    with tf.name_scope(self._name or "DirectPowerDecay"):
      step = tf.cast(step, tf.float32)
      learning_rate = self._initial_learning_rate
      learning_rate *= tf.math.pow(step, self._power)
      return learning_rate

  def get_config(self):
    """Get the configuration of the learning rate schedule."""
    return {
        "initial_learning_rate": self._initial_learning_rate,
        "power": self._power,
        "name": self._name,
    }
