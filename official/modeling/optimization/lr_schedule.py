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
      name = "{!s}WithWarmup".format(self._after_warmup_lr_sched.name)  # pytype: disable=attribute-error
      config = self._after_warmup_lr_sched.get_config()  # pytype: disable=attribute-error
    else:
      name = "ConstantWithWarmup"
      config = {"learning_rate": self._after_warmup_lr_sched}

    config.update({
        "warmup_steps": self._warmup_steps,
        "warmup_learning_rate": self._init_warmup_lr,
        "name": name
    })
    return config
