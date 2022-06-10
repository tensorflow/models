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

"""Optimizer and learning rate scheduler."""

import tensorflow as tf

from official.modeling.hyperparams import params_dict


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""

  def __init__(self, initial_learning_rate, hidden_size, warmup_steps):
    """Initialize configuration of the learning rate schedule.

    Args:
      initial_learning_rate: A float, the initial learning rate.
      hidden_size: An integer, the model dimension in the hidden layers.
      warmup_steps: An integer, the number of steps required for linear warmup.
    """
    super(LearningRateSchedule, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.hidden_size = hidden_size
    self.warmup_steps = tf.cast(warmup_steps, tf.float32)

  def __call__(self, global_step):
    """Calculate learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: An integer, the current global step used for learning rate
        calculation.

    Returns:
      A float, the learning rate needs to be used for current global step.
    """
    with tf.name_scope('learning_rate_schedule'):
      global_step = tf.cast(global_step, tf.float32)
      learning_rate = self.initial_learning_rate
      learning_rate *= (self.hidden_size**-0.5)
      # Apply linear warmup
      learning_rate *= tf.minimum(1.0, global_step / self.warmup_steps)
      # Apply rsqrt decay
      learning_rate /= tf.sqrt(tf.maximum(global_step, self.warmup_steps))
      return learning_rate

  def get_config(self):
    """Get the configuration of the learning rate schedule."""
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'hidden_size': self.hidden_size,
        'warmup_steps': self.warmup_steps,
    }


def create_optimizer(params: params_dict.ParamsDict):
  """Creates optimizer."""
  lr_schedule = LearningRateSchedule(params.learning_rate, params.hidden_size,
                                     params.learning_rate_warmup_steps)
  return tf.keras.optimizers.Adam(
      learning_rate=lr_schedule,
      beta_1=params.adam_beta1,
      beta_2=params.adam_beta2,
      epsilon=params.adam_epsilon)
