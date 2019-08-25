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
"""Optimizer from addons and learning rate scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
K = tf.keras.backend


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


class LearningRateFn(object):
  """Creates learning rate function."""

  def __init__(self, learning_rate, hidden_size, warmup_steps):
    self.learning_rate = learning_rate
    self.hidden_size = hidden_size
    self.warmup_steps = float(warmup_steps)

  def __call__(self, global_step):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    step = float(global_step)
    learning_rate = self.learning_rate
    learning_rate *= (self.hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= np.minimum(1.0, step / self.warmup_steps)
    # Apply rsqrt decay
    learning_rate /= np.sqrt(np.maximum(step, self.warmup_steps))
    return learning_rate


class LearningRateScheduler(tf.keras.callbacks.Callback):
  """Keras callback to schedule learning rate.

  TODO(tianlin): Refactor this scheduler and LearningRateBatchScheduler in
  official/resnet/keras/keras_common.py.
  """

  def __init__(self, schedule, init_steps=None, verbose=False):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose
    if init_steps is None:
      init_steps = 0.0
    self.steps = float(init_steps)   # Total steps during training.

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    if not hasattr(self.model.optimizer, 'iterations'):
      raise ValueError('Optimizer must have a "iterations" attribute.')

  def on_train_batch_begin(self, batch, logs=None):
    """Adjusts learning rate for each train batch."""
    if self.verbose > 0:
      iterations = K.get_value(self.model.optimizer.iterations)
      print('Original iteration %d' % iterations)

    self.steps += 1.0
    try:  # new API
      lr = float(K.get_value(self.model.optimizer.lr))
      lr = self.schedule(self.steps, lr)
    except TypeError:  # Support for old API for backward compatibility
      lr = self.schedule(self.steps)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    K.set_value(self.model.optimizer.lr, lr)
    K.set_value(self.model.optimizer.iterations, self.steps)

    if self.verbose > 0:
      print('Batch %05d Step %05d: LearningRateScheduler setting learning '
            'rate to %s.' % (batch + 1, self.steps, lr))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    logs['steps'] = self.steps
