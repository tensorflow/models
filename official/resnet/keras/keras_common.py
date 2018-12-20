# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Common util functions and classes used by both keras cifar and imagenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import flags
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2


FLAGS = flags.FLAGS
BASE_LEARNING_RATE = 0.1  # This matches Jing's version.

class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size):
    """Callback for logging performance (# image/second).

    Args:
      batch_size: Total batch size.

    """
    self._batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_batch_size = 100

  def on_train_begin(self, logs=None):
    self.batch_times_secs = []
    self.record_batch = True

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      self.batch_time_start = time.time()
      self.record_batch = False

  def on_batch_end(self, batch, logs=None):
    if batch % self.log_batch_size == 0:
      last_n_batches = time.time() - self.batch_time_start
      examples_per_second = 
        (self._batch_size * self.log_batch_size) / last_n_batches
      self.batch_times_secs.append(last_n_batches)
      self.record_batch = True
      # TODO(anjalisridhar): add timestamp as well.
      if batch != 0:
        tf.logging.info("BenchmarkMetric: {'num_batches':%d, 'time_taken': %f,"
                        "'images_per_second': %f}" %
                        (batch, last_n_batches, examples_per_second))

class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, num_images):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.batches_per_epoch = num_images / batch_size
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'learning_rate'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    lr = self.schedule(self.epochs, batch, self.batches_per_epoch, self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      # tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
      self.prev_lr = lr
      tf.logging.debug('Epoch %05d Batch %05d: LearningRateBatchScheduler change '
                   'learning rate to %s.', self.epochs, batch, lr)

def get_optimizer():
  if FLAGS.use_tf_momentum_optimizer:
    learning_rate = BASE_LEARNING_RATE * FLAGS.batch_size / 256
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
  else:
    optimizer = gradient_descent_v2.SGD(learning_rate=0.1, momentum=0.9)

  return optimizer


def get_callbacks(learning_rate_schedule_fn, num_images):
  time_callback = TimeHistory(FLAGS.batch_size)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=FLAGS.model_dir)

  lr_callback = LearningRateBatchScheduler(
    learning_rate_schedule_fn,
    batch_size=FLAGS.batch_size,
    num_images=num_images)

  return time_callback, tensorboard_callback, lr_callback

def analyze_fit_and_eval_result(history, eval_output):
  stats = {}
  stats['accuracy_top_1'] = eval_output[1]
  stats['eval_loss'] = eval_output[0]
  stats['training_loss'] = history.history['loss'][-1]
  stats['training_accuracy_top_1'] = history.history['categorical_accuracy'][-1]

  print('Test loss:{}'.format(stats['']))
  print('top_1 accuracy:{}'.format(stats['accuracy_top_1']))
  print('top_1_training_accuracy:{}'.format(stats['training_accuracy_top_1']))

  return stats
