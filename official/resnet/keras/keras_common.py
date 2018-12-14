# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_run_loop
from official.resnet.keras import keras_resnet_model
from official.resnet.keras import resnet_model_tpu
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size):
    """Callback for Keras models.

    Args:
      batch_size: Total batch size.

    """
    self._batch_size = batch_size
    super(TimeHistory, self).__init__()

  def on_train_begin(self, logs=None):
    self.epoch_times_secs = []
    self.batch_times_secs = []
    self.record_batch = True

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_times_secs.append(time.time() - self.epoch_time_start)

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      self.batch_time_start = time.time()
      self.record_batch = False

  def on_batch_end(self, batch, logs=None):
    n = 100
    if batch % n == 0:
      last_n_batches = time.time() - self.batch_time_start
      examples_per_second = (self._batch_size * n) / last_n_batches
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
    #if not hasattr(self.model.optimizer, 'learning_rate'):
    #  raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    lr = self.schedule(self.epochs, batch, self.batches_per_epoch, self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
      self.prev_lr = lr
      tf.logging.debug('Epoch %05d Batch %05d: LearningRateBatchScheduler change '
                   'learning rate to %s.', self.epochs, batch, lr)

def get_optimizer_loss_and_metrics():
  # Use Keras ResNet50 applications model and native keras APIs
  # initialize RMSprop optimizer
  # TODO(anjalisridhar): Move to using MomentumOptimizer.
  # opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
  # I am setting an initial LR of 0.001 since this will be reset
  # at the beginning of the training loop.
  opt = gradient_descent_v2.SGD(learning_rate=0.1, momentum=0.9)

  # TF Optimizer:
  # learning_rate = BASE_LEARNING_RATE * flags_obj.batch_size / 256
  # opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
  loss = 'categorical_crossentropy'
  accuracy = 'categorical_accuracy'

  return opt, loss, accuracy


def get_dist_strategy():
  if flags_obj.num_gpus == 1 and flags_obj.dist_strat_off:
    print('Not using distribution strategies.')
    strategy = None
  else:
    strategy = distribution_utils.get_distribution_strategy(
        num_gpus=flags_obj.num_gpus)

  return strategy
  

def get_fit_callbacks():
  time_callback = keras_common.TimeHistory(flags_obj.batch_size)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=flags_obj.model_dir)
    #update_freq="batch")  # Add this if want per batch logging.

  lr_callback = keras_common.LearningRateBatchScheduler(
    learning_rate_schedule,
    batch_size=flags_obj.batch_size,
    num_images=imagenet_main._NUM_IMAGES['train'])

  return time_callback, tensorboard_callback, lr_callback

def analyze_eval_result(eval_output):
  stats = {}
  stats['accuracy_top_1'] = eval_output[1]
  stats['eval_loss'] = eval_output[0]
  stats['training_loss'] = history.history['loss'][-1]
  stats['training_accuracy_top_1'] = history.history['categorical_accuracy'][-1]

  print('top_1 accuracy:{}'.format(stats['accuracy_top_1']))
  print('top_1_training_accuracy:{}'.format(stats['training_accuracy_top_1']))

  return stats