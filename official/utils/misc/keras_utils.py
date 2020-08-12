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
"""Helper functions for the Keras implementations of models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import time

from absl import logging
import tensorflow as tf

from tensorflow.python.eager import monitoring

global_batch_size_gauge = monitoring.IntGauge(
    '/tensorflow/training/global_batch_size', 'TF training global batch size')


class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps, initial_step=0, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      initial_step: Optional, initial step.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.last_log_step = initial_step
    self.steps_before_epoch = initial_step
    self.steps_in_epoch = 0
    self.start_time = None

    global_batch_size_gauge.get_cell().set(batch_size)

    if logdir:
      self.summary_writer = tf.summary.create_file_writer(logdir)
    else:
      self.summary_writer = None

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

    # Records the time each epoch takes to run from start to finish of epoch.
    self.epoch_runtime_log = []

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  @property
  def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return self.global_steps / sum(self.epoch_runtime_log)

  @property
  def average_examples_per_second(self):
    """The average number of training examples per second across all epochs."""
    return self.average_steps_per_second * self.batch_size

  def get_examples_per_sec(self, warmup=1):
    """Calculates examples/sec through timestamp_log and skip warmup period."""
    # First entry in timestamp_log is the start of the step 1. The rest of the
    # entries are the end of each step recorded.
    time_log = self.timestamp_log
    seconds = time_log[-1].timestamp - time_log[warmup].timestamp
    steps = time_log[-1].batch_index - time_log[warmup].batch_index
    return self.batch_size * steps / seconds

  def get_startup_time(self, start_time_sec):
    return self.timestamp_log[0].timestamp - start_time_sec

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

    if self.summary_writer:
      self.summary_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
      self.start_time = time.time()

    # Record the timestamp of the first global step
    if not self.timestamp_log:
      self.timestamp_log.append(
          BatchTimestamp(self.global_steps, self.start_time))

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
      now = time.time()
      elapsed_time = now - self.start_time
      steps_per_second = steps_since_last_log / elapsed_time
      examples_per_second = steps_per_second * self.batch_size

      self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
      logging.info(
          'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
          'and %d', elapsed_time, examples_per_second, self.last_log_step,
          self.global_steps)

      if self.summary_writer:
        with self.summary_writer.as_default():
          tf.summary.scalar('steps_per_second', steps_per_second,
                            self.global_steps)
          tf.summary.scalar('examples_per_second', examples_per_second,
                            self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0


class SimpleCheckpoint(tf.keras.callbacks.Callback):
  """Keras callback to save tf.train.Checkpoints."""

  def __init__(self, checkpoint_manager):
    super(SimpleCheckpoint, self).__init__()
    self.checkpoint_manager = checkpoint_manager

  def on_epoch_end(self, epoch, logs=None):
    step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
    self.checkpoint_manager.save(checkpoint_number=step_counter)


def set_session_config(enable_xla=False):
  """Sets the session config."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)


# TODO(hongkuny): remove set_config_v2 globally.
set_config_v2 = set_session_config


def set_gpu_thread_mode_and_count(gpu_thread_mode, datasets_num_private_threads,
                                  num_gpus, per_gpu_thread_count):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  logging.info('Logical CPU cores: %s', cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = per_gpu_thread_count or 2
  os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  logging.info('TF_GPU_THREAD_COUNT: %s', os.environ['TF_GPU_THREAD_COUNT'])
  logging.info('TF_GPU_THREAD_MODE: %s', os.environ['TF_GPU_THREAD_MODE'])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * num_gpus
  num_runtime_threads = num_gpus
  if not datasets_num_private_threads:
    datasets_num_private_threads = min(
        cpu_count - total_gpu_thread_count - num_runtime_threads, num_gpus * 8)
    logging.info('Set datasets_num_private_threads to %s',
                 datasets_num_private_threads)
