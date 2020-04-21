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
import tensorflow.compat.v2 as tf
from tensorflow.python import tf2
from tensorflow.python.profiler import profiler_v2 as profiler


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

  def __init__(self, batch_size, log_steps, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.last_log_step = 0
    self.steps_before_epoch = 0
    self.steps_in_epoch = 0
    self.start_time = None

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
      self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                               self.start_time))

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
          tf.summary.scalar('global_step/sec', steps_per_second,
                            self.global_steps)
          tf.summary.scalar('examples/sec', examples_per_second,
                            self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0


def get_profiler_callback(model_dir, profile_steps, enable_tensorboard,
                          steps_per_epoch):
  """Validate profile_steps flag value and return profiler callback."""
  profile_steps_error_message = (
      'profile_steps must be a comma separated pair of positive integers, '
      'specifying the first and last steps to be profiled.'
  )
  try:
    profile_steps = [int(i) for i in profile_steps.split(',')]
  except ValueError:
    raise ValueError(profile_steps_error_message)
  if len(profile_steps) != 2:
    raise ValueError(profile_steps_error_message)
  start_step, stop_step = profile_steps
  if start_step < 0 or start_step > stop_step:
    raise ValueError(profile_steps_error_message)
  if enable_tensorboard:
    logging.warning(
        'Both TensorBoard and profiler callbacks are used. Note that the '
        'TensorBoard callback profiles the 2nd step (unless otherwise '
        'specified). Please make sure the steps profiled by the two callbacks '
        'do not overlap.')
  return ProfilerCallback(model_dir, start_step, stop_step, steps_per_epoch)


class SimpleCheckpoint(tf.keras.callbacks.Callback):
  """Keras callback to save tf.train.Checkpoints."""

  def __init__(self, checkpoint_manager):
    super(SimpleCheckpoint, self).__init__()
    self.checkpoint_manager = checkpoint_manager

  def on_epoch_end(self, epoch, logs=None):
    step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
    self.checkpoint_manager.save(checkpoint_number=step_counter)


class ProfilerCallback(tf.keras.callbacks.Callback):
  """Save profiles in specified step range to log directory."""

  def __init__(self, log_dir, start_step, stop_step, steps_per_epoch):
    super(ProfilerCallback, self).__init__()
    self.log_dir = log_dir
    self.start_step = start_step
    self.stop_step = stop_step
    self.start_epoch = start_step // steps_per_epoch
    self.stop_epoch = stop_step // steps_per_epoch
    self.start_step_in_epoch = start_step % steps_per_epoch
    self.stop_step_in_epoch = stop_step % steps_per_epoch
    self.should_start = False
    self.should_stop = False

  def on_epoch_begin(self, epoch, logs=None):
    if epoch == self.start_epoch:
      self.should_start = True
    if epoch == self.stop_epoch:
      self.should_stop = True

  def on_batch_begin(self, batch, logs=None):
    if batch == self.start_step_in_epoch and self.should_start:
      self.should_start = False
      profiler.start(self.log_dir)
      logging.info('Profiler started at Step %s', self.start_step)

  def on_batch_end(self, batch, logs=None):
    if batch == self.stop_step_in_epoch and self.should_stop:
      self.should_stop = False
      profiler.stop()
      logging.info('Profiler saved profiles for steps between %s and %s to %s',
                   self.start_step, self.stop_step, self.log_dir)


def set_session_config(enable_eager=False,
                       enable_xla=False):
  """Sets the session config."""
  if is_v2_0():
    set_config_v2(enable_xla=enable_xla)
  else:
    config = get_config_proto_v1(enable_xla=enable_xla)
    if enable_eager:
      tf.compat.v1.enable_eager_execution(config=config)
    else:
      sess = tf.compat.v1.Session(config=config)
      tf.compat.v1.keras.backend.set_session(sess)


def get_config_proto_v1(enable_xla=False):
  """Return config proto according to flag settings, or None to use default."""
  config = None
  if enable_xla:
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_2)
  return config


def set_config_v2(enable_xla=False):
  """Config eager context according to flag values using TF 2.0 API."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)


def is_v2_0():
  """Returns true if using tf 2.0."""
  return tf2.enabled()


def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                  datasets_num_private_threads,
                                  num_gpus, per_gpu_thread_count):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  logging.info('Logical CPU cores: %s', cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = per_gpu_thread_count or 2
  os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  logging.info('TF_GPU_THREAD_COUNT: %s',
               os.environ['TF_GPU_THREAD_COUNT'])
  logging.info('TF_GPU_THREAD_MODE: %s',
               os.environ['TF_GPU_THREAD_MODE'])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * num_gpus
  num_runtime_threads = num_gpus
  if not datasets_num_private_threads:
    datasets_num_private_threads = min(
        cpu_count - total_gpu_thread_count - num_runtime_threads,
        num_gpus * 8)
    logging.info('Set datasets_num_private_threads to %s',
                 datasets_num_private_threads)
