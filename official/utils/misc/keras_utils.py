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

import time

import tensorflow as tf
from tensorflow.python.eager import profiler


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

  def __init__(self, batch_size, log_steps):
    """Callback for logging performance (# examples/second).

    Args:
      batch_size: Total batch size.
      log_steps: Interval of time history logs.

    """
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps

    # Logs start of step 0 then end of each step based on log_steps interval.
    self.timestamp_log = []

  def on_train_begin(self, logs=None):
    self.record_batch = True

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      timestamp = time.time()
      self.start_time = timestamp
      self.record_batch = False
      if batch == 0:
        self.timestamp_log.append(BatchTimestamp(batch, timestamp))

  def on_batch_end(self, batch, logs=None):
    if batch % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      if batch != 0:
        self.record_batch = True
        self.timestamp_log.append(BatchTimestamp(batch, timestamp))
        tf.compat.v1.logging.info(
            "BenchmarkMetric: {'num_batches':%d, 'time_taken': %f,"
            "'examples_per_second': %f}" %
            (batch, elapsed_time, examples_per_second))


def get_profiler_callback(model_dir, profile_steps, enable_tensorboard):
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
    tf.compat.v1.logging.warn(
        'Both TensorBoard and profiler callbacks are used. Note that the '
        'TensorBoard callback profiles the 2nd step (unless otherwise '
        'specified). Please make sure the steps profiled by the two callbacks '
        'do not overlap.')

  return ProfilerCallback(model_dir, start_step, stop_step)


class ProfilerCallback(tf.keras.callbacks.Callback):
  """Save profiles in specified step range to log directory."""

  def __init__(self, log_dir, start_step, stop_step):
    super(ProfilerCallback, self).__init__()
    self.log_dir = log_dir
    self.start_step = start_step
    self.stop_step = stop_step

  def on_batch_begin(self, batch, logs=None):
    if batch == self.start_step:
      profiler.start()
      tf.compat.v1.logging.info('Profiler started at Step %s', self.start_step)

  def on_batch_end(self, batch, logs=None):
    if batch == self.stop_step:
      results = profiler.stop()
      profiler.save(self.log_dir, results)
      tf.compat.v1.logging.info(
          'Profiler saved profiles for steps between %s and %s to %s',
          self.start_step, self.stop_step, self.log_dir)
