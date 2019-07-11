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
from tensorflow.core.protobuf import rewriter_config_pb2
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
    self.global_steps = 0

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_batch_begin(self, batch, logs=None):
    self.global_steps += 1
    if self.global_steps == 1:
      self.start_time = time.time()
      self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                               self.start_time))

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      self.timestamp_log.append(BatchTimestamp(self.global_steps, timestamp))
      tf.compat.v1.logging.info(
          "BenchmarkMetric: {'global step':%d, 'time_taken': %f,"
          "'examples_per_second': %f}" %
          (self.global_steps, elapsed_time, examples_per_second))
      self.start_time = timestamp


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


def set_session_config(enable_eager=False,
                       enable_xla=False,
                       enable_grappler_layout_optimizer=True):
  """Sets the session config."""
  if is_v2_0():
    set_config_v2(
        enable_xla=enable_xla,
        enable_grappler_layout_optimizer=enable_grappler_layout_optimizer)
  else:
    config = get_config_proto_v1(
        enable_xla=enable_xla,
        enable_grappler_layout_optimizer=enable_grappler_layout_optimizer)
    if enable_eager:
      tf.compat.v1.enable_eager_execution(config=config)
    else:
      sess = tf.Session(config=config)
      tf.keras.backend.set_session(sess)


def get_config_proto_v1(enable_xla=False,
                        enable_grappler_layout_optimizer=True):
  """Return config proto according to flag settings, or None to use default."""
  config = None
  if enable_xla:
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_2)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it causes
    # OOM and performance regression.
    config.graph_options.rewrite_options.pin_to_host_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
  # TODO(b/76028325): Remove when generic layout optimizer will be ready.
  if not enable_grappler_layout_optimizer:
    if config is None:
      config = tf.compat.v1.ConfigProto()
    # Disable LayoutOptimizer in grappler, because it might de-optimize fp16
    # graphs, and force NCHW data format in all convolutions and batch
    # normalizations.
    config.graph_options.rewrite_options.layout_optimizer = (
        rewriter_config_pb2.RewriterConfig.OFF)
  return config


def set_config_v2(enable_xla=False,
                  enable_grappler_layout_optimizer=False):
  """Config eager context according to flag values using TF 2.0 API."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it
    # causes OOM and performance regression.
    tf.config.optimizer.set_experimental_options(
        {'pin_to_host_optimization': False}
    )
  # TODO(b/76028325): Remove when generic layout optimizer will be ready.
  if not enable_grappler_layout_optimizer:
    # Disable LayoutOptimizer in grappler, because it might de-optimize fp16
    # graphs, and force NCHW data format in all convolutions and batch
    # normalizations.
    tf.config.optimizer.set_experimental_options(
        {'layout_optimizer': False}
    )

def is_v2_0():
  """Returns true if using tf 2.0."""
  if hasattr(tf, 'contrib'):
    return False
  else:
    return True
