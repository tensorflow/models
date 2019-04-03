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

from absl import flags
import tensorflow as tf


class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp


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
