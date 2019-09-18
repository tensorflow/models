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
"""Utility functions or classes shared between BERT benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

# pylint: disable=g-bad-import-order
import numpy as np
from absl import flags
from absl.testing import flagsaver
import tensorflow.compat.v2 as tf
# pylint: enable=g-bad-import-order

from official.utils.flags import core as flags_core

FLAGS = flags.FLAGS


class BenchmarkTimerCallback(tf.keras.callbacks.Callback):
  """Callback that records time it takes to run each batch."""

  def __init__(self, num_batches_to_skip=10):
    super(BenchmarkTimerCallback, self).__init__()
    self.num_batches_to_skip = num_batches_to_skip
    self.timer_records = []
    self.start_time = None

  def on_batch_begin(self, batch, logs=None):
    if batch < self.num_batches_to_skip:
      return
    self.start_time = time.time()

  def on_batch_end(self, batch, logs=None):
    if batch < self.num_batches_to_skip:
      return

    assert self.start_time
    self.timer_records.append(time.time() - self.start_time)

  def get_examples_per_sec(self, batch_size):
    return batch_size / np.mean(self.timer_records)


class BertBenchmarkBase(tf.test.Benchmark):
  """Base class to hold methods common to test classes."""
  local_flags = None

  def __init__(self, output_dir=None):
    self.num_gpus = 8

    if not output_dir:
      output_dir = '/tmp'
    self.output_dir = output_dir
    self.timer_callback = None

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    self.timer_callback = BenchmarkTimerCallback()

    if BertBenchmarkBase.local_flags is None:
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      BertBenchmarkBase.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(BertBenchmarkBase.local_flags)

  def _report_benchmark(self, stats, wall_time_sec, min_accuracy, max_accuracy):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from BERT models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      min_accuracy: Minimum classification accuracy constraint to verify
        correctness of the model.
      max_accuracy: Maximum classification accuracy constraint to verify
        correctness of the model.
    """
    metrics = [{
        'name': 'training_loss',
        'value': stats['train_loss'],
    }, {
        'name':
            'exp_per_second',
        'value':
            self.timer_callback.get_examples_per_sec(FLAGS.train_batch_size)
    }]

    if 'eval_metrics' in stats:
      metrics.append({
          'name': 'eval_accuracy',
          'value': stats['eval_metrics'],
          'min_value': min_accuracy,
          'max_value': max_accuracy,
      })
    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=stats['total_training_steps'],
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_str})
