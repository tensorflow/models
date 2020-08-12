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

import time

# pylint: disable=g-bad-import-order

import numpy as np
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.benchmark.perfzero_benchmark import PerfZeroBenchmark

FLAGS = flags.FLAGS


class BenchmarkTimerCallback(tf.keras.callbacks.Callback):
  """Callback that records time it takes to run each batch."""

  def __init__(self, num_batches_to_skip=10):
    super(BenchmarkTimerCallback, self).__init__()
    self.batch_start_times = {}
    self.batch_stop_times = {}

  def on_batch_begin(self, batch, logs=None):
    self.batch_start_times[batch] = time.time()

  def on_batch_end(self, batch, logs=None):
    # If there are multiple steps_per_loop, the end batch index will not be the
    # same as the starting index. Use the last starting index instead.
    if batch not in self.batch_start_times:
      batch = max(self.batch_start_times.keys())

    self.batch_stop_times[batch] = time.time()

  def get_examples_per_sec(self, batch_size, num_batches_to_skip=1):
    batch_durations = []
    for batch in self.batch_start_times:
      if batch in self.batch_stop_times and batch >= num_batches_to_skip:
        batch_durations.append(self.batch_stop_times[batch] -
                               self.batch_start_times[batch])
    return batch_size / np.mean(batch_durations)

  def get_startup_time(self, program_start_time):
    return self.batch_start_times[0] - program_start_time


class BertBenchmarkBase(PerfZeroBenchmark):
  """Base class to hold methods common to test classes."""
  local_flags = None

  def __init__(self, output_dir=None, tpu=None, **kwargs):
    super(BertBenchmarkBase, self).__init__(
        output_dir=output_dir, tpu=tpu, **kwargs)
    self.num_gpus = 8
    self.timer_callback = None

  def _setup(self):
    """Sets up and resets flags before each test."""
    super(BertBenchmarkBase, self)._setup()
    self.timer_callback = BenchmarkTimerCallback()

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
    }]
    if self.timer_callback:
      metrics.append({
          'name':
              'exp_per_second',
          'value':
              self.timer_callback.get_examples_per_sec(FLAGS.train_batch_size *
                                                       FLAGS.steps_per_loop)
      })
    else:
      metrics.append({
          'name': 'exp_per_second',
          'value': 0.0,
      })
    if self.timer_callback and 'start_time_sec' in stats:
      metrics.append({
          'name': 'startup_time',
          'value': self.timer_callback.get_startup_time(stats['start_time_sec'])
      })

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
