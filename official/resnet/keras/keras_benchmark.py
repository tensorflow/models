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
"""Executes Keras benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

FLAGS = flags.FLAGS


class KerasBenchmark(tf.test.Benchmark):
  """Base benchmark class with methods to simplify testing."""
  local_flags = None

  def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
    self.output_dir = output_dir
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if KerasBenchmark.local_flags is None:
      for flag_method in self.flag_methods:
        flag_method()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      # Overrides flag values with defaults for the class of tests.
      for k, v in self.default_flags.items():
        setattr(FLAGS, k, v)
      saved_flag_values = flagsaver.save_flag_values()
      KerasBenchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(KerasBenchmark.local_flags)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        top_1_max=None,
                        top_1_min=None,
                        log_steps=None,
                        total_batch_size=None,
                        warmup=1):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      top_1_max: highest passing level for top_1 accuracy.
      top_1_min: lowest passing level for top_1 accuracy.
      log_steps: How often the log was created for stats['step_timestamp_log'].
      total_batch_size: Global batch-size.
      warmup: number of entries in stats['step_timestamp_log'] to ignore.
    """

    extras = {}
    if 'accuracy_top_1' in stats:
      extras['accuracy_top_1'] = self._json_description(
          stats['accuracy_top_1'],
          priority=0,
          min_value=top_1_min,
          max_value=top_1_max)
      extras['top_1_train_accuracy'] = self._json_description(
          stats['training_accuracy_top_1'], priority=1)

    if (warmup and 'step_timestamp_log' in stats and
        len(stats['step_timestamp_log']) > warmup):
      # first entry in the time_log is start of step 1. The rest of the
      # entries are the end of each step recorded
      time_log = stats['step_timestamp_log']
      elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
      num_examples = (
          total_batch_size * log_steps * (len(time_log) - warmup - 1))
      examples_per_sec = num_examples / elapsed
      extras['exp_per_second'] = self._json_description(
          examples_per_sec, priority=2)

    if 'avg_exp_per_second' in stats:
      extras['avg_exp_per_second'] = self._json_description(
          stats['avg_exp_per_second'], priority=3)

    self.report_benchmark(iters=-1, wall_time=wall_time_sec, extras=extras)

  def _json_description(self,
                        value,
                        priority=None,
                        min_value=None,
                        max_value=None):
    """Get a json-formatted string describing the attributes for a metric"""

    attributes = {}
    attributes['value'] = value
    if priority:
      attributes['priority'] = priority
    if min_value:
      attributes['min_value'] = min_value
    if max_value:
      attributes['max_value'] = max_value

    if min_value or max_value:
      succeeded = True
      if min_value and value < min_value:
        succeeded = False
      if max_value and value > max_value:
        succeeded = False
      attributes['succeeded'] = succeeded

    return json.dumps(attributes)
