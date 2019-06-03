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
"""Executes CTL benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

FLAGS = flags.FLAGS


class CtlBenchmark(tf.test.Benchmark):
  """Base benchmark class with methods to simplify testing."""
  local_flags = None

  def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
    self.output_dir = output_dir
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}

    if not output_dir:
      self.output_dir = '/tmp/'

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if CtlBenchmark.local_flags is None:
      for flag_method in self.flag_methods:
        flag_method()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      # Overrides flag values with defaults for the class of tests.
      for k, v in self.default_flags.items():
        setattr(FLAGS, k, v)
      saved_flag_values = flagsaver.save_flag_values()
      CtlBenchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(CtlBenchmark.local_flags)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        top_1_max=None,
                        top_1_min=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      top_1_max: highest passing level for top_1 accuracy.
      top_1_min: lowest passing level for top_1 accuracy.
    """

    metrics = []
    if 'accuracy_top_1' in stats:
      metrics.append({'name': 'accuracy_top_1',
                      'value': stats['accuracy_top_1'],
                      'min_value': top_1_min,
                      'max_value': top_1_max})
      metrics.append({'name': 'top_1_train_accuracy',
                      'value': stats['train_acc']})

    if 'exp_per_second' in stats:
      metrics.append({'name': 'exp_per_second',
                      'value': stats['exp_per_second']})

    self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics)
