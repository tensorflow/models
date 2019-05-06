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

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.recommendation import ncf_common
from official.utils.flags import core

FLAGS = flags.FLAGS
NCF_DATA_DIR_NAME = 'movielens_data'


class KerasNCFBenchmarkBase(tf.test.Benchmark):
  """Base class for NCF model benchmark."""
  local_flags = None

  def __init__(self,
               run_func,
               output_dir=None,
               default_flags=None,
               **kwargs):
    self.run_func = run_func
    self.output_dir = output_dir
    self.default_flags = default_flags or {}

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if KerasNCFBenchmarkBase.local_flags is None:
      ncf_common.define_ncf_flags()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      core.set_defaults(**self.default_flags)
      saved_flag_values = flagsaver.save_flag_values()
      KerasNCFBenchmarkBase.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(KerasNCFBenchmarkBase.local_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = self.run_func.run_ncf(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    metrics = self._extract_benchmark_report_extras(stats)
    self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics)

  def _extract_benchmark_report_extras(self, stats):
    raise NotImplementedError('Not implemented')
