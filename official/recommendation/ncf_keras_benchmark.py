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

from official.recommendation import ncf_common
from official.recommendation import ncf_keras_main
from official.utils.flags import core

FLAGS = flags.FLAGS

class KerasNCFBenchmarkBase(tf.test.Benchmark):
  """Base class for NCF model benchmark."""
  local_flags = None

  def __init__(self,
               output_dir=None,
               root_data_dir=None,
               default_flags=None,
               **kwargs):

    self.output_dir = output_dir
    self.default_flags = default_flags or {}
    ncf_common.define_ncf_flags()

    if root_data_dir:
      FLAGS.data_dir = os.path.join(root_data_dir, 'movielens_data')

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if KerasNCFBenchmarkBase.local_flags is None:
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      core.set_defaults(**self.default_flags)
      saved_flag_values = flagsaver.save_flag_values()
      KerasNCFBenchmarkBase.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(KerasNCFBenchmarkBase.local_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = ncf_keras_main.run_ncf(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    extras = self._extract_benchmark_report_extras(stats)
    self.report_benchmark(iters=-1, wall_time=wall_time_sec, extras=extras)

  def _extract_benchmark_report_extras(self, stats):
    raise NotImplementedError("Not implemented")


class KerasNCFRealData(KerasNCFBenchmarkBase):
  """Benchmark NCF model using real data."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               **kwargs):

    default_flags = {}
    default_flags['dataset'] = 'ml-20m'
    default_flags['num_gpus'] = 1
    default_flags['train_epochs'] = 14
    default_flags['clean'] = True
    default_flags['batch_size'] = 16000
    default_flags['learning_rate'] = 0.00382059
    default_flags['beta1'] = 0.783529
    default_flags['beta2'] = 0.909003
    default_flags['epsilon'] = 1.45439e-07
    default_flags['layers'] = [256, 256, 128, 64]
    default_flags['num_factors'] = 64
    default_flags['hr_threshold'] = 0.635
    default_flags['ml_perf'] = True
    default_flags['use_synthetic_data'] = False

    super(KerasNCFRealData, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        **kwargs)

  def _extract_benchmark_report_extras(self, stats):
    extras = {}
    extras['train_loss'] = stats['loss']
    extras['eval_hit_rate'] = stats['eval_hit_rate']
    extras['examples_per_second'] = stats['avg_exp_per_second']
    return extras

  def benchmark_1_gpu(self):
    self._setup()
    self._run_and_report_benchmark()


class KerasNCFSyntheticData(KerasNCFBenchmarkBase):
  """Benchmark NCF model using synthetic data."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               **kwargs):

    default_flags = {}
    default_flags['dataset'] = 'ml-20m'
    default_flags['num_gpus'] = 1
    default_flags['train_epochs'] = 14
    default_flags['batch_size'] = 16000
    default_flags['learning_rate'] = 0.00382059
    default_flags['beta1'] = 0.783529
    default_flags['beta2'] = 0.909003
    default_flags['epsilon'] = 1.45439e-07
    default_flags['layers'] = [256, 256, 128, 64]
    default_flags['num_factors'] = 64
    default_flags['hr_threshold'] = 0.635
    default_flags['use_synthetic_data'] = True

    super(KerasNCFSyntheticData, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        **kwargs)

  def _extract_benchmark_report_extras(self, stats):
    extras = {}
    extras['examples_per_second'] = stats['avg_exp_per_second']
    return extras

  def benchmark_1_gpu(self):
    self._setup()
    self._run_and_report_benchmark()
