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
from official.recommendation import ncf_keras_main
from official.utils.flags import core

FLAGS = flags.FLAGS
NCF_DATA_DIR_NAME = 'movielens_data'


class NCFKerasBenchmarkBase(tf.test.Benchmark):
  """Base class for NCF model benchmark."""
  local_flags = None

  def __init__(self,
               output_dir=None,
               default_flags=None,
               **kwargs):
    self.output_dir = output_dir
    self.default_flags = default_flags or {}

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if NCFKerasBenchmarkBase.local_flags is None:
      ncf_common.define_ncf_flags()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      core.set_defaults(**self.default_flags)
      saved_flag_values = flagsaver.save_flag_values()
      NCFKerasBenchmarkBase.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(NCFKerasBenchmarkBase.local_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = ncf_keras_main.run_ncf(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    metrics = self._extract_benchmark_report_extras(stats)
    self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics)

  def _extract_benchmark_report_extras(self, stats):
    raise NotImplementedError('Not implemented')


class NCFKerasAccuracy(NCFKerasBenchmarkBase):
  """Benchmark NCF model using real data."""

  def __init__(self,
               output_dir=None,
               root_data_dir=None,
               default_flags=None,
               **kwargs):

    default_flags = {}
    default_flags['dataset'] = 'ml-20m'
    default_flags['num_gpus'] = 1
    default_flags['train_epochs'] = 10
    default_flags['clean'] = True
    default_flags['batch_size'] = 99000
    default_flags['learning_rate'] = 0.00382059
    default_flags['beta1'] = 0.783529
    default_flags['beta2'] = 0.909003
    default_flags['epsilon'] = 1.45439e-07
    default_flags['layers'] = [256, 256, 128, 64]
    default_flags['num_factors'] = 64
    default_flags['hr_threshold'] = 0.635
    default_flags['ml_perf'] = True
    default_flags['use_synthetic_data'] = False
    default_flags['data_dir'] = os.path.join(root_data_dir, NCF_DATA_DIR_NAME)

    super(NCFKerasAccuracy, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        **kwargs)

  def _extract_benchmark_report_extras(self, stats):
    metrics = []
    metrics.append({'name': 'exp_per_second',
                    'value': stats['avg_exp_per_second']})

    # Target is 0.635, but some runs are below that level. Until we have
    # multi-run tests, we have to accept a lower target.
    metrics.append({'name': 'hr_at_10',
                    'value': stats['eval_hit_rate'],
                    'min_value': 0.630,
                    'max_value': 0.640})

    metrics.append({'name': 'train_loss',
                    'value': stats['loss']})

    return metrics

  def benchmark_1_gpu_early_stop(self):
    self._setup()
    FLAGS.early_stopping = True
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_early_stop(self):
    self._setup()
    FLAGS.distribution_strategy = 'off'
    FLAGS.early_stopping = True
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_early_stop(self):
    self._setup()
    FLAGS.distribution_strategy = 'off'
    FLAGS.early_stopping = True
    FLAGS.run_eagerly = True
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_early_stop(self):
    self._setup()
    FLAGS.early_stopping = True
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def benchmark_1_gpu_ctl_early_stop(self):
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.early_stopping = True
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_ctl_early_stop(self):
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.early_stopping = True
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def benchmark_2_gpus_early_stop(self):
    self._setup()
    FLAGS.early_stopping = True
    FLAGS.num_gpus = 2
    self._run_and_report_benchmark()

  def benchmark_2_gpus_ctl_early_stop(self):
    """NCF with custom training loop. Works only in TF 2.0."""
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.early_stopping = True
    FLAGS.num_gpus = 2
    self._run_and_report_benchmark()

#############################################
# Tests below with mlperf in the test name are of two types
#  1) 1 GPU tests are based on MLPerf 0.5 and the TensorFlow pulled submission.
#  2) 8 GPU tests are based on MLPerf 0.5 and use NVIDIA's hyper parameters.
#
# The purpose of both is to get a number to compare to existing results. To do
# this the number of epochs is held constant rather than a race to a given
# accuracy. The accuracy validation is done by the "early_stop" tests.
#############################################

  def benchmark_1_gpu_mlperf_like(self):
    """1 GPU using keras fit/compile."""
    self._setup()
    FLAGS.train_epochs = 7
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_mlperf_like(self):
    """1 GPU using compile/fit without dist_strat."""
    self._setup()
    FLAGS.train_epochs = 7
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_mlperf_like(self):
    """1 GPU using compile/fit without dist_strat and force run eager."""
    self._setup()
    FLAGS.train_epochs = 7
    FLAGS.distribution_strategy = 'off'
    FLAGS.run_eagerly = True
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_mlperf_like(self):
    """1 GPU using compile/fit with XLA."""
    self._setup()
    FLAGS.train_epochs = 7
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def benchmark_1_gpu_ctl_mlperf_like(self):
    """1 GPU using CTL."""
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.train_epochs = 7
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_ctl_mlperf_like(self):
    """1 GPU using CTL with XLA."""
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.enable_xla = True
    FLAGS.train_epochs = 7
    self._run_and_report_benchmark()

  def benchmark_8_gpu_mlperf_like(self):
    """8 GPU using keras fit/compile."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.train_epochs = 17
    FLAGS.batch_size = 1048576
    FLAGS.learning_rate = 0.0045
    FLAGS.beta1 = 0.25
    FLAGS.beta2 = 0.5
    FLAGS.epsilon = 1e-8
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_mlperf_like(self):
    """8 GPU using keras fit/compile with XLA."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.enable_xla = True
    FLAGS.train_epochs = 17
    FLAGS.batch_size = 1048576
    FLAGS.learning_rate = 0.0045
    FLAGS.beta1 = 0.25
    FLAGS.beta2 = 0.5
    FLAGS.epsilon = 1e-8
    self._run_and_report_benchmark()

  def benchmark_8_gpu_ctl_mlperf_like(self):
    """8 GPU using CTL."""
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.num_gpus = 8
    FLAGS.train_epochs = 17
    FLAGS.batch_size = 1048576
    FLAGS.learning_rate = 0.0045
    FLAGS.beta1 = 0.25
    FLAGS.beta2 = 0.5
    FLAGS.epsilon = 1e-8
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_ctl_mlperf_like(self):
    """8 GPU using CTL with XLA."""
    self._setup()
    FLAGS.keras_use_ctl = True
    FLAGS.enable_xla = True
    FLAGS.num_gpus = 8
    FLAGS.train_epochs = 17
    FLAGS.batch_size = 1048576
    FLAGS.learning_rate = 0.0045
    FLAGS.beta1 = 0.25
    FLAGS.beta2 = 0.5
    FLAGS.epsilon = 1e-8
    self._run_and_report_benchmark()


class NCFKerasSynth(NCFKerasBenchmarkBase):
  """Benchmark NCF model using synthetic data."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               **kwargs):

    default_flags = {}
    default_flags['dataset'] = 'ml-20m'
    default_flags['num_gpus'] = 1
    default_flags['train_epochs'] = 8
    default_flags['batch_size'] = 99000
    default_flags['learning_rate'] = 0.00382059
    default_flags['beta1'] = 0.783529
    default_flags['beta2'] = 0.909003
    default_flags['epsilon'] = 1.45439e-07
    default_flags['layers'] = [256, 256, 128, 64]
    default_flags['num_factors'] = 64
    default_flags['hr_threshold'] = 0.635
    default_flags['use_synthetic_data'] = True

    super(NCFKerasSynth, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        **kwargs)

  def _extract_benchmark_report_extras(self, stats):
    metrics = []
    metrics.append({'name': 'exp_per_second',
                    'value': stats['avg_exp_per_second']})
    return metrics

  def benchmark_1_gpu(self):
    self._setup()
    self._run_and_report_benchmark()

  def benchmark_2_gpus(self):
    self._setup()
    FLAGS.num_gpus = 2
    self._run_and_report_benchmark()
