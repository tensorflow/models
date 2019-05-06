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
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.recommendation import benchmark_base
from official.recommendation import ncf_estimator_main

FLAGS = flags.FLAGS


class KerasNCFRealData(benchmark_base.KerasNCFBenchmarkBase):
  """Benchmark NCF model using real data."""

  def __init__(self,
               output_dir=None,
               root_data_dir=None,
               default_flags=None,
               **kwargs):

    default_flags = {}
    default_flags['dataset'] = 'ml-20m'
    default_flags['num_gpus'] = 1
    default_flags['train_epochs'] = 14
    default_flags['clean'] = True
    default_flags['batch_size'] = 160000
    default_flags['learning_rate'] = 0.00382059
    default_flags['beta1'] = 0.783529
    default_flags['beta2'] = 0.909003
    default_flags['epsilon'] = 1.45439e-07
    default_flags['layers'] = [256, 256, 128, 64]
    default_flags['num_factors'] = 64
    default_flags['hr_threshold'] = 0.635
    default_flags['ml_perf'] = True
    default_flags['use_synthetic_data'] = False
    default_flags['data_dir'] = os.path.join(
        root_data_dir,
        benchmark_base.NCF_DATA_DIR_NAME)

    super(KerasNCFRealData, self).__init__(
        ncf_estimator_main.run_ncf,
        output_dir=output_dir,
        default_flags=default_flags,
        **kwargs)

  def _extract_benchmark_report_extras(self, stats):
    metrics = []
    metrics.append({'name': 'exp_per_second',
                    'value': stats['avg_exp_per_second']})

    # Target is 0.625, but some runs are below that level. Until we have
    # multi-run tests, we have to accept a lower target.
    metrics.append({'name': 'hr_at_10',
                    'value': stats['eval_hit_rate'],
                    'min_value': 0.618,
                    'max_value': 0.635})

    metrics.append({'name': 'train_loss',
                    'value': stats['loss']})

    return metrics

  def benchmark_1_gpu(self):
    self._setup()
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_cloning(self):
    self._setup()
    FLAGS.clone_model_in_keras_dist_strat = False
    self._run_and_report_benchmark()

  def benchmark_2_gpus(self):
    self._setup()
    FLAGS.num_gpus = 2
    self._run_and_report_benchmark()

  def benchmark_2_gpus_no_cloning(self):
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.clone_model_in_keras_dist_strat = False
    self._run_and_report_benchmark()


class KerasNCFSyntheticData(benchmark_base.KerasNCFBenchmarkBase):
  """Benchmark NCF model using synthetic data."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               **kwargs):

    default_flags = {}
    default_flags['dataset'] = 'ml-20m'
    default_flags['num_gpus'] = 1
    default_flags['train_epochs'] = 14
    default_flags['batch_size'] = 160000
    default_flags['learning_rate'] = 0.00382059
    default_flags['beta1'] = 0.783529
    default_flags['beta2'] = 0.909003
    default_flags['epsilon'] = 1.45439e-07
    default_flags['layers'] = [256, 256, 128, 64]
    default_flags['num_factors'] = 64
    default_flags['hr_threshold'] = 0.635
    default_flags['use_synthetic_data'] = True

    super(KerasNCFSyntheticData, self).__init__(
        ncf_estimator_main.run_ncf,
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

  def benchmark_1_gpu_no_cloning(self):
    self._setup()
    FLAGS.clone_model_in_keras_dist_strat = False
    self._run_and_report_benchmark()

  def benchmark_2_gpus(self):
    self._setup()
    FLAGS.num_gpus = 2
    self._run_and_report_benchmark()

  def benchmark_2_gpus_no_cloning(self):
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.clone_model_in_keras_dist_strat = False
    self._run_and_report_benchmark()


if __name__ == '__main__':
  tf.test.main()
