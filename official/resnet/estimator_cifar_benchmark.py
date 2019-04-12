# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Executes Estimator benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main as cifar_main
from official.utils.logs import hooks


class EstimatorCifar10BenchmarkTests(tf.test.Benchmark):
  """Benchmarks and accuracy tests for Estimator ResNet56."""

  local_flags = None

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    self.output_dir = output_dir
    self.data_dir = os.path.join(root_data_dir, 'cifar-10-batches-bin')

  def resnet56_1_gpu(self):
    """Test layers model with Estimator and distribution strategies."""
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = self.data_dir
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_1_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    flags.FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def resnet56_fp16_1_gpu(self):
    """Test layers FP16 model with Estimator and distribution strategies."""
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = self.data_dir
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_fp16_1_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp16'
    flags.FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def resnet56_2_gpu(self):
    """Test layers model with Estimator and dist_strat. 2 GPUs."""
    self._setup()
    flags.FLAGS.num_gpus = 2
    flags.FLAGS.data_dir = self.data_dir
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_2_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    flags.FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def resnet56_fp16_2_gpu(self):
    """Test layers FP16 model with Estimator and dist_strat. 2 GPUs."""
    self._setup()
    flags.FLAGS.num_gpus = 2
    flags.FLAGS.data_dir = self.data_dir
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_fp16_2_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp16'
    flags.FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def unit_test(self):
    """A lightweight test that can finish quickly."""
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = self.data_dir
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 1
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_1_gpu')
    flags.FLAGS.resnet_size = 8
    flags.FLAGS.dtype = 'fp32'
    flags.FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    """Executes benchmark and reports result."""
    start_time_sec = time.time()
    stats = cifar_main.run_cifar(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    examples_per_sec_hook = None
    for hook in stats['train_hooks']:
      if isinstance(hook, hooks.ExamplesPerSecondHook):
        examples_per_sec_hook = hook
        break

    eval_results = stats['eval_results']
    metrics = []
    metrics.append({'name': 'accuracy_top_1',
                    'value': eval_results['accuracy'].item()})
    metrics.append({'name': 'accuracy_top_5',
                    'value': eval_results['accuracy_top_5'].item()})
    if examples_per_sec_hook:
      exp_per_second_list = examples_per_sec_hook.current_examples_per_sec_list
      # ExamplesPerSecondHook skips the first 10 steps.
      exp_per_sec = sum(exp_per_second_list) / (len(exp_per_second_list))
      metrics.append({'name': 'exp_per_second',
                      'value': exp_per_sec})

    self.report_benchmark(
        iters=eval_results['global_step'],
        wall_time=wall_time_sec,
        metrics=metrics)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if EstimatorCifar10BenchmarkTests.local_flags is None:
      cifar_main.define_cifar_flags()
      # Loads flags to get defaults to then override.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      EstimatorCifar10BenchmarkTests.local_flags = saved_flag_values
      return
    flagsaver.restore_flag_values(EstimatorCifar10BenchmarkTests.local_flags)
