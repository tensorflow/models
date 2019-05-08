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
"""Executes Estimator benchmarks and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.transformer import transformer_main as transformer_main
from official.utils.logs import hooks

TRANSFORMER_EN2DE_DATA_DIR_NAME = 'wmt32k-en2de-official'
EN2DE_2014_BLEU_DATA_DIR_NAME = 'newstest2014'
FLAGS = flags.FLAGS


class EstimatorBenchmark(tf.test.Benchmark):
  """Base class to hold methods common to test classes in the module.

     Code under test for Estimator models (ResNet50 and 56) report mostly the
     same data and require the same FLAG setup.
  """

  local_flags = None

  def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
    if not output_dir:
      output_dir = '/tmp'
    self.output_dir = output_dir
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    if EstimatorBenchmark.local_flags is None:
      for flag_method in self.flag_methods:
        flag_method()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      # Overrides flag values with defaults for the class of tests.
      for k, v in self.default_flags.items():
        setattr(FLAGS, k, v)
      saved_flag_values = flagsaver.save_flag_values()
      EstimatorBenchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(EstimatorBenchmark.local_flags)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        top_1_max=None,
                        top_1_min=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from estimator models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      top_1_max: highest passing level for top_1 accuracy.
      top_1_min: lowest passing level for top_1 accuracy.
    """

    examples_per_sec_hook = None
    for hook in stats['train_hooks']:
      if isinstance(hook, hooks.ExamplesPerSecondHook):
        examples_per_sec_hook = hook
        break

    eval_results = stats['eval_results']
    metrics = []
    if 'accuracy' in eval_results:
      metrics.append({'name': 'accuracy_top_1',
                      'value': eval_results['accuracy'].item(),
                      'min_value': top_1_min,
                      'max_value': top_1_max})
    if 'accuracy_top_5' in eval_results:
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


class TransformerBaseEstimatorAccuracy(EstimatorBenchmark):
  """Benchmark accuracy tests for ResNet50 w/ Estimator."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmark accuracy tests for ResNet50 w/ Estimator.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    flag_methods = [transformer_main.define_transformer_flags]

    self.train_data_dir = os.path.join(root_data_dir,
                                       TRANSFORMER_EN2DE_DATA_DIR_NAME)

    self.vocab_file = os.path.join(root_data_dir,
                                   TRANSFORMER_EN2DE_DATA_DIR_NAME,
                                   'vocab.ende.32768')

    self.bleu_source = os.path.join(root_data_dir,
                                    EN2DE_2014_BLEU_DATA_DIR_NAME,
                                    'newstest2014.en')

    self.bleu_ref = os.path.join(root_data_dir,
                                 EN2DE_2014_BLEU_DATA_DIR_NAME,
                                 'newstest2014.de')

    super(TransformerBaseEstimatorAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def benchmark_1_gpu(self):
    """Test FP16 graph rewrite 8 GPUs graph mode."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'base'
    FLAGS.batch_size = 2024
    # TODO(tobyboyd): Change to epochs not steps for accuracy test.
    FLAGS.train_steps = 600
    FLAGS.steps_between_evals = 600
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu')
    FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = transformer_main.run_transformer(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec
    self._report_benchmark(stats,
                           wall_time_sec,
                           top_1_min=0.762,
                           top_1_max=0.766)


class TransformerBaseEstimatorBenchmark(EstimatorBenchmark):
  """Benchmarks for ResNet50 using Estimator."""
  local_flags = None

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [transformer_main.define_transformer_flags]

    super(TransformerBaseEstimatorBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods)

  def benchmark_1_gpu(self):
    """Benchmarks graph fp16 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.param_set = 'base'
    FLAGS.batch_size = 2024
    # TODO(tobyboyd): Change to epochs not steps for accuracy test.
    FLAGS.train_steps = 600
    FLAGS.steps_between_evals = 600
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu')
    FLAGS.hooks = ['ExamplesPerSecondHook']
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = transformer_main.run_transformer(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec
    self._report_benchmark(stats,
                           wall_time_sec)


class TransformerBaseEstimatorBenchmarkSynth(TransformerBaseEstimatorBenchmark):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['use_synthetic_data'] = True
    def_flags['max_train_steps'] = 110
    def_flags['train_epochs'] = 1

    super(TransformerBaseEstimatorBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class TransformerBaseEstimatorBenchmarkReal(TransformerBaseEstimatorBenchmark):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    train_data_dir = os.path.join(root_data_dir,
                                  TRANSFORMER_EN2DE_DATA_DIR_NAME)
    vocab_file = os.path.join(root_data_dir,
                              TRANSFORMER_EN2DE_DATA_DIR_NAME,
                              'vocab.ende.32768')

    def_flags = {}
    def_flags['vocab_file'] = vocab_file
    def_flags['data_dir'] = train_data_dir

    super(TransformerBaseEstimatorBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)
