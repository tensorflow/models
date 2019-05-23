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
"""Executes Transformer w/Estimator benchmark and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.transformer.v2 import misc
from official.transformer.v2 import transformer_main as transformer_main

TRANSFORMER_EN2DE_DATA_DIR_NAME = 'wmt32k-en2de-official'
EN2DE_2014_BLEU_DATA_DIR_NAME = 'newstest2014'
FLAGS = flags.FLAGS


class EstimatorBenchmark(tf.test.Benchmark):
  """Methods common to executing transformer w/Estimator tests.

     Code under test for the Transformer Estimator models report the same data
     and require the same FLAG setup.
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
                        bleu_max=None,
                        bleu_min=None,
                        log_steps=None,
                        total_batch_size=None,
                        warmup=1):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      bleu_max: highest passing level for bleu score.
      bleu_min: lowest passing level for bleu score.
      log_steps: How often the log was created for stats['step_timestamp_log'].
      total_batch_size: Global batch-size.
      warmup: number of entries in stats['step_timestamp_log'] to ignore.
    """
    metrics = []
    if 'bleu_uncased' in stats:
      metrics.append({'name': 'bleu_uncased',
                      'value': stats['bleu_uncased'],
                      'min_value': bleu_min,
                      'max_value': bleu_max})

    if (warmup and 'step_timestamp_log' in stats and
        len(stats['step_timestamp_log']) > warmup):
      # first entry in the time_log is start of step 1. The rest of the
      # entries are the end of each step recorded
      time_log = stats['step_timestamp_log']
      elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
      num_examples = (
          total_batch_size * log_steps * (len(time_log) - warmup - 1))
      examples_per_sec = num_examples / elapsed
      metrics.append({'name': 'exp_per_second',
                      'value': examples_per_sec})

    if 'avg_exp_per_second' in stats:
      metrics.append({'name': 'avg_exp_per_second',
                      'value': stats['avg_exp_per_second']})

    self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics)


class TransformerKerasBenchmark(EstimatorBenchmark):
  """Benchmarks for Transformer (Base and Big) using Keras."""

  def __init__(self, output_dir=None, default_flags=None, batch_per_gpu=4096):
    """Initialize.

    Args:
      output_dir: Based directory for saving artifacts, e.g. checkpoints.
      default_flags: default flags to use for all tests.
      batch_per_gpu: batch size to use per gpu.
    """

    flag_methods = [misc.define_transformer_flags]
    self.batch_per_gpu = batch_per_gpu

    super(TransformerKerasBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods)

  def benchmark_graph_1_gpu(self):
    """Benchmark graph 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = transformer_main.main(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec
    self._report_benchmark(stats,
                           wall_time_sec,
                           total_batch_size=FLAGS.batch_size,
                           log_steps=FLAGS.log_steps)


class TransformerBaseKerasBenchmarkReal(TransformerKerasBenchmark):
  """Transformer based version real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    train_data_dir = os.path.join(root_data_dir,
                                  TRANSFORMER_EN2DE_DATA_DIR_NAME)
    vocab_file = os.path.join(root_data_dir,
                              TRANSFORMER_EN2DE_DATA_DIR_NAME,
                              'vocab.ende.32768')

    def_flags = {}
    def_flags['param_set'] = 'base'
    def_flags['vocab_file'] = vocab_file
    def_flags['data_dir'] = train_data_dir
    def_flags['train_steps'] = 200
    def_flags['train_epochs'] = 1
    def_flags['epochs_between_evals'] = 10
    def_flags['log_steps'] = 10

    super(TransformerBaseKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class TransformerBigKerasBenchmarkReal(TransformerKerasBenchmark):
  """Transformer based version real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    train_data_dir = os.path.join(root_data_dir,
                                  TRANSFORMER_EN2DE_DATA_DIR_NAME)
    vocab_file = os.path.join(root_data_dir,
                              TRANSFORMER_EN2DE_DATA_DIR_NAME,
                              'vocab.ende.32768')

    def_flags = {}
    def_flags['param_set'] = 'big'
    def_flags['vocab_file'] = vocab_file
    def_flags['data_dir'] = train_data_dir
    def_flags['train_steps'] = 200
    def_flags['train_epochs'] = 1
    def_flags['epochs_between_evals'] = 10
    def_flags['log_steps'] = 10

    super(TransformerBigKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags, batch_per_gpu=3072)
