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
"""Executes Transformer w/Keras benchmark and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags

from official.transformer.v2 import misc
from official.transformer.v2 import transformer_main as transformer_main
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark

TRANSFORMER_EN2DE_DATA_DIR_NAME = 'wmt32k-en2de-official'
EN2DE_2014_BLEU_DATA_DIR_NAME = 'newstest2014'
FLAGS = flags.FLAGS


class TransformerBenchmark(PerfZeroBenchmark):
  """Methods common to executing transformer w/keras tests.

     Code under test for the Transformer Keras models report the same data and
     require the same FLAG setup.
  """

  def __init__(self, output_dir=None, default_flags=None, root_data_dir=None,
               flag_methods=None):
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

    super(TransformerBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods)

  def _run_and_report_benchmark(self,
                                bleu_max=None,
                                bleu_min=None,
                                log_steps=None,
                                total_batch_size=None,
                                warmup=1):
    """Report benchmark results by writing to local protobuf file.

    Args:
      bleu_max: highest passing level for bleu score.
      bleu_min: lowest passing level for bleu score.
      log_steps: How often the log was created for stats['step_timestamp_log'].
      total_batch_size: Global batch-size.
      warmup: number of entries in stats['step_timestamp_log'] to ignore.
    """
    start_time_sec = time.time()
    task = transformer_main.TransformerTask(FLAGS)
    stats = task.train()
    wall_time_sec = time.time() - start_time_sec

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


class TransformerBaseKerasAccuracy(TransformerBenchmark):
  """Benchmark accuracy tests for Transformer Base model w/ Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmark accuracy tests for Transformer Base model w/ Keras.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    flag_methods = [misc.define_transformer_flags]

    super(TransformerBaseKerasAccuracy, self).__init__(
        output_dir=output_dir, root_data_dir=root_data_dir,
        flag_methods=flag_methods)

  def benchmark_1_gpu(self):
    """Benchmark 1 gpu.

      The paper uses 8 GPUs and a much larger effective batch size, this is will
      not converge to the 27.3 BLEU (uncased) SOTA.
    """
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'base'
    FLAGS.batch_size = 4096
    FLAGS.train_steps = 200000
    FLAGS.steps_between_evals = 5000
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    # These bleu scores are based on test runs after at this limited
    # number of steps and batch size after verifying SOTA at 8xV100s.
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=25.3,
                                   bleu_max=26)


class TransformerKerasBenchmark(TransformerBenchmark):
  """Benchmarks for Transformer (Base and Big) using Keras."""

  def __init__(self, output_dir=None, default_flags=None,
               root_data_dir=None, batch_per_gpu=4096):
    """Initialize.

    Args:
      output_dir: Based directory for saving artifacts, e.g. checkpoints.
      default_flags: default flags to use for all tests.
      root_data_dir: root directory for data, e.g. training.
      batch_per_gpu: batch size to use per gpu.
    """
    flag_methods = [misc.define_transformer_flags]
    self.batch_per_gpu = batch_per_gpu

    super(TransformerKerasBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        root_data_dir=root_data_dir,
        flag_methods=flag_methods)

  def benchmark_1_gpu(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
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
    def_flags['log_steps'] = 10

    super(TransformerBaseKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags,
        root_data_dir=root_data_dir, batch_per_gpu=4096)


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
    def_flags['log_steps'] = 10

    super(TransformerBigKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags,
        root_data_dir=root_data_dir, batch_per_gpu=3072)
