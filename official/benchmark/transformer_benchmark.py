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
import tensorflow as tf
from official.benchmark import benchmark_wrappers
from official.benchmark import owner_utils
from official.benchmark.perfzero_benchmark import PerfZeroBenchmark
from official.nlp.transformer import misc
from official.nlp.transformer import transformer_main as transformer_main
from official.utils.flags import core as flags_core

TRANSFORMER_EN2DE_DATA_DIR_NAME = 'wmt32k-en2de-official'
EN2DE_2014_BLEU_DATA_DIR_NAME = 'newstest2014'
FLAGS = flags.FLAGS
TMP_DIR = os.getenv('TMPDIR')


class TransformerBenchmark(PerfZeroBenchmark):
  """Methods common to executing transformer w/keras tests.

     Code under test for the Transformer Keras models report the same data and
     require the same FLAG setup.
  """

  def __init__(self, output_dir=None, default_flags=None, root_data_dir=None,
               flag_methods=None, tpu=None):
    root_data_dir = root_data_dir if root_data_dir else ''

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

    if default_flags is None:
      default_flags = {}
    default_flags['data_dir'] = self.train_data_dir
    default_flags['vocab_file'] = self.vocab_file

    super(TransformerBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods,
        tpu=tpu)

  @benchmark_wrappers.enable_runtime_flags
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
      if 'bleu_uncased_history' in stats:
        bleu_uncased_best = max(stats['bleu_uncased_history'],
                                key=lambda x: x[1])
        metrics.append({'name': 'bleu_uncased',
                        'value': bleu_uncased_best[1],
                        'min_value': bleu_min,
                        'max_value': bleu_max})
        metrics.append({'name': 'bleu_best_score_iteration',
                        'value': bleu_uncased_best[0]})
        metrics.append({'name': 'bleu_uncased_last',
                        'value': stats['bleu_uncased']})
      else:
        metrics.append({'name': 'bleu_uncased',
                        'value': stats['bleu_uncased'],
                        'min_value': bleu_min,
                        'max_value': bleu_max})

    if (warmup and 'step_timestamp_log' in stats and
        len(stats['step_timestamp_log']) > warmup + 1):
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

    if 'step_timestamp_log' in stats:
      time_log = stats['step_timestamp_log']
      metrics.append({'name': 'startup_time',
                      'value': time_log[0].timestamp - start_time_sec})

    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics,
                          extras={'flags': flags_str})


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
    FLAGS.batch_size = 2048
    FLAGS.train_steps = 1000
    FLAGS.steps_between_evals = 500
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    # These bleu scores are based on test runs after at this limited
    # number of steps and batch size after verifying SOTA at 8xV100s.
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=25.3,
                                   bleu_max=26)

  def benchmark_1_gpu_static_batch(self):
    """Benchmark 1 gpu with static_batch.

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
    FLAGS.train_steps = 100000
    FLAGS.steps_between_evals = 5000
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_static_batch')
    # These bleu scores are based on test runs after at this limited
    # number of steps and batch size after verifying SOTA at 8xV100s.
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=25.3,
                                   bleu_max=26)

  def benchmark_8_gpu(self):
    """Benchmark 8 gpu.

      Should converge to 27.3 BLEU (uncased). This has not been confirmed yet.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'base'
    FLAGS.batch_size = 4096*8
    FLAGS.train_steps = 100000
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=27,
                                   bleu_max=28)

  def benchmark_8_gpu_static_batch(self):
    """Benchmark 8 gpu.

      Should converge to 27.3 BLEU (uncased). This has not been confirmed yet.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'base'
    FLAGS.batch_size = 4096*8
    FLAGS.train_steps = 100000
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.steps_between_evals = 5000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_static_batch')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=27,
                                   bleu_max=28)


class TransformerBigKerasAccuracy(TransformerBenchmark):
  """Benchmark accuracy tests for Transformer Big model w/ Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmark accuracy tests for Transformer Big model w/ Keras.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    flag_methods = [misc.define_transformer_flags]

    super(TransformerBigKerasAccuracy, self).__init__(
        output_dir=output_dir, root_data_dir=root_data_dir,
        flag_methods=flag_methods)

  def benchmark_8_gpu(self):
    """Benchmark 8 gpu.

    Over 6 runs with eval every 20K steps the average highest value was 28.195
    (bleu uncased). 28.424 was the highest and 27.96 the lowest. The values are
    the highest value seen during a run and occurred at a median of iteration 9.
    Iterations are not epochs, an iteration is a number of steps between evals.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'big'
    FLAGS.batch_size = 3072*8
    FLAGS.train_steps = 20000 * 12
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=27.9,
                                   bleu_max=29.2)

  def benchmark_8_gpu_static_batch(self):
    """Benchmark 8 gpu.

    Should converge to 28.4 BLEU (uncased). This has not be verified yet."
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'big'
    FLAGS.batch_size = 3072*8
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.train_steps = 20000 * 12
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_static_batch')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=28,
                                   bleu_max=29.2)

  def benchmark_8_gpu_fp16(self):
    """Benchmark 8 gpu with dynamic batch and fp16.

    Over 6 runs with eval every 20K steps the average highest value was 28.247
    (bleu uncased). 28.424 was the highest and 28.09 the lowest. The values are
    the highest value seen during a run and occurred at a median of iteration
    11. While this could be interpreted as worse than FP32, if looking at the
    first iteration at which 28 is passed FP16 performs equal and possibly
    better. Although not part of the initial test runs, the highest value
    recorded with the arguments below was 28.9 at iteration 12. Iterations are
    not epochs, an iteration is a number of steps between evals.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'big'
    FLAGS.batch_size = 3072*8
    FLAGS.train_steps = 20000 * 12
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=28,
                                   bleu_max=29.2)

  def benchmark_8_gpu_fp16_amp(self):
    """Benchmark 8 gpu with dynamic batch and fp16 with automatic mixed precision.

      Should converge to 28.4 BLEU (uncased). This has not be verified yet."
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'big'
    FLAGS.batch_size = 3072*8
    FLAGS.train_steps = 20000 * 12
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16_amp')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=28,
                                   bleu_max=29)

  def benchmark_8_gpu_static_batch_fp16(self):
    """Benchmark 8 gpu with static batch and fp16.

      Should converge to 28.4 BLEU (uncased). This has not be verified yet."
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'big'
    FLAGS.batch_size = 3072*8
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.train_steps = 400000
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_static_batch_fp16')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=28,
                                   bleu_max=29.2)

  def benchmark_xla_8_gpu_static_batch_fp16(self):
    """Benchmark 8 gpu with static batch, XLA, and FP16.

      Should converge to 28.4 BLEU (uncased). This has not be verified yet."
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_xla = True
    FLAGS.data_dir = self.train_data_dir
    FLAGS.vocab_file = self.vocab_file
    # Sets values directly to avoid validation check.
    FLAGS['bleu_source'].value = self.bleu_source
    FLAGS['bleu_ref'].value = self.bleu_ref
    FLAGS.param_set = 'big'
    FLAGS.batch_size = 3072*8
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.train_steps = 400000
    FLAGS.steps_between_evals = 20000
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_static_batch_fp16')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps,
                                   bleu_min=28,
                                   bleu_max=29.2)


class TransformerKerasBenchmark(TransformerBenchmark):
  """Benchmarks for Transformer (Base and Big) using Keras."""

  def __init__(self, output_dir=None, default_flags=None,
               root_data_dir=None, batch_per_gpu=4096, tpu=None):
    """Initialize.

    Args:
      output_dir: Based directory for saving artifacts, e.g. checkpoints.
      default_flags: default flags to use for all tests.
      root_data_dir: root directory for data, e.g. training.
      batch_per_gpu: batch size to use per gpu.
      tpu: Target TPU to use.
    """
    flag_methods = [misc.define_transformer_flags]
    self.batch_per_gpu = batch_per_gpu

    super(TransformerKerasBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        root_data_dir=root_data_dir,
        flag_methods=flag_methods,
        tpu=tpu)

  def benchmark_1_gpu_no_dist_strat(self):
    """Benchmark 1 gpu without distribution strategy."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'off'
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_1_gpu_no_dist_strat_static_batch(self):
    """Benchmark 1 gpu without distribution strategy with static batch."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'off'
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_ds_sb')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_1_gpu(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_1_gpu_fp16(self):
    """Benchmark 1 gpu FP16."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_1_gpu(self):
    """Benchmark 1 gpu w/xla."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu')
    FLAGS.enable_xla = True
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_1_gpu_fp16(self):
    """Benchmark 1 gpu w/xla and FP16."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16')
    FLAGS.enable_xla = True
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_1_gpu_static_batch(self):
    """Benchmark 1 gpu with static batch."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_static_batch')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_1_gpu_static_batch(self):
    """Benchmark 1 gpu with static batch w/xla."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_static_batch')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.enable_xla = True
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_1_gpu_static_batch_fp16(self):
    """Benchmark 1 gpu with static batch FP16."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_static_batch_fp16')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_1_gpu_static_batch_fp16(self):
    """Benchmark 1 gpu with static batch w/xla and FP16."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = self.batch_per_gpu
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_1_gpu_static_batch_fp16')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    FLAGS.enable_xla = True
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_8_gpu(self):
    """Benchmark 8 gpu."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_8_gpu_fp16(self):
    """Benchmark 8 gpu FP16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_8_gpu(self):
    """Benchmark 8 gpu w/xla."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.enable_xla = True
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_8_gpu_fp16(self):
    """Benchmark 8 gpu w/xla and FP16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.enable_xla = True
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16')
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_8_gpu_static_batch(self):
    """Benchmark 8 gpu with static batch."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_static_batch')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_8_gpu_static_batch_fp16(self):
    """Benchmark 8 gpu with static batch FP16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_8_gpu_static_batch_fp16')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_8_gpu_static_batch(self):
    """Benchmark 8 gpu with static batch w/xla."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.enable_xla = True
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_static_batch')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)

  def benchmark_xla_8_gpu_static_batch_fp16(self):
    """Benchmark 8 gpu with static batch w/xla and FP16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.enable_xla = True
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = self.batch_per_gpu * 8
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_static_batch_fp16')
    FLAGS.static_batch = True
    FLAGS.max_length = 64
    self._run_and_report_benchmark(total_batch_size=FLAGS.batch_size,
                                   log_steps=FLAGS.log_steps)


class TransformerBaseKerasBenchmarkReal(TransformerKerasBenchmark):
  """Transformer based version real data benchmark tests."""

  def __init__(self, output_dir=TMP_DIR, root_data_dir=TMP_DIR, **kwargs):
    def_flags = {}
    def_flags['param_set'] = 'base'
    def_flags['train_steps'] = 50
    def_flags['log_steps'] = 10

    super(TransformerBaseKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags,
        root_data_dir=root_data_dir, batch_per_gpu=4096)


class TransformerBigKerasBenchmarkReal(TransformerKerasBenchmark):
  """Transformer based version real data benchmark tests."""

  def __init__(self, output_dir=TMP_DIR, root_data_dir=TMP_DIR,
               tpu=None, **kwargs):
    def_flags = {}
    def_flags['param_set'] = 'big'
    def_flags['train_steps'] = 50
    def_flags['log_steps'] = 10

    super(TransformerBigKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags,
        root_data_dir=root_data_dir, batch_per_gpu=3072,
        tpu=tpu)

  def benchmark_2x2_tpu(self):
    """Port of former snaggletooth transformer_big model on 2x2."""
    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_2x2_tpu')
    FLAGS.train_steps = 300
    FLAGS.log_steps = 150
    FLAGS.steps_between_evals = 150
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.static_batch = True
    FLAGS.use_ctl = True
    FLAGS.batch_size = 6144
    FLAGS.max_length = 64
    FLAGS.decode_batch_size = 32
    FLAGS.decode_max_length = 97
    FLAGS.padded_decode = True
    FLAGS.enable_checkpointing = False

    self._run_and_report_benchmark(
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)

  def benchmark_4x4_tpu(self):
    """Port of former GCP transformer_big model on 4x4."""
    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_4x4_tpu')
    FLAGS.train_steps = 300
    FLAGS.log_steps = 150
    FLAGS.steps_between_evals = 150
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.static_batch = True
    FLAGS.use_ctl = True
    FLAGS.batch_size = 24576
    FLAGS.max_length = 64
    FLAGS.decode_batch_size = 32
    FLAGS.decode_max_length = 97
    FLAGS.padded_decode = True
    FLAGS.enable_checkpointing = False

    self._run_and_report_benchmark(
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)

  @owner_utils.Owner('tf-graph-compiler')
  def benchmark_4x4_tpu_mlir(self):
    """Run transformer_big model on 4x4 with the MLIR Bridge enabled."""
    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_4x4_tpu')
    FLAGS.train_steps = 300
    FLAGS.log_steps = 150
    FLAGS.steps_between_evals = 150
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.static_batch = True
    FLAGS.use_ctl = True
    FLAGS.batch_size = 24576
    FLAGS.max_length = 64
    FLAGS.decode_batch_size = 32
    FLAGS.decode_max_length = 97
    FLAGS.padded_decode = True
    FLAGS.enable_checkpointing = False
    tf.config.experimental.enable_mlir_bridge()

    self._run_and_report_benchmark(
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


if __name__ == '__main__':
  tf.test.main()
