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
"""Executes CTL benchmarks and accuracy tests."""
# pylint: disable=line-too-long,g-bad-import-order
from __future__ import print_function

import os
import time

from absl import flags
import tensorflow as tf

from official.benchmark import owner_utils
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import resnet_ctl_imagenet_main
from official.benchmark.perfzero_benchmark import PerfZeroBenchmark
from official.benchmark import benchmark_wrappers
from official.utils.flags import core as flags_core

MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77

FLAGS = flags.FLAGS


class CtlBenchmark(PerfZeroBenchmark):
  """Base benchmark class with methods to simplify testing."""

  def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}
    super(CtlBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=self.default_flags,
        flag_methods=self.flag_methods)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        top_1_max=None,
                        top_1_min=None,
                        total_batch_size=None,
                        log_steps=None,
                        warmup=1,
                        start_time_sec=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      top_1_max: highest passing level for top_1 accuracy.
      top_1_min: lowest passing level for top_1 accuracy.
      total_batch_size: Global batch-size.
      log_steps: How often the log was created for stats['step_timestamp_log'].
      warmup: number of entries in stats['step_timestamp_log'] to ignore.
      start_time_sec: the start time of the program in seconds since epoch.
    """

    metrics = []
    if 'eval_acc' in stats:
      metrics.append({
          'name': 'accuracy_top_1',
          'value': stats['eval_acc'],
          'min_value': top_1_min,
          'max_value': top_1_max
      })
      metrics.append({'name': 'eval_loss', 'value': stats['eval_loss']})

      metrics.append({
          'name': 'top_1_train_accuracy',
          'value': stats['train_acc']
      })
      metrics.append({'name': 'train_loss', 'value': stats['train_loss']})

    if (warmup and 'step_timestamp_log' in stats and
        len(stats['step_timestamp_log']) > warmup + 1):
      # first entry in the time_log is start of step 0. The rest of the
      # entries are the end of each step recorded
      time_log = stats['step_timestamp_log']
      steps_elapsed = time_log[-1].batch_index - time_log[warmup].batch_index
      time_elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
      examples_per_sec = total_batch_size * (steps_elapsed / time_elapsed)
      metrics.append({'name': 'exp_per_second', 'value': examples_per_sec})

    if 'avg_exp_per_second' in stats:
      metrics.append({
          'name': 'avg_exp_per_second',
          'value': stats['avg_exp_per_second']
      })

    if start_time_sec and 'step_timestamp_log' in stats:
      time_log = stats['step_timestamp_log']
      # time_log[0] is recorded at the beginning of the first step.
      startup_time = time_log[0].timestamp - start_time_sec
      metrics.append({'name': 'startup_time', 'value': startup_time})

    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=-1,
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_str})


class Resnet50CtlAccuracy(CtlBenchmark):
  """Benchmark accuracy tests for ResNet50 in CTL."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
        constructor forward compatible in case PerfZero provides more named
        arguments before updating the constructor.
    """

    flag_methods = [common.define_keras_flags]

    self.data_dir = os.path.join(root_data_dir, 'imagenet')
    super(Resnet50CtlAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def benchmark_8_gpu(self):
    """Test Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16(self):
    """Test Keras model with eager, 8 GPUs with tf.keras mixed precision."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 256 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_amp(self):
    """Test Keras model with 8 GPUs and mixed precision via graph rewrite."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 256 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_ctl_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50CtlAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=FLAGS.batch_size,
        log_steps=100,
        start_time_sec=start_time_sec)


class Resnet50CtlBenchmarkBase(CtlBenchmark):
  """Resnet50 benchmarks."""

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [common.define_keras_flags]

    super(Resnet50CtlBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_ctl_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    # Warmup means the number of logged step time entries that are excluded in
    # performance report. Default to exclude 1 FLAGS.log_steps time.
    super(Resnet50CtlBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps,
        warmup=1,
        start_time_sec=start_time_sec)

  def benchmark_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test Keras model with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16(self):
    """Test Keras model with 1 GPU with tf.keras mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16')
    FLAGS.batch_size = 256
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_amp(self):
    """Test Keras model with 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_amp')
    FLAGS.batch_size = 256
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_amp(self):
    """Test Keras model with XLA and 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_amp')
    FLAGS.batch_size = 256
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def benchmark_1_gpu_eager(self):
    """Test Keras model with 1 GPU in pure eager mode."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_eager')
    FLAGS.batch_size = 120
    FLAGS.use_tf_function = False
    FLAGS.use_tf_while_loop = False
    FLAGS.single_l2_loss_op = True
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16_eager(self):
    """Test Keras model with 1 GPU with fp16 and pure eager mode."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16_eager')
    FLAGS.batch_size = 240
    FLAGS.dtype = 'fp16'
    FLAGS.use_tf_function = False
    FLAGS.use_tf_while_loop = False
    FLAGS.single_l2_loss_op = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Test Keras model with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16(self):
    """Test Keras model with 8 GPUs with tf.keras mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.dtype = 'fp16'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_eager(self):
    """Test Keras model with 8 GPUs, eager, fp32."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.use_tf_function = False
    FLAGS.use_tf_while_loop = False
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_eager')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_8_gpu_eager_fp16(self):
    """Test Keras model with 8 GPUs, eager, fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.use_tf_function = False
    FLAGS.use_tf_while_loop = False
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_eager_fp16')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_8_gpu_amp(self):
    """Test Keras model with 8 GPUs with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_amp(self):
    """Test Keras model with XLA and 8 GPUs with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_amp')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def _set_df_common(self):
    FLAGS.steps_per_loop = 500
    FLAGS.train_epochs = 2
    FLAGS.train_steps = None
    FLAGS.skip_eval = True
    FLAGS.enable_eager = True
    FLAGS.enable_tensorboard = False
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.report_accuracy_metrics = False
    FLAGS.log_steps = 50
    FLAGS.single_l2_loss_op = True
    FLAGS.use_tf_function = True
    FLAGS.enable_checkpoint_and_export = False

  def benchmark_2x2_tpu_bf16(self):
    self._setup()
    self._set_df_common()
    FLAGS.batch_size = 1024
    FLAGS.dtype = 'bf16'
    self._run_and_report_benchmark()

  def benchmark_4x4_tpu_bf16(self):
    self._setup()
    self._set_df_common()
    FLAGS.batch_size = 4096
    FLAGS.dtype = 'bf16'
    self._run_and_report_benchmark()

  @owner_utils.Owner('tf-graph-compiler')
  def benchmark_4x4_tpu_bf16_mlir(self):
    """Run resnet model on 4x4 with the MLIR Bridge enabled."""
    self._setup()
    self._set_df_common()
    FLAGS.batch_size = 4096
    FLAGS.dtype = 'bf16'
    tf.config.experimental.enable_mlir_bridge()
    self._run_and_report_benchmark()

  def benchmark_8x16_tpu_bf16(self):
    self._setup()
    self._set_df_common()
    FLAGS.batch_size = 8192
    FLAGS.dtype = 'bf16'
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(Resnet50CtlBenchmarkBase, self).fill_report_object(
        stats, total_batch_size=FLAGS.batch_size, log_steps=FLAGS.log_steps)


class Resnet50CtlBenchmarkSynth(Resnet50CtlBenchmarkBase):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110
    def_flags['steps_per_loop'] = 20
    def_flags['log_steps'] = 10

    super(Resnet50CtlBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class Resnet50CtlBenchmarkReal(Resnet50CtlBenchmarkBase):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 110
    def_flags['steps_per_loop'] = 20
    def_flags['log_steps'] = 10

    super(Resnet50CtlBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


if __name__ == '__main__':
  tf.test.main()
