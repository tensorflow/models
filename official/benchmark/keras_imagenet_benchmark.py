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
from __future__ import print_function

import os
import time

from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.benchmark import keras_benchmark
from official.utils.testing import benchmark_wrappers
from official.vision.image_classification import resnet_imagenet_main

MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77

FLAGS = flags.FLAGS


class Resnet50KerasAccuracy(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for ResNet50 in Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    self.data_dir = os.path.join(root_data_dir, 'imagenet')
    super(Resnet50KerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def benchmark_graph_8_gpu(self):
    """Test Keras model with Keras fit/dist_strat and 8 GPUs."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.use_tensor_lr = True
    self._run_and_report_benchmark()

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
    FLAGS.enable_eager = True
    # Add some thread tunings to improve performance.
    FLAGS.datasets_num_private_threads = 14
    FLAGS.use_tensor_lr = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu_amp(self):
    """Test Keras model with eager, dist_strat and 8 GPUs with automatic mixed precision."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.fp16_implementation = 'graph_rewrite'
    # Add some thread tunings to improve performance.
    FLAGS.datasets_num_private_threads = 14
    FLAGS.use_tensor_lr = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16(self):
    """Test Keras model with eager, dist_strat, 8 GPUs, and fp16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 256 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    # Thread tuning to improve performance.
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.use_tensor_lr = True
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16(self):
    """Test Keras model with XLA, eager, dist_strat, 8 GPUs and fp16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 256 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    # Thread tuning to improve performance.
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.use_tensor_lr = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu_mlperf_like(self):
    """Test similar to the rules for MLPerf 0.5.

    Listed below are reasons this comparison is not to the MLSpec, but this is
    still a decent directional measurement:
      - Eval is every 4 epochs and again at the end. ~2 extra times.
      - Learning rate is not tuned to hit 75%, but we know the model is correct.
      - We measure total time and MLPerf 0.5 excluded some startup time.
      - Eval is not on the total set, need to set eval batch_size where
        8*batch_size/50K is even. 250 is a good number.
      - Not sure if we are doing any extra or too few steps due to epoch bleed.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 256 * 8
    FLAGS.train_epochs = 61
    FLAGS.epochs_between_evals = 4
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mlperf_like')
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    self._run_and_report_benchmark(top_1_min=0.736)

  def benchmark_xla_8_gpu_fp16_dynamic(self):
    """Test Keras model with XLA, eager, dist_strat, 8 GPUs, dynamic fp16."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 256 * 8
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.loss_scale = 'dynamic'
    # Thread tuning to improve performance.
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.use_tensor_lr = True
    self._run_and_report_benchmark(top_1_min=0.736)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                top_1_min=MIN_TOP_1_ACCURACY,
                                top_1_max=MAX_TOP_1_ACCURACY):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50KerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)


class Resnet50KerasBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Resnet50 benchmarks."""

  def __init__(self, output_dir=None, default_flags=None, tpu=None):
    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    super(Resnet50KerasBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags,
        tpu=tpu)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self, skip_steps=None):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec
    # Number of logged step time entries that are excluded in performance
    # report. We keep results from last 100 batches, or skip the steps based on
    # input skip_steps.
    warmup = (skip_steps or (FLAGS.train_steps - 100)) // FLAGS.log_steps

    super(Resnet50KerasBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps,
        warmup=warmup,
        start_time_sec=start_time_sec)

  def benchmark_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly(self):
    """Test Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly')
    FLAGS.batch_size = 64
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_tweaked(self):
    """Test Keras model with 1 GPU, no distribution strategy, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.explicit_gpu_placement = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_tweaked')
    FLAGS.batch_size = 64
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16(self):
    """Test with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly_fp16_tweaked(self):
    """Test with 1 GPU, no distribution strategy, fp16, run eagerly."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.explicit_gpu_placement = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_no_dist_strat(self):
    """Test Keras model in legacy graph mode with 1 GPU, no dist strat."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu_no_dist_strat')
    FLAGS.batch_size = 96  # BatchNorm is less efficient in legacy graph mode
    # due to its reliance on v1 cond.
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test Keras model with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_amp(self):
    """Test Keras model with 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_amp')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu(self):
    """Test Keras model with XLA and 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_amp(self):
    """Test Keras model with XLA and 1 GPU with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_amp')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16(self):
    """Test Keras model with 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16_dynamic(self):
    """Test Keras model with 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.loss_scale = 'dynamic'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16(self):
    """Test Keras model with XLA, 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_tweaked(self):
    """Test Keras model with XLA, 1 GPU, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_dynamic(self):
    """Test Keras model with XLA, 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_dynamic')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.loss_scale = 'dynamic'
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu(self):
    """Test Keras model in legacy graph mode with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_xla_1_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_fp16(self):
    """Test Keras model in legacy graph mode with 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu_fp16')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_graph_xla_1_gpu_fp16(self):
    """Test Keras model in legacy graph mode with 1 GPU, fp16 and XLA."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_1_gpu_fp16')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_graph_xla_1_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph with 1 GPU, fp16, XLA, and tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_1_gpu_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Test Keras model with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_amp(self):
    """Test Keras model with 8 GPUs with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_tweaked(self):
    """Test Keras model with manual config tuning and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_tweaked')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.datasets_num_private_threads = 14
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu(self):
    """Test Keras model with XLA and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_amp(self):
    """Test Keras model with XLA and 8 GPUs with automatic mixed precision."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.dtype = 'fp16'
    FLAGS.fp16_implementation = 'graph_rewrite'
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_amp')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_tweaked(self):
    """Test Keras model with manual config tuning, 8 GPUs, and XLA."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_tweaked')
    FLAGS.batch_size = 128 * 8
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 24
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16(self):
    """Test Keras model with 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_tweaked(self):
    """Test Keras model with 8 GPUs, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_dynamic_tweaked(self):
    """Test Keras model with 8 GPUs, fp16, dynamic loss scaling, and tuned."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16(self):
    """Test Keras model with XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked(self):
    """Test Keras model with manual config tuning, XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked_delay_measure(self):
    """Test with manual config tuning, XLA, 8 GPUs and fp16.

    Delay performance measurement for stable performance on 96 vCPU platforms.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_tweaked_delay_measure')
    FLAGS.batch_size = 256 * 8
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.train_steps = 310
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_dynamic_tweaked(self):
    """Test Keras model with config tuning, XLA, 8 GPUs and dynamic fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu(self):
    """Test Keras model in legacy graph mode with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_fp16(self):
    """Test Keras model in legacy graph mode with 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu_fp16(self):
    """Test Keras model in legacy graph mode with XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph mode, tuning, 8 GPUs, and FP16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph tuning, XLA_FP16, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu_fp16_tweaked_delay_measure(self):
    """Test in legacy graph mode with manual config tuning, XLA, 8 GPUs, fp16.

    Delay performance measurement for stable performance on 96 vCPU platforms.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_tweaked_delay_measure')
    FLAGS.batch_size = 256 * 8
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.train_steps = 310
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_fp16_dynamic_tweaked(self):
    """Test graph Keras with config tuning, 8 GPUs and dynamic fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu_fp16_dynamic_tweaked(self):
    """Test graph Keras with config tuning, XLA, 8 GPUs and dynamic fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.loss_scale = 'dynamic'
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_2x2_tpu_fp16(self):
    """Test Keras model with 2x2 TPU, fp16."""
    self._setup()

    FLAGS.dtype = 'bf16'
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.model_dir = self._get_model_dir('benchmark_2x2_tpu_fp16')
    FLAGS.batch_size = 1024
    self._run_and_report_benchmark()

  def benchmark_4x4_tpu_fp16(self):
    """Test Keras model with 4x4 TPU, fp16."""
    self._setup()

    FLAGS.dtype = 'bf16'
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.model_dir = self._get_model_dir('benchmark_4x4_tpu_fp16')
    FLAGS.batch_size = 4096
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(Resnet50KerasBenchmarkBase, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50KerasBenchmarkSynth(Resnet50KerasBenchmarkBase):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, tpu=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags, tpu=tpu)


class Resnet50KerasBenchmarkReal(Resnet50KerasBenchmarkBase):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, tpu=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags, tpu=tpu)


class Resnet50KerasBenchmarkRemoteData(Resnet50KerasBenchmarkBase):
  """Resnet50 real data (stored in remote storage) benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    # Defining multiple epochs overrides the train_steps setting in benchmarks.
    def_flags['train_epochs'] = 2
    # Cache dataset so performance is stable after the first epoch.
    def_flags['training_dataset_cache'] = True
    def_flags['log_steps'] = 100

    super(Resnet50KerasBenchmarkRemoteData, self).__init__(
        output_dir=output_dir, default_flags=def_flags)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    # skip the first epoch for performance measurement.
    super(Resnet50KerasBenchmarkRemoteData,
          self)._run_and_report_benchmark(skip_steps=600)


class TrivialKerasBenchmarkReal(keras_benchmark.KerasBenchmark):
  """Trivial model with real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]

    def_flags = {}
    def_flags['use_trivial_model'] = True
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_tensor_lr'] = True
    def_flags['dtype'] = 'fp16'
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 600
    def_flags['log_steps'] = 100
    def_flags['distribution_strategy'] = 'mirrored'

    super(TrivialKerasBenchmarkReal, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=def_flags)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(TrivialKerasBenchmarkReal, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)

  def benchmark_8_gpu_warmup(self):
    """Dummy test that runs over an epoch to warmup the machine."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_warmup')
    FLAGS.batch_size = 256 * 8
    FLAGS.train_steps = 700
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test trivial Keras model (input pipeline) with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu(self):
    """Test trivial Keras model (input pipeline) with 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Test trivial Keras model (input pipeline) with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.batch_size = 256 * 8
    self._run_and_report_benchmark()

  def benchmark_8_gpu_tweaked(self):
    """Test trivial Keras model with tuning and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_tweaked')
    FLAGS.batch_size = 256 * 8
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu(self):
    """Test trivial Keras model in legacy graph mode with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
    FLAGS.batch_size = 256 * 8
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_tweaked(self):
    """Test trivial Keras model in legacy graph mode with tuning and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu_tweaked')
    FLAGS.batch_size = 256 * 8
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(TrivialKerasBenchmarkReal, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50MultiWorkerKerasAccuracy(keras_benchmark.KerasBenchmark):
  """Resnet50 distributed accuracy tests with multiple workers."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]
    self.data_dir = os.path.join(root_data_dir, 'imagenet')
    super(Resnet50MultiWorkerKerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def _benchmark_common(self, eager, num_workers, all_reduce_alg):
    """Common to all benchmarks in this class."""
    self._setup()

    num_gpus = 8
    FLAGS.num_gpus = num_gpus
    FLAGS.data_dir = self.data_dir
    FLAGS.train_epochs = 90
    FLAGS.epochs_between_evals = 10
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = eager
    FLAGS.enable_xla = False
    FLAGS.distribution_strategy = 'multi_worker_mirrored'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 32
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_{}_8_gpu_{}_worker_fp16_{}_tweaked'.format(
            'eager' if eager else 'graph', num_workers, all_reduce_alg))
    FLAGS.batch_size = 256 * num_gpus * num_workers
    FLAGS.all_reduce_alg = all_reduce_alg

    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                top_1_min=MIN_TOP_1_ACCURACY,
                                top_1_max=MAX_TOP_1_ACCURACY):
    start_time_sec = time.time()
    stats = resnet_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50MultiWorkerKerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=top_1_min,
        top_1_max=top_1_max,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def benchmark_eager_8_gpu_2_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_2_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='nccl')

  def benchmark_eager_8_gpu_8_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_8_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='nccl')


class Resnet50MultiWorkerKerasBenchmark(Resnet50KerasBenchmarkBase):
  """Resnet50 distributed benchmark tests with multiple workers."""

  def __init__(self, output_dir=None, default_flags=None):
    super(Resnet50MultiWorkerKerasBenchmark, self).__init__(
        output_dir=output_dir, default_flags=default_flags)

  def _benchmark_common(self, eager, num_workers, all_reduce_alg):
    """Common to all benchmarks in this class."""
    self._setup()

    num_gpus = 8
    FLAGS.num_gpus = num_gpus
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = eager
    FLAGS.enable_xla = False
    FLAGS.distribution_strategy = 'multi_worker_mirrored'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 32
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_{}_8_gpu_{}_worker_fp16_{}_tweaked'.format(
            'eager' if eager else 'graph', num_workers, all_reduce_alg))
    FLAGS.batch_size = 256 * num_gpus * num_workers
    FLAGS.all_reduce_alg = all_reduce_alg

    self._run_and_report_benchmark()

  def benchmark_eager_8_gpu_1_worker_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 1 worker, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=1, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_1_worker_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 1 worker, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=1, all_reduce_alg='nccl')

  def benchmark_eager_8_gpu_2_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_2_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 2 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=2, all_reduce_alg='nccl')

  def benchmark_eager_8_gpu_8_workers_fp16_ring_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, ring all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='ring')

  def benchmark_eager_8_gpu_8_workers_fp16_nccl_tweaked(self):
    """Eager, 8 GPUs per worker, 8 workers, fp16, nccl all-reduce."""
    self._benchmark_common(eager=True, num_workers=8, all_reduce_alg='nccl')


class Resnet50MultiWorkerKerasBenchmarkSynth(Resnet50MultiWorkerKerasBenchmark):
  """Resnet50 multi-worker synthetic data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50MultiWorkerKerasBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class Resnet50MultiWorkerKerasBenchmarkReal(Resnet50MultiWorkerKerasBenchmark):
  """Resnet50 multi-worker real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50MultiWorkerKerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


if __name__ == '__main__':
  tf.test.main()
