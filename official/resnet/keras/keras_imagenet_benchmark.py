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

from official.resnet.keras import keras_benchmark
from official.resnet.keras import keras_imagenet_main

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

    flag_methods = [keras_imagenet_main.define_imagenet_keras_flags]

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

  def benchmark_8_gpu_mlperf_like_tweaked(self):
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
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mlperf_like_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
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
    self._run_and_report_benchmark()

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
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = keras_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50KerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)


class Resnet50KerasBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Resnet50 benchmarks."""

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [keras_imagenet_main.define_imagenet_keras_flags]

    super(Resnet50KerasBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = keras_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec
    # Number of logged step time entries that are excluded in performance
    # report. We keep results from last 100 batches in this case.
    warmup = (FLAGS.train_steps - 100) // FLAGS.log_steps

    super(Resnet50KerasBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps,
        warmup=warmup)

  def benchmark_1_gpu_no_dist_strat(self):
    """Test Keras model with 1 GPU, no distribution strategy."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_tweaked(self):
    """Test with 1 GPU, no distribution strategy, and manual tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.explicit_gpu_placement = True
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.set_learning_phase_to_train = False
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_tweaked')
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_layout_off(self):
    """Test Keras model with 1 GPU and no layout optimizer."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_layout_off')
    FLAGS.batch_size = 128
    FLAGS.enable_grappler_layout_optimizer = False
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu(self):
    """Test Keras model with XLA and 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_layout_off(self):
    """Test Keras model with 1 GPU and xla w/no layout optimizer."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_layout_off')
    FLAGS.batch_size = 128
    FLAGS.enable_grappler_layout_optimizer = False
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16(self):
    """Test Keras model with 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16_layout_off(self):
    """Test Keras model with 1 GPU and FP16 w/no layout optimizer."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_fp16_layout_off')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.enable_grappler_layout_optimizer = False
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_fp16_dynamic(self):
    """Test Keras model with 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_layout_off(self):
    """Test Keras model with FP16+XLA w/no layout optimizer."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_layout_off')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.enable_grappler_layout_optimizer = False
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_tweaked(self):
    """Test Keras model with XLA, 1 GPU, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_slack(self):
    """Test Keras model tf.data's experimental_slack functionality."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_fp16_slack')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.tf_data_experimental_slack = True
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu_fp16_dynamic(self):
    """Test Keras model with XLA, 1 GPU, fp16, and dynamic loss scaling."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_xla_1_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 1 GPU."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_fp16(self):
    """Test Keras model in legacy graph mode with 1 GPU and fp16."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_1_gpu_fp16')
    FLAGS.batch_size = 256
    self._run_and_report_benchmark()

  def benchmark_graph_xla_1_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph with 1 GPU, fp16, XLA, and tuning."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_1_gpu_fp16_tweaked')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_graph_xla_1_gpu_fp16_slack(self):
    """Test model in legacy graph with tf.data's experimental_slack."""
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_1_gpu_fp16_slack')
    FLAGS.dtype = 'fp16'
    FLAGS.batch_size = 256
    FLAGS.tf_data_experimental_slack = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Test Keras model with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_tweaked(self):
    """Test Keras model with manual config tuning and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_tweaked')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.datasets_num_private_threads = 14
    self._run_and_report_benchmark()

  def benchmark_8_gpu_slack(self):
    """Test Keras model with tf.data's experimental_slack and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_slack')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    FLAGS.tf_data_experimental_slack = True
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu(self):
    """Test Keras model with XLA and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_tweaked(self):
    """Test Keras model with manual config tuning, 8 GPUs, and XLA."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_layout_off(self):
    """Test Keras model with 8 GPUs, fp16, and layout off."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16_layout_off')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.enable_grappler_layout_optimizer = False
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_tweaked(self):
    """Test Keras model with 8 GPUs, fp16, and manual config tuning."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_tweaked_layout_off(self):
    """Test Keras model with 8 GPUs, fp16,tuning, and layout off."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_8_gpu_fp16_tweaked_layout_off')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.data_delay_prefetch = True
    FLAGS.enable_grappler_layout_optimizer = False
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_8_gpu_fp16_dynamic_tweaked(self):
    """Test Keras model with 8 GPUs, fp16, dynamic loss scaling, and tuned."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_optional_next(self):
    """Test Keras model with XLA, 8 GPUs and fp16.

    This test also enables get_next_as_optional.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_optional_next')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.enable_get_next_as_optional = True
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16(self):
    """Test Keras model with XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_layout_off(self):
    """Test Keras model with XLA, 8 GPUs, fp16, and layout off."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16_layout_off')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.enable_grappler_layout_optimizer = False
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked(self):
    """Test Keras model with manual config tuning, XLA, 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked_layout_off(self):
    """Test with tuning, FP16+XLA, and layout_off."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_tweaked_layout_off')
    FLAGS.batch_size = 256 * 8
    FLAGS.use_tensor_lr = True
    FLAGS.enable_grappler_layout_optimizer = False
    FLAGS.data_format = 'channels_last'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_tweaked_delay_measure')
    FLAGS.batch_size = 256 * 8
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.train_steps = 310
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tweaked_optional_next(self):
    """Test Keras model with manual config tuning, XLA, 8 GPUs, fp16.

    This test also enables get_next_as_optional.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_tweaked_optional_next')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    FLAGS.enable_get_next_as_optional = True
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_slack(self):
    """Test Keras model with XLA, 8 GPUs and fp16.

    This test also enable tf.data's experimental_slack functionality.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_fp16_slack')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.tf_data_experimental_slack = True
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_dynamic_tweaked(self):
    """Test Keras model with config tuning, XLA, 8 GPUs and dynamic fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.loss_scale = 'dynamic'
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu_fp16_tensorboard_tweaked(self):
    """Test to track Tensorboard performance overhead."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = True
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_xla_8_gpu_fp16_tensorboard_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 48
    FLAGS.enable_tensorboard = True
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu(self):
    """Test Keras model in legacy graph mode with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu(self):
    """Test Keras model in legacy graph mode with XLA and 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_fp16(self):
    """Test Keras model in legacy graph mode with 8 GPUs and fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_xla_8_gpu_fp16')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_fp16_tweaked(self):
    """Test Keras model in legacy graph mode, tuning, 8 GPUs, and FP16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_tweaked_delay_measure')
    FLAGS.batch_size = 256 * 8
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.train_steps = 310
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu_fp16_tweaked_optional_next(self):
    """Test in legacy graph mode with manual config tuning, XLA, 8 GPUs, fp16.

    This test also enables get_next_as_optional.
    """
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_tweaked_optional_next')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.enable_get_next_as_optional = True
    self._run_and_report_benchmark()

  def benchmark_graph_xla_8_gpu_fp16_slack(self):
    """Test legacy graph mode with tf.data's experimental_slack."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_slack')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.tf_data_experimental_slack = True
    self._run_and_report_benchmark()

  def benchmark_graph_8_gpu_fp16_dynamic_tweaked(self):
    """Test graph Keras with config tuning, 8 GPUs and dynamic fp16."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.dtype = 'fp16'
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_graph_xla_8_gpu_fp16_dynamic_tweaked')
    FLAGS.batch_size = 256 * 8  # 8 GPUs
    FLAGS.use_tensor_lr = True
    FLAGS.loss_scale = 'dynamic'
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(Resnet50KerasBenchmarkBase, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50KerasBenchmarkSynth(Resnet50KerasBenchmarkBase):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class Resnet50KerasBenchmarkReal(Resnet50KerasBenchmarkBase):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class TrivialKerasBenchmarkReal(keras_benchmark.KerasBenchmark):
  """Trivial model with real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    flag_methods = [keras_imagenet_main.define_imagenet_keras_flags]

    def_flags = {}
    def_flags['use_trivial_model'] = True
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_tensor_lr'] = True
    def_flags['dtype'] = 'fp16'
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 600
    def_flags['log_steps'] = 100
    def_flags['distribution_strategy'] = 'default'

    super(TrivialKerasBenchmarkReal, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=def_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = keras_imagenet_main.run(FLAGS)
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


if __name__ == '__main__':
  tf.test.main()
