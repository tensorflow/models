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

from official.benchmark import keras_benchmark
from official.vision.image_classification import resnet_cifar_main

MIN_TOP_1_ACCURACY = 0.929
MAX_TOP_1_ACCURACY = 0.938

FLAGS = flags.FLAGS
CIFAR_DATA_DIR_NAME = 'cifar-10-batches-bin'


class Resnet56KerasAccuracy(keras_benchmark.KerasBenchmark):
  """Accuracy tests for ResNet56 Keras CIFAR-10."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    self.data_dir = os.path.join(root_data_dir, CIFAR_DATA_DIR_NAME)
    flag_methods = [resnet_cifar_main.define_cifar_flags]

    super(Resnet56KerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def benchmark_graph_1_gpu(self):
    """Test keras based model with Keras fit and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test keras based model with eager and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  def benchmark_cpu(self):
    """Test keras based model on CPU."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_cpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_cpu_no_dist_strat(self):
    """Test keras based model on CPU without distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_cpu_no_dist_strat')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_cpu_no_dist_strat_run_eagerly(self):
    """Test keras based model on CPU w/forced eager and no dist_strat."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_cpu_no_dist_strat_run_eagerly')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat(self):
    """Test keras based model with eager and no dist strat."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.explicit_gpu_placement = True
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly(self):
    """Test keras based model w/forced eager and no dist_strat."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_no_dist_strat(self):
    """Test keras based model with Keras fit but not distribution strategies."""
    self._setup()
    FLAGS.distribution_strategy = 'off'
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu_no_dist_strat')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_force_v1_path(self):
    """No dist strat forced v1 execution path."""
    self._setup()
    FLAGS.distribution_strategy = 'off'
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_force_v1_path')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.force_v2_in_keras_compile = False
    self._run_and_report_benchmark()

  def benchmark_2_gpu(self):
    """Test keras based model with eager and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  def benchmark_graph_2_gpu(self):
    """Test keras based model with Keras fit and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_2_gpu')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_cifar_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet56KerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)


class Resnet56KerasBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Short performance tests for ResNet56 via Keras and CIFAR-10."""

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [resnet_cifar_main.define_cifar_flags]

    super(Resnet56KerasBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_cifar_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet56KerasBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)

  def benchmark_1_gpu(self):
    """Test 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_xla(self):
    """Test 1 gpu with xla enabled."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = False
    FLAGS.enable_xla = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_xla')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_force_v1_path(self):
    """Test 1 gpu using forced v1 execution path."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_force_v1_path')
    FLAGS.batch_size = 128
    FLAGS.force_v2_in_keras_compile = False
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu(self):
    """Test 1 gpu graph."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.run_eagerly = False
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat(self):
    """Test 1 gpu without distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_no_dist_strat(self):
    """Test 1 gpu graph mode without distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_run_eagerly(self):
    """Test 1 gpu without distribution strategy and forced eager."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 128
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_run_eagerly')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_force_v1_path(self):
    """No dist strat but forced v1 execution path."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 128
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_force_v1_path')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.force_v2_in_keras_compile = False
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_dist_strat_force_v1_path_run_eagerly(self):
    """Forced v1 execution path and forced eager."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 128
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_1_gpu_no_dist_strat_force_v1_path_run_eagerly')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.force_v2_in_keras_compile = False
    self._run_and_report_benchmark()

  def benchmark_2_gpu(self):
    """Test 2 gpu."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = False
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu')
    FLAGS.batch_size = 128 * 2  # 2 GPUs
    self._run_and_report_benchmark()

  def benchmark_graph_2_gpu(self):
    """Test 2 gpu graph mode."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.enable_eager = False
    FLAGS.run_eagerly = False
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_2_gpu')
    FLAGS.batch_size = 128 * 2  # 2 GPUs
    self._run_and_report_benchmark()

  def benchmark_cpu(self):
    """Test cpu."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.enable_eager = True
    FLAGS.model_dir = self._get_model_dir('benchmark_cpu')
    FLAGS.batch_size = 128
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_graph_cpu(self):
    """Test cpu graph mode."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.enable_eager = False
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_cpu')
    FLAGS.batch_size = 128
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_cpu_no_dist_strat_run_eagerly(self):
    """Test cpu without distribution strategy and forced eager."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.distribution_strategy = 'off'
    FLAGS.enable_eager = True
    FLAGS.run_eagerly = True
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_cpu_no_dist_strat_run_eagerly')
    FLAGS.batch_size = 128
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_cpu_no_dist_strat(self):
    """Test cpu without distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_cpu_no_dist_strat')
    FLAGS.batch_size = 128
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()

  def benchmark_cpu_no_dist_strat_force_v1_path(self):
    """Test cpu without dist strat and force v1 in model.compile."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_cpu_no_dist_strat_force_v1_path')
    FLAGS.batch_size = 128
    FLAGS.data_format = 'channels_last'
    FLAGS.force_v2_in_keras_compile = False
    self._run_and_report_benchmark()

  def benchmark_graph_cpu_no_dist_strat(self):
    """Test cpu graph mode without distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_cpu_no_dist_strat')
    FLAGS.batch_size = 128
    FLAGS.data_format = 'channels_last'
    self._run_and_report_benchmark()


class Resnet56KerasBenchmarkSynth(Resnet56KerasBenchmarkBase):
  """Synthetic benchmarks for ResNet56 and Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    default_flags = {}
    default_flags['skip_eval'] = True
    default_flags['use_synthetic_data'] = True
    default_flags['train_steps'] = 110
    default_flags['log_steps'] = 10

    super(Resnet56KerasBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=default_flags)


class Resnet56KerasBenchmarkReal(Resnet56KerasBenchmarkBase):
  """Real data benchmarks for ResNet56 and Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    default_flags = {}
    default_flags['skip_eval'] = True
    default_flags['data_dir'] = os.path.join(root_data_dir, CIFAR_DATA_DIR_NAME)
    default_flags['train_steps'] = 110
    default_flags['log_steps'] = 10

    super(Resnet56KerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=default_flags)


if __name__ == '__main__':
  tf.test.main()
