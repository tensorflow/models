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

from absl import flags

from official.resnet import imagenet_main
from official.resnet.keras import keras_benchmark
from official.resnet.keras import keras_common
from official.resnet.keras import keras_imagenet_main

MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77
DATA_DIR = '/data/imagenet/'

FLAGS = flags.FLAGS


class Resnet50KerasAccuracy(keras_benchmark.KerasBenchmark):
  """Benchmark accuracy tests for ResNet50 in Keras."""

  def __init__(self, output_dir=None):
    flag_methods = [keras_common.define_keras_flags,
                    imagenet_main.define_imagenet_flags]

    super(Resnet50KerasAccuracy, self).__init__(output_dir=output_dir,
                                                flag_methods=flag_methods)

  def benchmark_graph_8_gpu(self):
    """Test Keras model with Keras fit/dist_strat and 8 GPUs."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = DATA_DIR
    FLAGS.batch_size = 128*8
    FLAGS.train_epochs = 90
    FLAGS.model_dir = self._get_model_dir('keras_resnet50_8_gpu')
    FLAGS.dtype = 'fp32'
    stats = keras_imagenet_main.run(FLAGS)
    self._fill_report_object(stats, FLAGS.batch_size)

  def benchmark_8_gpu(self):
    """Test Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.data_dir = DATA_DIR
    FLAGS.batch_size = 128*8
    FLAGS.train_epochs = 90
    FLAGS.model_dir = self._get_model_dir('keras_resnet50_eager_8_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    stats = keras_imagenet_main.run(FLAGS)
    self._fill_report_object(stats, FLAGS.batch_size)

  def fill_report_object(self, stats, total_batch_size):
    super(Resnet50KerasAccuracy, self).fill_report_object(
        stats,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=total_batch_size,
        log_steps=100)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)


class Resnet50KerasBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Resnet50 benchmarks."""

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [keras_common.define_keras_flags,
                    imagenet_main.define_imagenet_flags]

    super(Resnet50KerasBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags)

  def _run_benchmark(self):
    stats = keras_imagenet_main.run(FLAGS)
    self.fill_report_object(stats)

  def benchmark_1_gpu_no_dist_strat(self):
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.turn_off_distribution_strategy = True
    FLAGS.batch_size = 128

    self._run_benchmark()

  def benchmark_graph_1_gpu_no_dist_strat(self):
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.turn_off_distribution_strategy = True
    FLAGS.batch_size = 128

    self._run_benchmark()

  def benchmark_1_gpu(self):
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.turn_off_distribution_strategy = False
    FLAGS.batch_size = 128

    self._run_benchmark()

  def benchmark_graph_1_gpu(self):
    self._setup()

    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.turn_off_distribution_strategy = False
    FLAGS.batch_size = 128

    self._run_benchmark()

  def benchmark_8_gpu(self):
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = True
    FLAGS.turn_off_distribution_strategy = False
    FLAGS.batch_size = 128 * 8  # 8 GPUs

    self._run_benchmark()

  def benchmark_graph_8_gpu(self):
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.enable_eager = False
    FLAGS.turn_off_distribution_strategy = False
    FLAGS.batch_size = 128 * 8  # 8 GPUs

    self._run_benchmark()

  def fill_report_object(self, stats):
    super(Resnet50KerasBenchmarkBase, self).fill_report_object(
        stats,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)


class Resnet50KerasBenchmarkSynth(Resnet50KerasBenchmarkBase):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkSynth, self).__init__(output_dir=output_dir,
                                                      default_flags=def_flags)


class Resnet50KerasBenchmarkReal(Resnet50KerasBenchmarkBase):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['data_dir'] = DATA_DIR
    def_flags['train_steps'] = 110
    def_flags['log_steps'] = 10

    super(Resnet50KerasBenchmarkReal, self).__init__(output_dir=output_dir,
                                                     default_flags=def_flags)
