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
from __future__ import print_function

import os
import time

from absl import flags
import tensorflow as tf # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.resnet.ctl import ctl_benchmark
from official.resnet.ctl import ctl_imagenet_main
from official.resnet.ctl import ctl_common


MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77

FLAGS = flags.FLAGS


class Resnet50CtlAccuracy(ctl_benchmark.CtlBenchmark):
  """Benchmark accuracy tests for ResNet50 in CTL."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    flag_methods = [
        ctl_common.define_ctl_flags,
        lambda: imagenet_main.define_imagenet_flags()
    ]

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
    FLAGS.epochs_between_eval = 10
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.dtype = 'fp32'
    # Add some thread tunings to improve performance.
    FLAGS.datasets_num_private_threads = 14
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = ctl_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50CtlAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY)

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)


class Resnet50CtlBenchmarkBase(ctl_benchmark.CtlBenchmark):
  """Resnet50 benchmarks."""

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [
        ctl_common.define_ctl_flags,
        lambda: imagenet_main.define_imagenet_flags()
    ]

    super(Resnet50CtlBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = ctl_imagenet_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet50CtlBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec)

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
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Test Keras model with 8 GPUs."""
    self._setup()

    FLAGS.num_gpus = 8
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.batch_size = 128 * 8  # 8 GPUs
    self._run_and_report_benchmark()

  def fill_report_object(self, stats):
    super(Resnet50CtlBenchmarkBase, self).fill_report_object(
        stats)


class Resnet50CtlBenchmarkSynth(Resnet50CtlBenchmarkBase):
  """Resnet50 synthetic benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['use_synthetic_data'] = True
    def_flags['train_steps'] = 110

    super(Resnet50CtlBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=def_flags)


class Resnet50CtlBenchmarkReal(Resnet50CtlBenchmarkBase):
  """Resnet50 real data benchmark tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    def_flags = {}
    def_flags['skip_eval'] = True
    def_flags['report_accuracy_metrics'] = False
    def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
    def_flags['train_steps'] = 110

    super(Resnet50CtlBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=def_flags)

if __name__ == '__main__':
  tf.test.main()
