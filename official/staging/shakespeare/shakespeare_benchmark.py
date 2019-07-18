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
"""Executes Shakespeare (LSTM) benchmark and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags

from official.staging.shakespeare import shakespeare_main
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark

SHAKESPEARE_TRAIN_DATA = 'shakespeare/shakespeare.txt'
FLAGS = flags.FLAGS


class ShakespeareBenchmarkBase(PerfZeroBenchmark):
  """Base class for Shakespeare (LSTM) benchmark and accuracy tests."""

  def __init__(self, output_dir=None, default_flags=None, root_data_dir=None):
    super(ShakespeareBenchmarkBase, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=[shakespeare_main.define_flags])

  def _run_and_report_benchmark(self,
                                top_1_train_min=0.923,
                                top_1_train_max=0.93,
                                warmup=1,
                                log_steps=100):
    """Report benchmark results by writing to local protobuf file.

    Average epoch time is calculated by skipping the first epoch. This average
    ignores time spent between epoch and is recorded by begin and end epoch. To
    skip accuracy check set `top_1_train_min=None`.

    Args:
      top_1_train_min: lowest passing value.
      top_1_train_max: highest passing value.
      warmup: number of entries in `timestamp_log` to ignore.
      log_steps: How often the log was created for `timestamp_log`.
    """
    total_batch_size = FLAGS.batch_size
    metrics = []
    start_time_sec = time.time()
    stats = shakespeare_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    if top_1_train_min:
      metrics.append({'name': 'accuracy_top_1_train',
                      'value': stats['history']['RecallAt1'][-1],
                      'min_value': top_1_train_min,
                      'max_value': top_1_train_max})

    # Look for the time history callback which was used during keras.fit
    for callback in stats['callbacks']:
      if isinstance(callback, keras_utils.TimeHistory):
        epoch_timings = callback.epoch_runtime_log
        average_time = sum(epoch_timings[1:]) / len(epoch_timings[1:])
        metrics.append({'name': 'avg_epoch_time',
                        'value': average_time})

      # First entry in timestamp_log is the start of step 1. The rest of the
      # entries are the end of each step recorded.
      time_log = callback.timestamp_log
      elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
      num_examples = (
          total_batch_size * log_steps * (len(time_log) - warmup - 1))
      examples_per_sec = num_examples / elapsed
      metrics.append({'name': 'exp_per_second',
                      'value': examples_per_sec})

    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(iters=-1, wall_time=wall_time_sec,
                          metrics=metrics,
                          extras={'flags': flags_str})


class ShakespeareAccuracy(ShakespeareBenchmarkBase):
  """Shakespeare accuracy tests.

  This is not an ideal test. The best we can use for the accuracy check is to
  validate top_1 of the training set. At batch size 64 the top_1 training
  stabilizes to ~0.92 around 40-45 epochs.
  """

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Shakespeare accuracy tests.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    self.train_data = os.path.join(root_data_dir, SHAKESPEARE_TRAIN_DATA)
    super(ShakespeareAccuracy, self).__init__(
        output_dir=output_dir, root_data_dir=root_data_dir)

  def benchmark_cpu(self):
    """Benchmark cpu."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    self._run_and_report_benchmark()

  def benchmark_cpu_no_ds_run_eagerly(self):
    """Benchmark cpu without distribution strategies and run eagerly."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_ds_run_eagerly(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'

    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu(self):
    """Benchmark 1 gpu w/xla."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Benchmark 8 gpu.

    This is test is for accuracy not scaling.  The batch-size is not scaled to
    the number of gpus.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu(self):
    """Benchmark 8 gpu w/xla.

    This is test is for accuracy not scaling.  The batch-size is not scaled to
    the number of gpus.
    """
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.training_data = self.train_data
    FLAGS.batch_size = 64
    FLAGS.train_epochs = 43
    FLAGS.model_dir = ''
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()


class ShakespeareKerasBenchmarkReal(ShakespeareBenchmarkBase):
  """Benchmark accuracy tests."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmark tests w/Keras.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """
    self.train_data = os.path.join(root_data_dir, SHAKESPEARE_TRAIN_DATA)

    def_flags = {}
    def_flags['training_data'] = self.train_data
    def_flags['model_dir'] = ''
    def_flags['train_epochs'] = 4

    super(ShakespeareKerasBenchmarkReal, self).__init__(
        output_dir=output_dir,
        root_data_dir=root_data_dir,
        default_flags=def_flags)

  def benchmark_cpu(self):
    """Benchmark cpu."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.batch_size = 64
    self._run_and_report_benchmark()

  def benchmark_cpu_no_ds_run_eagerly(self):
    """Benchmark cpu without distribution strategy and run eagerly."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.batch_size = 64
    FLAGS.distribution_strategy = 'off'
    FLAGS.run_eagerly = True
    self._run_and_report_benchmark()

  def benchmark_cpu_no_ds(self):
    """Benchmark cpu without distribution strategy."""
    self._setup()
    FLAGS.num_gpus = 0
    FLAGS.batch_size = 64
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 64
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_ds(self):
    """Benchmark 1 gpu without distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 64
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_1_gpu_no_ds_run_eagerly(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 64
    FLAGS.run_eagerly = True
    FLAGS.distribution_strategy = 'off'
    self._run_and_report_benchmark()

  def benchmark_xla_1_gpu(self):
    """Benchmark 1 gpu."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 64
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def benchmark_8_gpu(self):
    """Benchmark 8 gpu."""
    self._setup()
    FLAGS.num_gpus = 8
    FLAGS.batch_size = 64 * 8
    self._run_and_report_benchmark()

  def benchmark_xla_8_gpu(self):
    """Benchmark 8 gpu w/xla."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 64 * 8
    FLAGS.enable_xla = True
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    """Run and report benchmark."""
    super(ShakespeareKerasBenchmarkReal, self)._run_and_report_benchmark(
        top_1_train_min=None)
