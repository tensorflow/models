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
"""Executes RetinaNet benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
import json
import time

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.benchmark import benchmark_wrappers
from official.benchmark import perfzero_benchmark
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.vision.detection import main as detection
from official.vision.detection.configs import base_config

FLAGS = flags.FLAGS

# pylint: disable=line-too-long
COCO_TRAIN_DATA = 'gs://tf-perfzero-data/coco/train*'
COCO_EVAL_DATA = 'gs://tf-perfzero-data/coco/val*'
COCO_EVAL_JSON = 'gs://tf-perfzero-data/coco/instances_val2017.json'
RESNET_CHECKPOINT_PATH = 'gs://cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07'
# pylint: enable=line-too-long


class TimerCallback(keras_utils.TimeHistory):
  """TimeHistory subclass for benchmark reporting."""

  def get_examples_per_sec(self, warmup=1):
    # First entry in timestamp_log is the start of the step 1. The rest of the
    # entries are the end of each step recorded.
    time_log = self.timestamp_log
    seconds = time_log[-1].timestamp - time_log[warmup].timestamp
    steps = time_log[-1].batch_index - time_log[warmup].batch_index
    return self.batch_size * steps / seconds

  def get_startup_time(self, start_time_sec):
    return self.timestamp_log[0].timestamp - start_time_sec


class DetectionBenchmarkBase(perfzero_benchmark.PerfZeroBenchmark):
  """Base class to hold methods common to test classes."""

  def __init__(self, **kwargs):
    super(DetectionBenchmarkBase, self).__init__(**kwargs)
    self.timer_callback = None

  def _report_benchmark(self, stats, start_time_sec, wall_time_sec, min_ap,
                        max_ap, warmup):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from Detection models with known entries.
      start_time_sec: the start of the benchmark execution in seconds
      wall_time_sec: the duration of the benchmark execution in seconds
      min_ap: Minimum detection AP constraint to verify correctness of the
        model.
      max_ap: Maximum detection AP accuracy constraint to verify correctness of
        the model.
      warmup: Number of time log entries to ignore when computing examples/sec.
    """
    metrics = [{
        'name': 'total_loss',
        'value': stats['total_loss'],
    }]
    if self.timer_callback:
      metrics.append({
          'name': 'exp_per_second',
          'value': self.timer_callback.get_examples_per_sec(warmup)
      })
      metrics.append({
          'name': 'startup_time',
          'value': self.timer_callback.get_startup_time(start_time_sec)
      })
    else:
      metrics.append({
          'name': 'exp_per_second',
          'value': 0.0,
      })

    if 'eval_metrics' in stats:
      metrics.append({
          'name': 'AP',
          'value': stats['AP'],
          'min_value': min_ap,
          'max_value': max_ap,
      })
    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=stats['total_steps'],
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_str})


class RetinanetBenchmarkBase(DetectionBenchmarkBase):
  """Base class to hold methods common to test classes in the module."""

  def __init__(self, **kwargs):
    self.train_data_path = COCO_TRAIN_DATA
    self.eval_data_path = COCO_EVAL_DATA
    self.eval_json_path = COCO_EVAL_JSON
    self.resnet_checkpoint_path = RESNET_CHECKPOINT_PATH
    super(RetinanetBenchmarkBase, self).__init__(**kwargs)

  def _run_detection_main(self):
    """Starts detection job."""
    if self.timer_callback:
      FLAGS.log_steps = 0  # prevent detection.run from adding the same callback
      return detection.run(callbacks=[self.timer_callback])
    else:
      return detection.run()


class RetinanetAccuracy(RetinanetBenchmarkBase):
  """Accuracy test for RetinaNet model.

  Tests RetinaNet detection task model accuracy. The naming
  convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self,
                                params,
                                min_ap=0.325,
                                max_ap=0.35,
                                do_eval=True,
                                warmup=1):
    """Starts RetinaNet accuracy benchmark test."""
    FLAGS.params_override = json.dumps(params)
    # Need timer callback to measure performance
    self.timer_callback = TimerCallback(
        batch_size=params['train']['batch_size'],
        log_steps=FLAGS.log_steps,
    )

    start_time_sec = time.time()
    FLAGS.mode = 'train'
    summary, _ = self._run_detection_main()
    wall_time_sec = time.time() - start_time_sec

    if do_eval:
      FLAGS.mode = 'eval'
      eval_metrics = self._run_detection_main()
      summary.update(eval_metrics)

    summary['total_steps'] = params['train']['total_steps']
    self._report_benchmark(summary, start_time_sec, wall_time_sec, min_ap,
                           max_ap, warmup)

  def _setup(self):
    super(RetinanetAccuracy, self)._setup()
    FLAGS.model = 'retinanet'

  def _params(self):
    return {
        'train': {
            'batch_size': 64,
            'iterations_per_loop': 100,
            'total_steps': 22500,
            'train_file_pattern': self.train_data_path,
            'checkpoint': {
                'path': self.resnet_checkpoint_path,
                'prefix': 'resnet50/'
            },
            # Speed up ResNet training when loading from the checkpoint.
            'frozen_variable_prefix': base_config.RESNET_FROZEN_VAR_PREFIX,
        },
        'eval': {
            'batch_size': 8,
            'eval_samples': 5000,
            'val_json_file': self.eval_json_path,
            'eval_file_pattern': self.eval_data_path,
        },
    }

  @flagsaver.flagsaver
  def benchmark_8_gpu_coco(self):
    """Run RetinaNet model accuracy test with 8 GPUs."""
    self._setup()
    params = self._params()
    FLAGS.num_gpus = 8
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_coco')
    FLAGS.strategy_type = 'mirrored'
    self._run_and_report_benchmark(params)


class RetinanetBenchmarkReal(RetinanetAccuracy):
  """Short benchmark performance tests for RetinaNet model.

  Tests RetinaNet performance in different GPU configurations.
  The naming convention of below test cases follow
  `benchmark_(number of gpus)_gpu` format.
  """

  def _setup(self):
    super(RetinanetBenchmarkReal, self)._setup()
    # Use negative value to avoid saving checkpoints.
    FLAGS.save_checkpoint_freq = -1

  @flagsaver.flagsaver
  def benchmark_8_gpu_coco(self):
    """Run RetinaNet model accuracy test with 8 GPUs."""
    self._setup()
    params = self._params()
    params['train']['total_steps'] = 1875  # One epoch.
    # The iterations_per_loop must be one, otherwise the number of examples per
    # second would be wrong. Currently only support calling callback per batch
    # when each loop only runs on one batch, i.e. host loop for one step. The
    # performance of this situation might be lower than the case of
    # iterations_per_loop > 1.
    # Related bug: b/135933080
    params['train']['iterations_per_loop'] = 1
    params['eval']['eval_samples'] = 8
    FLAGS.num_gpus = 8
    FLAGS.model_dir = self._get_model_dir('real_benchmark_8_gpu_coco')
    FLAGS.strategy_type = 'mirrored'
    self._run_and_report_benchmark(params)

  @flagsaver.flagsaver
  def benchmark_1_gpu_coco(self):
    """Run RetinaNet model accuracy test with 1 GPU."""
    self._setup()
    params = self._params()
    params['train']['batch_size'] = 8
    params['train']['total_steps'] = 200
    params['train']['iterations_per_loop'] = 1
    params['eval']['eval_samples'] = 8
    FLAGS.num_gpus = 1
    FLAGS.model_dir = self._get_model_dir('real_benchmark_1_gpu_coco')
    FLAGS.strategy_type = 'one_device'
    self._run_and_report_benchmark(params)

  @flagsaver.flagsaver
  def benchmark_xla_1_gpu_coco(self):
    """Run RetinaNet model accuracy test with 1 GPU and XLA enabled."""
    self._setup()
    params = self._params()
    params['train']['batch_size'] = 8
    params['train']['total_steps'] = 200
    params['train']['iterations_per_loop'] = 1
    params['eval']['eval_samples'] = 8
    FLAGS.num_gpus = 1
    FLAGS.model_dir = self._get_model_dir('real_benchmark_xla_1_gpu_coco')
    FLAGS.strategy_type = 'one_device'
    FLAGS.enable_xla = True
    self._run_and_report_benchmark(params)

  @flagsaver.flagsaver
  def benchmark_2x2_tpu_coco(self):
    """Run RetinaNet model accuracy test with 4 TPUs."""
    self._setup()
    params = self._params()
    params['train']['batch_size'] = 64
    params['train']['total_steps'] = 1875  # One epoch.
    params['train']['iterations_per_loop'] = 500
    FLAGS.model_dir = self._get_model_dir('real_benchmark_2x2_tpu_coco')
    FLAGS.strategy_type = 'tpu'
    self._run_and_report_benchmark(params, do_eval=False, warmup=0)


if __name__ == '__main__':
  tf.test.main()
