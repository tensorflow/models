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
import copy
import json
import os
import time

from absl import flags
from absl import logging
from absl.testing import flagsaver
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.benchmark import bert_benchmark_utils as benchmark_utils
from official.utils.flags import core as flags_core
from official.vision.detection import main as detection

TMP_DIR = os.getenv('TMPDIR')
FLAGS = flags.FLAGS

# pylint: disable=line-too-long
COCO_TRAIN_DATA = 'gs://tf-perfzero-data/coco/train*'
COCO_EVAL_DATA = 'gs://tf-perfzero-data/coco/val*'
COCO_EVAL_JSON = 'gs://tf-perfzero-data/coco/instances_val2017.json'
RESNET_CHECKPOINT_PATH = 'gs://cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07'
# pylint: enable=line-too-long


class DetectionBenchmarkBase(tf.test.Benchmark):
  """Base class to hold methods common to test classes."""
  local_flags = None

  def __init__(self, output_dir=None):
    self.num_gpus = 8

    if not output_dir:
      output_dir = '/tmp'
    self.output_dir = output_dir
    self.timer_callback = None

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    self.timer_callback = benchmark_utils.BenchmarkTimerCallback()

    if DetectionBenchmarkBase.local_flags is None:
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      DetectionBenchmarkBase.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(DetectionBenchmarkBase.local_flags)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        min_ap,
                        max_ap,
                        train_batch_size=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from Detection models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      min_ap: Minimum detection AP constraint to verify correctness of the
        model.
      max_ap: Maximum detection AP accuracy constraint to verify correctness of
        the model.
      train_batch_size: Train batch size. It is needed for computing
        exp_per_second.
    """
    metrics = [{
        'name': 'total_loss',
        'value': stats['total_loss'],
    }]
    if self.timer_callback:
      metrics.append({
          'name': 'exp_per_second',
          'value': self.timer_callback.get_examples_per_sec(train_batch_size)
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

  def __init__(self, output_dir=None, **kwargs):
    self.train_data_path = COCO_TRAIN_DATA
    self.eval_data_path = COCO_EVAL_DATA
    self.eval_json_path = COCO_EVAL_JSON
    self.resnet_checkpoint_path = RESNET_CHECKPOINT_PATH

    super(RetinanetBenchmarkBase, self).__init__(output_dir=output_dir)

  def _run_detection_main(self):
    """Starts detection job."""
    if self.timer_callback:
      return detection.run(callbacks=[self.timer_callback])
    else:
      return detection.run()


class RetinanetAccuracy(RetinanetBenchmarkBase):
  """Accuracy test for RetinaNet model.

  Tests RetinaNet detection task model accuracy. The naming
  convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

  def __init__(self, output_dir=TMP_DIR, **kwargs):
    super(RetinanetAccuracy, self).__init__(output_dir=output_dir)

  def _run_and_report_benchmark(self, min_ap=0.325, max_ap=0.35):
    """Starts RetinaNet accuracy benchmark test."""

    start_time_sec = time.time()
    FLAGS.mode = 'train'
    summary, _ = self._run_detection_main()
    wall_time_sec = time.time() - start_time_sec

    FLAGS.mode = 'eval'
    eval_metrics = self._run_detection_main()
    summary.update(eval_metrics)

    summary['train_batch_size'] = self.params_override['train']['batch_size']
    summary['total_steps'] = self.params_override['train']['total_steps']
    super(RetinanetAccuracy, self)._report_benchmark(
        stats=summary,
        wall_time_sec=wall_time_sec,
        min_ap=min_ap,
        max_ap=max_ap,
        train_batch_size=self.params_override['train']['batch_size'])

  def _setup(self):
    super(RetinanetAccuracy, self)._setup()
    FLAGS.strategy_type = 'mirrored'
    FLAGS.model = 'retinanet'

    self.params_override = {
        'train': {
            'batch_size': 64,
            'iterations_per_loop': 100,
            'total_steps': 22500,
            'train_file_pattern': self.train_data_path,
            'checkpoint': {
                'path': self.resnet_checkpoint_path,
                'prefix': 'resnet50/'
            },
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
    params = copy.deepcopy(self.params_override)
    FLAGS.params_override = json.dumps(params)
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_coco')
    # Sets timer_callback to None as we do not use it now.
    self.timer_callback = None

    self._run_and_report_benchmark()


class RetinanetBenchmarkReal(RetinanetAccuracy):
  """Short benchmark performance tests for RetinaNet model.

  Tests RetinaNet performance in different GPU configurations.
  The naming convention of below test cases follow
  `benchmark_(number of gpus)_gpu` format.
  """

  def __init__(self, output_dir=TMP_DIR, **kwargs):
    super(RetinanetBenchmarkReal, self).__init__(output_dir=output_dir)

  @flagsaver.flagsaver
  def benchmark_8_gpu_coco(self):
    """Run RetinaNet model accuracy test with 8 GPUs."""
    self._setup()
    params = copy.deepcopy(self.params_override)
    params['train']['total_steps'] = 1875  # One epoch.
    # The iterations_per_loop must be one, otherwise the number of examples per
    # second would be wrong. Currently only support calling callback per batch
    # when each loop only runs on one batch, i.e. host loop for one step. The
    # performance of this situation might be lower than the case of
    # iterations_per_loop > 1.
    # Related bug: b/135933080
    params['train']['iterations_per_loop'] = 1
    params['eval']['eval_samples'] = 8
    FLAGS.params_override = json.dumps(params)
    FLAGS.model_dir = self._get_model_dir('real_benchmark_8_gpu_coco')
    # Use negative value to avoid saving checkpoints.
    FLAGS.save_checkpoint_freq = -1
    if self.timer_callback is None:
      logging.error('Cannot measure performance without timer callback')
    else:
      self._run_and_report_benchmark()

  @flagsaver.flagsaver
  def benchmark_1_gpu_coco(self):
    """Run RetinaNet model accuracy test with 1 GPU."""
    self.num_gpus = 1
    self._setup()
    params = copy.deepcopy(self.params_override)
    params['train']['batch_size'] = 8
    params['train']['total_steps'] = 200
    params['train']['iterations_per_loop'] = 1
    params['eval']['eval_samples'] = 8
    FLAGS.params_override = json.dumps(params)
    FLAGS.model_dir = self._get_model_dir('real_benchmark_1_gpu_coco')
    FLAGS.strategy_type = 'one_device_gpu'
    # Use negative value to avoid saving checkpoints.
    FLAGS.save_checkpoint_freq = -1
    if self.timer_callback is None:
      logging.error('Cannot measure performance without timer callback')
    else:
      self._run_and_report_benchmark()

if __name__ == '__main__':
  tf.test.main()
