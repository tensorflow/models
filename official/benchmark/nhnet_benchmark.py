# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Executes benchmark testing for bert pretraining."""
# pylint: disable=line-too-long
from __future__ import print_function

import time
from typing import Optional

from absl import flags
import tensorflow as tf

from official.benchmark import benchmark_wrappers
from official.benchmark import owner_utils
from official.benchmark import perfzero_benchmark
from official.nlp.nhnet import trainer
from official.utils.flags import core as flags_core

MIN_LOSS = 0.40
MAX_LOSS = 0.55
NHNET_DATA = 'gs://tf-perfzero-data/nhnet/v1/processed/train.tfrecord*'
PRETRAINED_CHECKPOINT_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/bert_model.ckpt'

FLAGS = flags.FLAGS


class NHNetBenchmark(perfzero_benchmark.PerfZeroBenchmark):
  """Base benchmark class for NHNet."""

  def __init__(self, output_dir=None, default_flags=None, tpu=None, **kwargs):
    self.default_flags = default_flags or {}
    flag_methods = trainer.define_flags()
    super(NHNetBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods,
        tpu=tpu,
        **kwargs)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        max_value=None,
                        min_value=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      max_value: highest passing level.
      min_value: lowest passing level.
    """

    metrics = []
    metrics.append({
        'name': 'training_loss',
        'value': stats['training_loss'],
        'min_value': min_value,
        'max_value': max_value
    })
    # These metrics are placeholders to avoid PerfZero failure.
    metrics.append({
        'name': 'exp_per_second',
        'value': 0.0,
    })
    metrics.append({
        'name': 'startup_time',
        'value': 9999.,
    })
    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=-1,
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_str})


class NHNetAccuracyBenchmark(NHNetBenchmark):
  """Benchmark accuracy tests for NHNet."""

  def __init__(self,
               output_dir: Optional[str] = None,
               tpu: Optional[str] = None,
               **kwargs):
    default_flags = dict(
        mode='train',
        train_file_pattern=NHNET_DATA,
        train_batch_size=1024,
        model_type='nhnet',
        len_title=15,
        len_passage=200,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_nhnet_articles=5,
        steps_per_loop=1000,
        params_override='init_from_bert2bert=false')
    super(NHNetAccuracyBenchmark, self).__init__(
        output_dir=output_dir, default_flags=default_flags, tpu=tpu, **kwargs)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self, max_value=MAX_LOSS, min_value=MIN_LOSS):
    """Runs and reports the benchmark given the provided configuration."""
    start_time_sec = time.time()
    stats = trainer.run()
    wall_time_sec = time.time() - start_time_sec
    self._report_benchmark(
        stats, wall_time_sec, max_value=max_value, min_value=min_value)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_accuracy_4x4_tpu_f32_50k_steps(self):
    """Test bert pretraining with 4x4 TPU for 50k steps."""
    # This is used for accuracy test.
    self._setup()
    FLAGS.train_steps = 50000
    FLAGS.checkpoint_interval = FLAGS.train_steps
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.init_checkpoint = PRETRAINED_CHECKPOINT_PATH
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_accuracy_4x4_tpu_bf32_50k_steps')
    self._run_and_report_benchmark()

  @owner_utils.Owner('tf-model-garden')
  def benchmark_accuracy_4x4_tpu_f32_1k_steps(self):
    """Test bert pretraining with 4x4 TPU for 1k steps."""
    self._setup()
    FLAGS.train_steps = 1000
    FLAGS.checkpoint_interval = FLAGS.train_steps
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_accuracy_4x4_tpu_bf32_1k_steps')
    self._run_and_report_benchmark()


if __name__ == '__main__':
  tf.test.main()
