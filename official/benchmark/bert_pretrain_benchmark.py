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

import json
import os
import time
from typing import Optional

from absl import flags
from absl import logging
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.benchmark import benchmark_wrappers
from official.benchmark import bert_benchmark_utils
from official.benchmark import owner_utils
from official.nlp.bert import run_pretraining
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils

# Pretrain masked lanauge modeling accuracy range:
MIN_MLM_ACCURACY = 0.635
MAX_MLM_ACCURACY = 0.645

# Pretrain next sentence prediction accuracy range:
MIN_NSP_ACCURACY = 0.94
MAX_NSP_ACCURACY = 0.96

BERT_PRETRAIN_FILES_SEQ128 = 'gs://mlcompass-data/bert/pretraining_data/seq_128/wikipedia.tfrecord*,gs://mlcompass-data/bert/pretraining_data/seq_128/books.tfrecord*'
BERT_BASE_CONFIG_FILE = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/bert_config.json'

FLAGS = flags.FLAGS


class BertPretrainAccuracyBenchmark(bert_benchmark_utils.BertBenchmarkBase):
  """Benchmark accuracy tests for BERT Pretraining."""

  def __init__(self,
               output_dir: Optional[str] = None,
               tpu: Optional[str] = None,
               **kwargs):
    """Inits BertPretrainAccuracyBenchmark class.

    Args:
      output_dir: Directory where to output e.g. log files
      tpu: TPU name to use in a TPU benchmark.
      **kwargs: Additional keyword arguments.
    """
    super(BertPretrainAccuracyBenchmark, self).__init__(
        output_dir=output_dir, tpu=tpu, **kwargs)

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self, summary_path: str, report_accuracy: bool):
    """Runs and reports the benchmark given the provided configuration."""
    distribution = distribution_utils.get_distribution_strategy(
        distribution_strategy='tpu', tpu_address=self.tpu)
    logging.info('Flags: %s', flags_core.get_nondefault_flags_as_str())
    start_time_sec = time.time()
    run_pretraining.run_bert_pretrain(
        strategy=distribution, custom_callbacks=self.timer_callback)
    wall_time_sec = time.time() - start_time_sec

    with tf.io.gfile.GFile(summary_path, 'rb') as reader:
      summary = json.loads(reader.read().decode('utf-8'))
    self._report_benchmark(summary, start_time_sec, wall_time_sec,
                           report_accuracy)

  def _report_benchmark(self, summary, start_time_sec, wall_time_sec,
                        report_accuracy):
    metrics = [{
        'name': 'train_loss',
        'value': summary['train_loss'],
    }, {
        'name':
            'exp_per_second',
        'value':
            self.timer_callback.get_examples_per_sec(FLAGS.train_batch_size *
                                                     FLAGS.steps_per_loop)
    }, {
        'name': 'startup_time',
        'value': self.timer_callback.get_startup_time(start_time_sec)
    }]
    if report_accuracy:
      metrics.extend([{
          'name': 'masked_lm_accuracy',
          'value': summary['masked_lm_accuracy'],
          'min_value': MIN_MLM_ACCURACY,
          'max_value': MAX_MLM_ACCURACY,
      }, {
          'name': 'next_sentence_accuracy',
          'value': summary['next_sentence_accuracy'],
          'min_value': MIN_NSP_ACCURACY,
          'max_value': MAX_NSP_ACCURACY,
      }])
    self.report_benchmark(
        iters=summary['total_training_steps'],
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_core.get_nondefault_flags_as_str()})

  def _specify_common_flags(self):
    FLAGS.bert_config_file = BERT_BASE_CONFIG_FILE
    FLAGS.train_batch_size = 512
    FLAGS.learning_rate = 1e-4
    FLAGS.warmup_steps = 10000
    FLAGS.steps_per_loop = 10000
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.input_files = BERT_PRETRAIN_FILES_SEQ128
    FLAGS.max_seq_length = 128
    FLAGS.max_predictions_per_seq = 20
    FLAGS.dtype = 'bf16'

  @owner_utils.Owner('tf-model-garden')
  def benchmark_accuracy_8x8_tpu_bf16_seq128_500k_steps(self):
    """Test bert pretraining with 8x8 TPU for 500k steps."""
    # This is used for accuracy test.
    self._setup()
    self._specify_common_flags()
    FLAGS.num_steps_per_epoch = 500000
    FLAGS.num_train_epochs = 1
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_accuracy_8x8_tpu_bf16_seq128_500k_steps')
    summary_path = os.path.join(FLAGS.model_dir,
                                'summaries/training_summary.txt')
    # Set train_summary_interval to -1 to disable training summary, because
    # writing summary to gcs may fail and summaries are not needed for this
    # accuracy benchmark test.
    FLAGS.train_summary_interval = -1
    self._run_and_report_benchmark(summary_path=summary_path,
                                   report_accuracy=True)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_perf_2x2_tpu_bf16_seq128_10k_steps(self):
    """Test bert pretraining with 2x2 TPU for 10000 steps."""
    self._setup()
    self._specify_common_flags()
    FLAGS.num_steps_per_epoch = 5000
    FLAGS.num_train_epochs = 2
    FLAGS.train_batch_size = 128
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_perf_2x2_tpu_bf16_seq128_10k_steps')
    summary_path = os.path.join(FLAGS.model_dir,
                                'summaries/training_summary.txt')
    # Disable accuracy check.
    self._run_and_report_benchmark(
        summary_path=summary_path, report_accuracy=False)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_perf_2x2_tpu_bf16_seq128_10k_steps_mlir(self):
    """Test bert pretraining with 2x2 TPU with MLIR for 10000 steps."""
    self._setup()
    self._specify_common_flags()
    FLAGS.num_steps_per_epoch = 5000
    FLAGS.num_train_epochs = 2
    FLAGS.train_batch_size = 128
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_perf_2x2_tpu_bf16_seq128_10k_steps_mlir')
    summary_path = os.path.join(FLAGS.model_dir,
                                'summaries/training_summary.txt')
    tf.config.experimental.enable_mlir_bridge()
    # Disable accuracy check.
    self._run_and_report_benchmark(
        summary_path=summary_path, report_accuracy=False)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_perf_4x4_tpu_bf16_seq128_10k_steps(self):
    """Test bert pretraining with 4x4 TPU for 10000 steps."""
    self._setup()
    self._specify_common_flags()
    FLAGS.num_steps_per_epoch = 5000
    FLAGS.num_train_epochs = 2
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_perf_4x4_tpu_bf16_seq128_10k_steps')
    summary_path = os.path.join(FLAGS.model_dir,
                                'summaries/training_summary.txt')
    # Disable accuracy check.
    self._run_and_report_benchmark(
        summary_path=summary_path, report_accuracy=False)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_perf_4x4_tpu_bf16_seq128_10k_steps_mlir(self):
    """Test bert pretraining with 4x4 TPU with MLIR for 10000 steps."""
    self._setup()
    self._specify_common_flags()
    FLAGS.num_steps_per_epoch = 5000
    FLAGS.num_train_epochs = 2
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_perf_4x4_tpu_bf16_seq128_10k_steps_mlir')
    summary_path = os.path.join(FLAGS.model_dir,
                                'summaries/training_summary.txt')
    tf.config.experimental.enable_mlir_bridge()
    # Disable accuracy check.
    self._run_and_report_benchmark(
        summary_path=summary_path, report_accuracy=False)

  @owner_utils.Owner('tf-model-garden')
  def benchmark_perf_8x8_tpu_bf16_seq128_10k_steps(self):
    """Test bert pretraining with 8x8 TPU for 10000 steps."""
    self._setup()
    self._specify_common_flags()
    FLAGS.num_steps_per_epoch = 5000
    FLAGS.num_train_epochs = 2
    FLAGS.model_dir = self._get_model_dir(
        'benchmark_perf_8x8_tpu_bf16_seq128_10k_steps')
    summary_path = os.path.join(FLAGS.model_dir,
                                'summaries/training_summary.txt')
    # Disable accuracy check.
    self._run_and_report_benchmark(summary_path=summary_path,
                                   report_accuracy=False)


if __name__ == '__main__':
  tf.test.main()
