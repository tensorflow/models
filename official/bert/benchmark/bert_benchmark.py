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
"""Executes BERT benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.bert import run_classifier

# pylint: disable=line-too-long
PRETRAINED_CHECKPOINT_PATH = 'gs://tf-perfzero-data/bert/bert_model.ckpt'
CLASSIFIER_TRAIN_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_train.tf_record'
CLASSIFIER_EVAL_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_eval.tf_record'
CLASSIFIER_INPUT_META_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_meta_data'
MODEL_CONFIG_FILE_PATH = 'gs://tf-perfzero-data/bert/bert_config'
# pylint: enable=line-too-long

FLAGS = flags.FLAGS


class BertBenchmarkBase(tf.test.Benchmark):
  """Base class to hold methods common to test classes in the module."""
  local_flags = None

  def __init__(self, output_dir=None):
    if not output_dir:
      output_dir = '/tmp'
    self.output_dir = output_dir

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    if BertBenchmarkBase.local_flags is None:
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      BertBenchmarkBase.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(BertBenchmarkBase.local_flags)

  def _report_benchmark(self, stats, wall_time_sec):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from BERT models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
    """
    metrics = [{
        'name': 'training_loss',
        'value': stats['train_loss'],
    }]

    if 'train_metrics' in stats:
      metrics.append({
          'name': 'train_accuracy',
          'value': stats['train_metrics'],
      })
    if 'eval_metric' in stats:
      metrics.append({
          'name': 'eval_accuracy',
          'value': stats['eval_metrics'],
      })

    self.report_benchmark(
        iters=stats['total_training_steps'],
        wall_time=wall_time_sec,
        metrics=metrics)


class BertBenchmarkAccuracyTest(BertBenchmarkBase):
  """Short benchmark tests for BERT model."""

  def __init__(self, output_dir=None, **kwargs):
    self.train_data_path = CLASSIFIER_TRAIN_DATA_PATH
    self.eval_data_path = CLASSIFIER_EVAL_DATA_PATH
    self.bert_config_file = MODEL_CONFIG_FILE_PATH
    self.input_meta_data_path = CLASSIFIER_INPUT_META_DATA_PATH
    self.pretrained_checkpoint_path = PRETRAINED_CHECKPOINT_PATH

    super(BertBenchmarkAccuracyTest, self).__init__(output_dir=output_dir)

  @flagsaver.flagsaver
  def _run_bert_classifier(self):
    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
      input_meta_data = json.loads(reader.read().decode('utf-8'))

    strategy = tf.distribute.MirroredStrategy()
    run_classifier.run_bert(strategy, input_meta_data)

  def _run_and_report_benchmark(self, training_summary_path):
    start_time_sec = time.time()
    self._run_bert_classifier()
    wall_time_sec = time.time() - start_time_sec

    with tf.io.gfile.GFile(training_summary_path, 'rb') as reader:
      summary = json.loads(reader.read().decode('utf-8'))

    super(BertBenchmarkAccuracyTest, self)._report_benchmark(
        stats=summary, wall_time_sec=wall_time_sec)

  def benchmark_8_gpu(self):
    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    FLAGS.train_data_path = self.train_data_path
    FLAGS.eval_data_path = self.eval_data_path
    FLAGS.input_meta_data_path = self.input_meta_data_path
    FLAGS.bert_config_file = self.bert_config_file
    FLAGS.init_checkpoint = self.pretrained_checkpoint_path

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)


if __name__ == '__main__':
  tf.test.main()
