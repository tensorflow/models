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
"""Executes XLNet benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

# pylint: disable=g-bad-import-order
from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.benchmark import bert_benchmark_utils as benchmark_utils
from official.nlp.xlnet import run_classifier

# pylint: disable=line-too-long
PRETRAINED_CHECKPOINT_PATH = 'gs://cloud-tpu-checkpoints/xlnet/large/xlnet_model-1'
CLASSIFIER_TRAIN_DATA_PATH = 'gs://tf-perfzero-data/xlnet/imdb/spiece.model.len-512.train.tf_record'
CLASSIFIER_EVAL_DATA_PATH = 'gs://tf-perfzero-data/xlnet/imdb/spiece.model.len-512.dev.eval.tf_record'
# pylint: enable=line-too-long

FLAGS = flags.FLAGS


class XLNetClassifyBenchmarkBase(benchmark_utils.BertBenchmarkBase):
  """Base class to hold methods common to test classes in the module."""

  def __init__(self, output_dir=None):
    super(XLNetClassifyBenchmarkBase, self).__init__(output_dir)
    self.num_epochs = None
    self.num_steps_per_epoch = None

  @flagsaver.flagsaver
  def _run_xlnet_classifier(self):
    """Starts XLNet classification task."""
    run_classifier.main(unused_argv=None)


class XLNetClassifyAccuracy(XLNetClassifyBenchmarkBase):
  """Short accuracy test for XLNet model.

  Tests XLNet classification task model accuracy. The naming
  convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

  def __init__(self, output_dir=None, **kwargs):
    self.train_data_path = CLASSIFIER_TRAIN_DATA_PATH
    self.eval_data_path = CLASSIFIER_EVAL_DATA_PATH
    self.pretrained_checkpoint_path = PRETRAINED_CHECKPOINT_PATH

    super(XLNetClassifyAccuracy, self).__init__(output_dir=output_dir)

  def _run_and_report_benchmark(self,
                                training_summary_path,
                                min_accuracy=0.95,
                                max_accuracy=0.97):
    """Starts XLNet accuracy benchmark test."""

    start_time_sec = time.time()
    self._run_xlnet_classifier()
    wall_time_sec = time.time() - start_time_sec

    with tf.io.gfile.GFile(training_summary_path, 'rb') as reader:
      summary = json.loads(reader.read().decode('utf-8'))

    super(XLNetClassifyAccuracy, self)._report_benchmark(
        stats=summary,
        wall_time_sec=wall_time_sec,
        min_accuracy=min_accuracy,
        max_accuracy=max_accuracy)

  def _setup(self):
    super(XLNetClassifyAccuracy, self)._setup()
    FLAGS.train_data_size = 25000
    FLAGS.test_data_size = 25024
    FLAGS.train_batch_size = 16
    FLAGS.seq_len = 512
    FLAGS.reuse_len = 256
    FLAGS.mem_len = 0
    FLAGS.n_layer = 24
    FLAGS.d_model = 1024
    FLAGS.d_embed = 1024
    FLAGS.n_head = 16
    FLAGS.d_head = 64
    FLAGS.d_inner = 4096
    FLAGS.untie_r = True
    FLAGS.n_class = 2
    FLAGS.ff_activation = 'gelu'
    FLAGS.strategy_type = 'mirror'
    FLAGS.learning_rate = 2e-5
    FLAGS.train_steps = 4000
    FLAGS.warmup_steps = 500
    FLAGS.iterations = 200
    FLAGS.bi_data = False
    FLAGS.init_checkpoint = self.pretrained_checkpoint_path
    FLAGS.train_tfrecord_path = self.train_data_path
    FLAGS.test_tfrecord_path = self.eval_data_path

  def benchmark_8_gpu_imdb(self):
    """Run XLNet model accuracy test with 8 GPUs."""
    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_imdb')
    # Sets timer_callback to None as we do not use it now.
    self.timer_callback = None

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)


if __name__ == '__main__':
  tf.test.main()
