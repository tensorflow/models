# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Executes Estimator benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main as cifar_main

DATA_DIR = '/data/cifar10_data/'


class EstimatorCifar10BenchmarkTests(object):
  """Benchmarks and accuracy tests for Estimator ResNet56."""

  local_flags = None

  def __init__(self, output_dir=None):
    self.oss_report_object = None
    self.output_dir = output_dir

  def resnet56_1_gpu(self):
    """Test layers model with Estimator and distribution strategies."""
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_1_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    stats = cifar_main.run_cifar(flags.FLAGS)
    self._fill_report_object(stats)

  def resnet56_fp16_1_gpu(self):
    """Test layers FP16 model with Estimator and distribution strategies."""
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_fp16_1_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp16'
    stats = cifar_main.run_cifar(flags.FLAGS)
    self._fill_report_object(stats)

  def resnet56_2_gpu(self):
    """Test layers model with Estimator and dist_strat. 2 GPUs."""
    self._setup()
    flags.FLAGS.num_gpus = 1
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_2_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp32'
    stats = cifar_main.run_cifar(flags.FLAGS)
    self._fill_report_object(stats)

  def resnet56_fp16_2_gpu(self):
    """Test layers FP16 model with Estimator and dist_strat. 2 GPUs."""
    self._setup()
    flags.FLAGS.num_gpus = 2
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 128
    flags.FLAGS.train_epochs = 182
    flags.FLAGS.model_dir = self._get_model_dir('resnet56_fp16_2_gpu')
    flags.FLAGS.resnet_size = 56
    flags.FLAGS.dtype = 'fp16'
    stats = cifar_main.run_cifar(flags.FLAGS)
    self._fill_report_object(stats)

  def _fill_report_object(self, stats):
    # Also "available global_step"
    if self.oss_report_object:
      self.oss_report_object.top_1 = stats['accuracy'].item()
      self.oss_report_object.top_5 = stats['accuracy_top_5'].item()
    else:
      raise ValueError('oss_report_object has not been set.')

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    if EstimatorCifar10BenchmarkTests.local_flags is None:
      cifar_main.define_cifar_flags()
      # Loads flags to get defaults to then override.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      EstimatorCifar10BenchmarkTests.local_flags = saved_flag_values
      return
    flagsaver.restore_flag_values(EstimatorCifar10BenchmarkTests.local_flags)
