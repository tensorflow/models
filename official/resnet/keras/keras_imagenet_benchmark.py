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
from absl.testing import flagsaver
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.resnet.keras import keras_common
from official.resnet.keras import keras_imagenet_main


DATA_DIR = '/data/imagenet/'


class KerasImagenetBenchmarkTests(object):
  """Benchmarks and accuracy tests for KerasCifar10."""

  local_flags = None

  def __init__(self, output_dir=None):
    self.oss_report_object = None
    self.output_dir = output_dir

  def keras_resnet50_8_gpu(self):
    """Test Keras model with Keras fit/dist_strat and 8 GPUs."""
    self._setup()
    flags.FLAGS.num_gpus = 8
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 64*8
    flags.FLAGS.train_epochs = 90
    flags.FLAGS.model_dir = self._get_model_dir('keras_resnet50_8_gpu')
    flags.FLAGS.dtype = 'fp32'
    stats = keras_imagenet_main.run(flags.FLAGS)
    self._fill_report_object(stats)

  def keras_resnet50_eager_8_gpu(self):
    """Test Keras model with eager, dist_strat and 8 GPUs."""
    self._setup()
    flags.FLAGS.num_gpus = 8
    flags.FLAGS.data_dir = DATA_DIR
    flags.FLAGS.batch_size = 64*8
    flags.FLAGS.train_epochs = 90
    flags.FLAGS.model_dir = self._get_model_dir('keras_resnet50_eager_8_gpu')
    flags.FLAGS.dtype = 'fp32'
    flags.FLAGS.enable_eager = True
    stats = keras_imagenet_main.run(flags.FLAGS)
    self._fill_report_object(stats)

  def _fill_report_object(self, stats):
    if self.oss_report_object:
      self.oss_report_object.top_1 = stats['accuracy_top_1']
      self.oss_report_object.add_other_quality(stats['training_accuracy_top_1'],
                                               'top_1_train_accuracy')
    else:
      raise ValueError('oss_report_object has not been set.')

  def _get_model_dir(self, folder_name):
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Setups up and resets flags before each test."""
    tf.logging.set_verbosity(tf.logging.DEBUG)
    if KerasImagenetBenchmarkTests.local_flags is None:
      keras_common.define_keras_flags()
      imagenet_main.define_imagenet_flags()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      KerasImagenetBenchmarkTests.local_flags = saved_flag_values
      return
    flagsaver.restore_flag_values(KerasImagenetBenchmarkTests.local_flags)
