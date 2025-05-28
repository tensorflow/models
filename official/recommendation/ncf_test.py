# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf, tf_keras
from tensorflow.python.eager import context  # pylint: disable=ungrouped-imports
from official.recommendation import constants as rconst
from official.recommendation import ncf_common
from official.recommendation import ncf_keras_main
from official.utils.testing import integration

NUM_TRAIN_NEG = 4


class NcfTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(NcfTest, cls).setUpClass()
    ncf_common.define_ncf_flags()

  def setUp(self):
    super().setUp()
    self.top_k_old = rconst.TOP_K
    self.num_eval_negatives_old = rconst.NUM_EVAL_NEGATIVES
    rconst.NUM_EVAL_NEGATIVES = 2

  def tearDown(self):
    super().tearDown()
    rconst.NUM_EVAL_NEGATIVES = self.num_eval_negatives_old
    rconst.TOP_K = self.top_k_old

  _BASE_END_TO_END_FLAGS = ['-batch_size', '1044', '-train_epochs', '1']

  @unittest.mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  def test_end_to_end_keras_no_dist_strat(self):
    integration.run_synthetic(
        ncf_keras_main.main,
        tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS +
        ['-distribution_strategy', 'off'])

  @unittest.mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  def test_end_to_end_keras_dist_strat(self):
    integration.run_synthetic(
        ncf_keras_main.main,
        tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS + ['-num_gpus', '0'])

  @unittest.mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  def test_end_to_end_keras_dist_strat_ctl(self):
    flags = (
        self._BASE_END_TO_END_FLAGS + ['-num_gpus', '0'] +
        ['-keras_use_ctl', 'True'])
    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(), extra_flags=flags)

  @unittest.mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  def test_end_to_end_keras_1_gpu_dist_strat_fp16(self):
    if context.num_gpus() < 1:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'
          .format(1, context.num_gpus()))

    integration.run_synthetic(
        ncf_keras_main.main,
        tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS +
        ['-num_gpus', '1', '--dtype', 'fp16'])

  @unittest.mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  def test_end_to_end_keras_1_gpu_dist_strat_ctl_fp16(self):
    if context.num_gpus() < 1:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'
          .format(1, context.num_gpus()))

    integration.run_synthetic(
        ncf_keras_main.main,
        tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS +
        ['-num_gpus', '1', '--dtype', 'fp16', '--keras_use_ctl'])

  @unittest.mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  def test_end_to_end_keras_2_gpu_fp16(self):
    if context.num_gpus() < 2:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'
          .format(2, context.num_gpus()))

    integration.run_synthetic(
        ncf_keras_main.main,
        tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS +
        ['-num_gpus', '2', '--dtype', 'fp16'])


if __name__ == '__main__':
  tf.test.main()
