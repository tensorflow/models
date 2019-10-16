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
"""Tests NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import unittest

import mock
import numpy as np
import tensorflow as tf

from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import neumf_model
from official.recommendation import ncf_common
from official.recommendation import ncf_estimator_main
from official.recommendation import ncf_keras_main
from official.utils.misc import keras_utils
from official.utils.testing import integration

from tensorflow.python.eager import context # pylint: disable=ungrouped-imports


NUM_TRAIN_NEG = 4


class NcfTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(NcfTest, cls).setUpClass()
    ncf_common.define_ncf_flags()

  def setUp(self):
    self.top_k_old = rconst.TOP_K
    self.num_eval_negatives_old = rconst.NUM_EVAL_NEGATIVES
    rconst.NUM_EVAL_NEGATIVES = 2

  def tearDown(self):
    rconst.NUM_EVAL_NEGATIVES = self.num_eval_negatives_old
    rconst.TOP_K = self.top_k_old

  @unittest.skipIf(keras_utils.is_v2_0(), "TODO(b/136018594)")
  def get_hit_rate_and_ndcg(self, predicted_scores_by_user, items_by_user,
                            top_k=rconst.TOP_K, match_mlperf=False):
    rconst.TOP_K = top_k
    rconst.NUM_EVAL_NEGATIVES = predicted_scores_by_user.shape[1] - 1
    batch_size = items_by_user.shape[0]

    users = np.repeat(np.arange(batch_size)[:, np.newaxis],
                      rconst.NUM_EVAL_NEGATIVES + 1, axis=1)
    users, items, duplicate_mask = \
      data_pipeline.BaseDataConstructor._assemble_eval_batch(
          users, items_by_user[:, -1:], items_by_user[:, :-1], batch_size)

    g = tf.Graph()
    with g.as_default():
      logits = tf.convert_to_tensor(
          predicted_scores_by_user.reshape((-1, 1)), tf.float32)
      softmax_logits = tf.concat([tf.zeros(logits.shape, dtype=logits.dtype),
                                  logits], axis=1)
      duplicate_mask = tf.convert_to_tensor(duplicate_mask, tf.float32)

      metric_ops = neumf_model._get_estimator_spec_with_metrics(
          logits=logits, softmax_logits=softmax_logits,
          duplicate_mask=duplicate_mask, num_training_neg=NUM_TRAIN_NEG,
          match_mlperf=match_mlperf).eval_metric_ops

      hr = metric_ops[rconst.HR_KEY]
      ndcg = metric_ops[rconst.NDCG_KEY]

      init = [tf.compat.v1.global_variables_initializer(),
              tf.compat.v1.local_variables_initializer()]

    with self.session(graph=g) as sess:
      sess.run(init)
      return sess.run([hr[1], ndcg[1]])

  def test_hit_rate_and_ndcg(self):
    # Test with no duplicate items
    predictions = np.array([
        [2., 0., 1.],  # In top 2
        [1., 0., 2.],  # In top 1
        [2., 1., 0.],  # In top 3
        [3., 4., 2.]   # In top 3
    ])
    items = np.array([
        [2, 3, 1],
        [3, 1, 2],
        [2, 1, 3],
        [1, 3, 2],
    ])

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 1)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 2)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 3)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 1,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 2,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 3,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

    # Test with duplicate items. In the MLPerf case, we treat the duplicates as
    # a single item. Otherwise, we treat the duplicates as separate items.
    predictions = np.array([
        [2., 2., 3., 1.],  # In top 4. MLPerf: In top 3
        [1., 0., 2., 3.],  # In top 1. MLPerf: In top 1
        [2., 3., 2., 0.],  # In top 4. MLPerf: In top 3
        [2., 4., 2., 3.]   # In top 2. MLPerf: In top 2
    ])
    items = np.array([
        [2, 2, 3, 1],
        [2, 3, 4, 1],
        [2, 3, 2, 1],
        [3, 2, 1, 4],
    ])
    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 1)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 2)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 3)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 4)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(5)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 1,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 2,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 3,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 4,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

  _BASE_END_TO_END_FLAGS = ['-batch_size', '1044', '-train_epochs', '1']

  @unittest.skipIf(keras_utils.is_v2_0(), "TODO(b/136018594)")
  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end_estimator(self):
    integration.run_synthetic(
        ncf_estimator_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS)

  @unittest.skipIf(keras_utils.is_v2_0(), "TODO(b/136018594)")
  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end_estimator_mlperf(self):
    integration.run_synthetic(
        ncf_estimator_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS + ['-ml_perf', 'True'])

  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end_keras_no_dist_strat(self):
    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS +
        ['-distribution_strategy', 'off'])

  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_end_to_end_keras_dist_strat(self):
    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS + ['-num_gpus', '0'])

  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_end_to_end_keras_dist_strat_ctl(self):
    flags = (self._BASE_END_TO_END_FLAGS +
             ['-num_gpus', '0'] +
             ['-keras_use_ctl', 'True'])
    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=flags)

  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_end_to_end_keras_1_gpu_dist_strat_fp16(self):
    if context.num_gpus() < 1:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(1, context.num_gpus()))

    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS + ['-num_gpus', '1',
                                                   '--dtype', 'fp16'])

  @mock.patch.object(rconst, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_end_to_end_keras_1_gpu_dist_strat_ctl_fp16(self):
    if context.num_gpus() < 1:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'.
          format(1, context.num_gpus()))

    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS + ['-num_gpus', '1',
                                                   '--dtype', 'fp16',
                                                   '--keras_use_ctl'])

  @mock.patch.object(rconst, 'SYNTHETIC_BATCHES_PER_EPOCH', 100)
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_end_to_end_keras_2_gpu_fp16(self):
    if context.num_gpus() < 2:
      self.skipTest(
          "{} GPUs are not available for this test. {} GPUs are available".
          format(2, context.num_gpus()))

    integration.run_synthetic(
        ncf_keras_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=self._BASE_END_TO_END_FLAGS + ['-num_gpus', '2',
                                                   '--dtype', 'fp16'])

if __name__ == "__main__":
  tf.test.main()
