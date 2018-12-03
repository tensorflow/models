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
import mock

import numpy as np
import tensorflow as tf

from absl import flags
from absl.testing import flagsaver
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation import neumf_model
from official.recommendation import ncf_main
from official.recommendation import stat_utils


NUM_TRAIN_NEG = 4


class NcfTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(NcfTest, cls).setUpClass()
    ncf_main.define_ncf_flags()

  def setUp(self):
    self.top_k_old = rconst.TOP_K
    self.num_eval_negatives_old = rconst.NUM_EVAL_NEGATIVES
    rconst.NUM_EVAL_NEGATIVES = 2

  def tearDown(self):
    rconst.NUM_EVAL_NEGATIVES = self.num_eval_negatives_old
    rconst.TOP_K = self.top_k_old

  def get_hit_rate_and_ndcg(self, predicted_scores_by_user, items_by_user,
                            top_k=rconst.TOP_K, match_mlperf=False):
    rconst.TOP_K = top_k
    rconst.NUM_EVAL_NEGATIVES = predicted_scores_by_user.shape[1] - 1

    g = tf.Graph()
    with g.as_default():
      logits = tf.convert_to_tensor(
          predicted_scores_by_user.reshape((-1, 1)), tf.float32)
      softmax_logits = tf.concat([tf.zeros(logits.shape, dtype=logits.dtype),
                                  logits], axis=1)
      duplicate_mask = tf.convert_to_tensor(
          stat_utils.mask_duplicates(items_by_user, axis=1), tf.float32)

      metric_ops = neumf_model.compute_eval_loss_and_metrics(
          logits=logits, softmax_logits=softmax_logits,
          duplicate_mask=duplicate_mask, num_training_neg=NUM_TRAIN_NEG,
          match_mlperf=match_mlperf).eval_metric_ops

      hr = metric_ops[rconst.HR_KEY]
      ndcg = metric_ops[rconst.NDCG_KEY]

      init = [tf.global_variables_initializer(),
              tf.local_variables_initializer()]

    with self.test_session(graph=g) as sess:
      sess.run(init)
      return sess.run([hr[1], ndcg[1]])



  def test_hit_rate_and_ndcg(self):
    # Test with no duplicate items
    predictions = np.array([
        [1., 2., 0.],  # In top 2
        [2., 1., 0.],  # In top 1
        [0., 2., 1.],  # In top 3
        [2., 3., 4.]   # In top 3
    ])
    items = np.array([
        [1, 2, 3],
        [2, 3, 1],
        [3, 2, 1],
        [2, 1, 3],
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
        [1., 2., 2., 3.],  # In top 4. MLPerf: In top 3
        [3., 1., 0., 2.],  # In top 1. MLPerf: In top 1
        [0., 2., 3., 2.],  # In top 4. MLPerf: In top 3
        [3., 2., 4., 2.]   # In top 2. MLPerf: In top 2
    ])
    items = np.array([
        [1, 2, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 2],
        [4, 3, 2, 1],
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

    # Test with duplicate items, where the predictions for the same item can
    # differ. In the MLPerf case, we should take the first prediction.
    predictions = np.array([
        [3., 2., 4., 4.],  # In top 3. MLPerf: In top 2
        [3., 4., 2., 4.],  # In top 3. MLPerf: In top 3
        [2., 3., 4., 1.],  # In top 3. MLPerf: In top 2
        [4., 3., 5., 2.]   # In top 2. MLPerf: In top 1
    ])
    items = np.array([
        [1, 2, 2, 3],
        [4, 3, 3, 2],
        [2, 1, 1, 1],
        [4, 2, 2, 1],
    ])
    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 1)
    self.assertAlmostEqual(hr, 0 / 4)
    self.assertAlmostEqual(ndcg, 0 / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 2)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, (math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 3)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (math.log(2) / math.log(3) +
                                  3 * math.log(2) / math.log(4)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 4)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (math.log(2) / math.log(3) +
                                  3 * math.log(2) / math.log(4)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 1,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 2,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 3 / 4)
    self.assertAlmostEqual(ndcg, (1 + 2 * math.log(2) / math.log(3)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 3,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + 2 * math.log(2) / math.log(3) +
                                  math.log(2) / math.log(4)) / 4)

    hr, ndcg = self.get_hit_rate_and_ndcg(predictions, items, 4,
                                          match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + 2 * math.log(2) / math.log(3) +
                                  math.log(2) / math.log(4)) / 4)

  _BASE_END_TO_END_FLAGS = {
      "batch_size": 1024,
      "train_epochs": 1,
      "use_synthetic_data": True
  }

  @flagsaver.flagsaver(**_BASE_END_TO_END_FLAGS)
  @mock.patch.object(data_preprocessing, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end(self):
    ncf_main.main(None)

  @flagsaver.flagsaver(ml_perf=True, **_BASE_END_TO_END_FLAGS)
  @mock.patch.object(data_preprocessing, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end_mlperf(self):
    ncf_main.main(None)

  @flagsaver.flagsaver(use_estimator=False, **_BASE_END_TO_END_FLAGS)
  @mock.patch.object(data_preprocessing, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end_no_estimator(self):
    ncf_main.main(None)
    flags.FLAGS.ml_perf = True
    ncf_main.main(None)

  @flagsaver.flagsaver(use_estimator=False, **_BASE_END_TO_END_FLAGS)
  @mock.patch.object(data_preprocessing, "SYNTHETIC_BATCHES_PER_EPOCH", 100)
  def test_end_to_end_while_loop(self):
    # We cannot set use_while_loop = True in the flagsaver constructor, because
    # if the flagsaver sets it to True before setting use_estimator to False,
    # the flag validator will throw an error.
    flags.FLAGS.use_while_loop = True
    ncf_main.main(None)
    flags.FLAGS.ml_perf = True
    ncf_main.main(None)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
