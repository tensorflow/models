# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Tests for Ranking losses."""

import tensorflow as tf
from delf.python.training.losses import ranking_losses


class RankingLossesTest(tf.test.TestCase):

  def testContrastiveLoss(self):
    # Testing the correct numeric value.
    queries = tf.math.l2_normalize(tf.constant([[1.0, 2.0, -2.0]]))
    positives = tf.math.l2_normalize(tf.constant([[-1.0, 2.0, 0.0]]))
    negatives = tf.math.l2_normalize(tf.constant([[[-5.0, 0.0, 3.0]]]))

    result = ranking_losses.contrastive_loss(queries, positives, negatives,
                                             margin=0.7, eps=1e-6)
    exp_output = 0.55278635
    self.assertAllClose(exp_output, result)

  def testTripletLossZeroLoss(self):
    # Testing the correct numeric value in case if query-positive distance is
    # smaller than the query-negative distance.
    queries = tf.math.l2_normalize(tf.constant([[1.0, 2.0, -2.0]]))
    positives = tf.math.l2_normalize(tf.constant([[-1.0, 2.0, 0.0]]))
    negatives = tf.math.l2_normalize(tf.constant([[[-5.0, 0.0, 3.0]]]))

    result = ranking_losses.triplet_loss(queries, positives, negatives,
                                         margin=0.1)
    exp_output = 0.0
    self.assertAllClose(exp_output, result)

  def testTripletLossNonZeroLoss(self):
    # Testing the correct numeric value in case if query-positive distance is
    # bigger than the query-negative distance.
    queries = tf.math.l2_normalize(tf.constant([[1.0, 2.0, -2.0]]))
    positives = tf.math.l2_normalize(tf.constant([[-5.0, 0.0, 3.0]]))
    negatives = tf.math.l2_normalize(tf.constant([[[-1.0, 2.0, 0.0]]]))

    result = ranking_losses.triplet_loss(queries, positives, negatives,
                                         margin=0.1)
    exp_output = 2.2520838
    self.assertAllClose(exp_output, result)


if __name__ == '__main__':
  tf.test.main()
