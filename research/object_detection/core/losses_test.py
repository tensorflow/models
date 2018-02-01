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

"""Tests for google3.research.vale.object_detection.losses."""
import math

import numpy as np
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import matcher


class WeightedL2LocalizationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    batch_size = 3
    num_anchors = 10
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], tf.float32)
    loss_op = losses.WeightedL2LocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    expected_loss = (3 * 5 * 4) / 2.0
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectAnchorwiseLoss(self):
    batch_size = 3
    num_anchors = 16
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.ones([batch_size, num_anchors])
    loss_op = losses.WeightedL2LocalizationLoss(anchorwise_output=True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    expected_loss = np.ones((batch_size, num_anchors)) * 2
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectLossSum(self):
    batch_size = 3
    num_anchors = 16
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.ones([batch_size, num_anchors])
    loss_op = losses.WeightedL2LocalizationLoss(anchorwise_output=False)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    expected_loss = tf.nn.l2_loss(prediction_tensor - target_tensor)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      expected_loss_output = sess.run(expected_loss)
      self.assertAllClose(loss_output, expected_loss_output)

  def testReturnsCorrectNanLoss(self):
    batch_size = 3
    num_anchors = 10
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.concat([
        tf.zeros([batch_size, num_anchors, code_size / 2]),
        tf.ones([batch_size, num_anchors, code_size / 2]) * np.nan
    ], axis=2)
    weights = tf.ones([batch_size, num_anchors])
    loss_op = losses.WeightedL2LocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                   ignore_nan_targets=True)

    expected_loss = (3 * 5 * 4) / 2.0
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, expected_loss)


class WeightedSmoothL1LocalizationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    batch_size = 2
    num_anchors = 3
    code_size = 4
    prediction_tensor = tf.constant([[[2.5, 0, .4, 0],
                                      [0, 0, 0, 0],
                                      [0, 2.5, 0, .4]],
                                     [[3.5, 0, 0, 0],
                                      [0, .4, 0, .9],
                                      [0, 0, 1.5, 0]]], tf.float32)
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.constant([[2, 1, 1],
                           [0, 3, 0]], tf.float32)
    loss_op = losses.WeightedSmoothL1LocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = 7.695
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class WeightedIOULocalizationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    prediction_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                      [0, 0, 1, 1],
                                      [0, 0, .5, .25]]])
    target_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                  [0, 0, 1, 1],
                                  [50, 50, 500.5, 100.25]]])
    weights = [[1.0, .5, 2.0]]
    loss_op = losses.WeightedIOULocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)
    exp_loss = 2.0
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class WeightedSigmoidClassificationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSigmoidClassificationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = -2 * math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSigmoidClassificationLoss(True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithClassIndices(self):
    prediction_tensor = tf.constant([[[-100, 100, -100, 100],
                                      [100, -100, -100, -100],
                                      [100, 0, -100, 100],
                                      [-100, -100, 100, -100]],
                                     [[-100, 0, 100, 100],
                                      [-100, 100, -100, 100],
                                      [100, 100, 100, 100],
                                      [0, 0, -1, 100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0, 0],
                                  [1, 0, 0, 1],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 1]],
                                 [[0, 0, 1, 0],
                                  [0, 1, 0, 0],
                                  [1, 1, 1, 0],
                                  [1, 0, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    # Ignores the last class.
    class_indices = tf.constant([0, 1, 2], tf.int32)
    loss_op = losses.WeightedSigmoidClassificationLoss(True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                   class_indices=class_indices)

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


def _logit(probability):
  return math.log(probability / (1. - probability))


class SigmoidFocalClassificationLossTest(tf.test.TestCase):

  def testEasyExamplesProduceSmallLossComparedToSigmoidXEntropy(self):
    prediction_tensor = tf.constant([[[_logit(0.97)],
                                      [_logit(0.90)],
                                      [_logit(0.73)],
                                      [_logit(0.27)],
                                      [_logit(0.09)],
                                      [_logit(0.03)]]], tf.float32)
    target_tensor = tf.constant([[[1],
                                  [1],
                                  [1],
                                  [0],
                                  [0],
                                  [0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=True, gamma=2.0, alpha=None)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      order_of_ratio = np.power(10,
                                np.floor(np.log10(sigmoid_loss / focal_loss)))
      self.assertAllClose(order_of_ratio, [[1000, 100, 10, 10, 100, 1000]])

  def testHardExamplesProduceLossComparableToSigmoidXEntropy(self):
    prediction_tensor = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
    target_tensor = tf.constant([[[1],
                                  [1],
                                  [1],
                                  [0],
                                  [0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=True, gamma=2.0, alpha=None)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      order_of_ratio = np.power(10,
                                np.floor(np.log10(sigmoid_loss / focal_loss)))
      self.assertAllClose(order_of_ratio, [[1., 1., 1., 1., 1.]])

  def testNonAnchorWiseOutputComparableToSigmoidXEntropy(self):
    prediction_tensor = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
    target_tensor = tf.constant([[[1],
                                  [1],
                                  [1],
                                  [0],
                                  [0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=False, gamma=2.0, alpha=None)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=False)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      order_of_ratio = np.power(10,
                                np.floor(np.log10(sigmoid_loss / focal_loss)))
      self.assertAlmostEqual(order_of_ratio, 1.)

  def testIgnoreNegativeExampleLossViaAlphaMultiplier(self):
    prediction_tensor = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
    target_tensor = tf.constant([[[1],
                                  [1],
                                  [1],
                                  [0],
                                  [0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=True, gamma=2.0, alpha=1.0)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      self.assertAllClose(focal_loss[0][3:], [0., 0.])
      order_of_ratio = np.power(10,
                                np.floor(np.log10(sigmoid_loss[0][:3] /
                                                  focal_loss[0][:3])))
      self.assertAllClose(order_of_ratio, [1., 1., 1.])

  def testIgnorePositiveExampleLossViaAlphaMultiplier(self):
    prediction_tensor = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
    target_tensor = tf.constant([[[1],
                                  [1],
                                  [1],
                                  [0],
                                  [0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=True, gamma=2.0, alpha=0.0)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      self.assertAllClose(focal_loss[0][:3], [0., 0., 0.])
      order_of_ratio = np.power(10,
                                np.floor(np.log10(sigmoid_loss[0][3:] /
                                                  focal_loss[0][3:])))
      self.assertAllClose(order_of_ratio, [1., 1.])

  def testSimilarToSigmoidXEntropyWithHalfAlphaAndZeroGammaUpToAScale(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=True, alpha=0.5, gamma=0.0)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      self.assertAllClose(sigmoid_loss, focal_loss * 2)

  def testSameAsSigmoidXEntropyWithNoAlphaAndZeroGamma(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=True, alpha=None, gamma=0.0)
    sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                   weights=weights)

    with self.test_session() as sess:
      sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
      self.assertAllClose(sigmoid_loss, focal_loss)

  def testExpectedLossWithAlphaOneAndZeroGamma(self):
    # All zeros correspond to 0.5 probability.
    prediction_tensor = tf.constant([[[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=False, alpha=1.0, gamma=0.0)

    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    with self.test_session() as sess:
      focal_loss = sess.run(focal_loss)
      self.assertAllClose(
          (-math.log(.5) *  # x-entropy per class per anchor
           1.0 *            # alpha
           8),              # positives from 8 anchors
          focal_loss)

  def testExpectedLossWithAlpha75AndZeroGamma(self):
    # All zeros correspond to 0.5 probability.
    prediction_tensor = tf.constant([[[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 1]], tf.float32)
    focal_loss_op = losses.SigmoidFocalClassificationLoss(
        anchorwise_output=False, alpha=0.75, gamma=0.0)

    focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                               weights=weights)
    with self.test_session() as sess:
      focal_loss = sess.run(focal_loss)
      self.assertAllClose(
          (-math.log(.5) *  # x-entropy per class per anchor.
           ((0.75 *         # alpha for positives.
             8) +           # positives from 8 anchors.
            (0.25 *         # alpha for negatives.
             8 * 2))),      # negatives from 8 anchors for two classes.
          focal_loss)


class WeightedSoftmaxClassificationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [0, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 0],
                                      [-100, 100, -100],
                                      [-100, 100, -100],
                                      [100, -100, -100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, .5, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSoftmaxClassificationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = - 1.5 * math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [0, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 0],
                                      [-100, 100, -100],
                                      [-100, 100, -100],
                                      [100, -100, -100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, .5, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSoftmaxClassificationLoss(True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = np.matrix([[0, 0, - 0.5 * math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLossWithHighLogitScaleSetting(self):
    """At very high logit_scale, all predictions will be ~0.33."""
    # TODO(yonib): Also test logit_scale with anchorwise=False.
    logit_scale = 10e16
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [0, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 0],
                                      [-100, 100, -100],
                                      [-100, 100, -100],
                                      [100, -100, -100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 1]], tf.float32)
    loss_op = losses.WeightedSoftmaxClassificationLoss(
        anchorwise_output=True, logit_scale=logit_scale)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    uniform_distribution_loss = - math.log(.33333333333)
    exp_loss = np.matrix([[uniform_distribution_loss] * 4,
                          [uniform_distribution_loss] * 4])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class BootstrappedSigmoidClassificationLossTest(tf.test.TestCase):

  def testReturnsCorrectLossSoftBootstrapping(self):
    prediction_tensor = tf.constant([[[-100, 100, 0],
                                      [100, -100, -100],
                                      [100, -100, -100],
                                      [-100, -100, 100]],
                                     [[-100, -100, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    alpha = tf.constant(.5, tf.float32)
    loss_op = losses.BootstrappedSigmoidClassificationLoss(
        alpha, bootstrap_type='soft')
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)
    exp_loss = -math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossHardBootstrapping(self):
    prediction_tensor = tf.constant([[[-100, 100, 0],
                                      [100, -100, -100],
                                      [100, -100, -100],
                                      [-100, -100, 100]],
                                     [[-100, -100, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    alpha = tf.constant(.5, tf.float32)
    loss_op = losses.BootstrappedSigmoidClassificationLoss(
        alpha, bootstrap_type='hard')
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)
    exp_loss = -math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    alpha = tf.constant(.5, tf.float32)
    loss_op = losses.BootstrappedSigmoidClassificationLoss(
        alpha, bootstrap_type='hard', anchorwise_output=True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class HardExampleMinerTest(tf.test.TestCase):

  def testHardMiningWithSingleLossType(self):
    location_losses = tf.constant([[100, 90, 80, 0],
                                   [0, 1, 2, 3]], tf.float32)
    cls_losses = tf.constant([[0, 10, 50, 110],
                              [9, 6, 3, 0]], tf.float32)
    box_corners = tf.constant([[0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9]], tf.float32)
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    # Uses only location loss to select hard examples
    loss_op = losses.HardExampleMiner(num_hard_examples=1,
                                      iou_threshold=0.0,
                                      loss_type='loc',
                                      cls_loss_weight=1,
                                      loc_loss_weight=1)
    (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                   decoded_boxlist_list)
    exp_loc_loss = 100 + 3
    exp_cls_loss = 0 + 0
    with self.test_session() as sess:
      loc_loss_output = sess.run(loc_loss)
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      cls_loss_output = sess.run(cls_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testHardMiningWithBothLossType(self):
    location_losses = tf.constant([[100, 90, 80, 0],
                                   [0, 1, 2, 3]], tf.float32)
    cls_losses = tf.constant([[0, 10, 50, 110],
                              [9, 6, 3, 0]], tf.float32)
    box_corners = tf.constant([[0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9]], tf.float32)
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    loss_op = losses.HardExampleMiner(num_hard_examples=1,
                                      iou_threshold=0.0,
                                      loss_type='both',
                                      cls_loss_weight=1,
                                      loc_loss_weight=1)
    (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                   decoded_boxlist_list)
    exp_loc_loss = 80 + 0
    exp_cls_loss = 50 + 9
    with self.test_session() as sess:
      loc_loss_output = sess.run(loc_loss)
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      cls_loss_output = sess.run(cls_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testHardMiningNMS(self):
    location_losses = tf.constant([[100, 90, 80, 0],
                                   [0, 1, 2, 3]], tf.float32)
    cls_losses = tf.constant([[0, 10, 50, 110],
                              [9, 6, 3, 0]], tf.float32)
    box_corners = tf.constant([[0.1, 0.1, 0.9, 0.9],
                               [0.9, 0.9, 0.99, 0.99],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9]], tf.float32)
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    loss_op = losses.HardExampleMiner(num_hard_examples=2,
                                      iou_threshold=0.5,
                                      loss_type='cls',
                                      cls_loss_weight=1,
                                      loc_loss_weight=1)
    (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                   decoded_boxlist_list)
    exp_loc_loss = 0 + 90 + 0 + 1
    exp_cls_loss = 110 + 10 + 9 + 6
    with self.test_session() as sess:
      loc_loss_output = sess.run(loc_loss)
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      cls_loss_output = sess.run(cls_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testEnforceNegativesPerPositiveRatio(self):
    location_losses = tf.constant([[100, 90, 80, 0, 1, 2,
                                    3, 10, 20, 100, 20, 3]], tf.float32)
    cls_losses = tf.constant([[0, 0, 100, 0, 90, 70,
                               0, 60, 0, 17, 13, 0]], tf.float32)
    box_corners = tf.constant([[0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.5, 0.1],
                               [0.0, 0.0, 0.6, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.8, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 1.0, 0.1],
                               [0.0, 0.0, 1.1, 0.1],
                               [0.0, 0.0, 0.2, 0.1]], tf.float32)
    match_results = tf.constant([2, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 3])
    match_list = [matcher.Match(match_results)]
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))

    max_negatives_per_positive_list = [0.0, 0.5, 1.0, 1.5, 10]
    exp_loc_loss_list = [80 + 2,
                         80 + 1 + 2,
                         80 + 1 + 2 + 10,
                         80 + 1 + 2 + 10 + 100,
                         80 + 1 + 2 + 10 + 100 + 20]
    exp_cls_loss_list = [100 + 70,
                         100 + 90 + 70,
                         100 + 90 + 70 + 60,
                         100 + 90 + 70 + 60 + 17,
                         100 + 90 + 70 + 60 + 17 + 13]

    for max_negatives_per_positive, exp_loc_loss, exp_cls_loss in zip(
        max_negatives_per_positive_list, exp_loc_loss_list, exp_cls_loss_list):
      loss_op = losses.HardExampleMiner(
          num_hard_examples=None, iou_threshold=0.9999, loss_type='cls',
          cls_loss_weight=1, loc_loss_weight=1,
          max_negatives_per_positive=max_negatives_per_positive)
      (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                     decoded_boxlist_list, match_list)
      loss_op.summarize()

      with self.test_session() as sess:
        loc_loss_output = sess.run(loc_loss)
        self.assertAllClose(loc_loss_output, exp_loc_loss)
        cls_loss_output = sess.run(cls_loss)
        self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testEnforceNegativesPerPositiveRatioWithMinNegativesPerImage(self):
    location_losses = tf.constant([[100, 90, 80, 0, 1, 2,
                                    3, 10, 20, 100, 20, 3]], tf.float32)
    cls_losses = tf.constant([[0, 0, 100, 0, 90, 70,
                               0, 60, 0, 17, 13, 0]], tf.float32)
    box_corners = tf.constant([[0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.5, 0.1],
                               [0.0, 0.0, 0.6, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.8, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 1.0, 0.1],
                               [0.0, 0.0, 1.1, 0.1],
                               [0.0, 0.0, 0.2, 0.1]], tf.float32)
    match_results = tf.constant([-1] * 12)
    match_list = [matcher.Match(match_results)]
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))

    min_negatives_per_image_list = [0, 1, 2, 4, 5, 6]
    exp_loc_loss_list = [0,
                         80,
                         80 + 1,
                         80 + 1 + 2 + 10,
                         80 + 1 + 2 + 10 + 100,
                         80 + 1 + 2 + 10 + 100 + 20]
    exp_cls_loss_list = [0,
                         100,
                         100 + 90,
                         100 + 90 + 70 + 60,
                         100 + 90 + 70 + 60 + 17,
                         100 + 90 + 70 + 60 + 17 + 13]

    for min_negatives_per_image, exp_loc_loss, exp_cls_loss in zip(
        min_negatives_per_image_list, exp_loc_loss_list, exp_cls_loss_list):
      loss_op = losses.HardExampleMiner(
          num_hard_examples=None, iou_threshold=0.9999, loss_type='cls',
          cls_loss_weight=1, loc_loss_weight=1,
          max_negatives_per_positive=3,
          min_negatives_per_image=min_negatives_per_image)
      (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                     decoded_boxlist_list, match_list)
      with self.test_session() as sess:
        loc_loss_output = sess.run(loc_loss)
        self.assertAllClose(loc_loss_output, exp_loc_loss)
        cls_loss_output = sess.run(cls_loss)
        self.assertAllClose(cls_loss_output, exp_cls_loss)


if __name__ == '__main__':
  tf.test.main()
