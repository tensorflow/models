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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import matcher
from object_detection.utils import test_case


class WeightedL2LocalizationLossTest(test_case.TestCase):

  def testReturnsCorrectWeightedLoss(self):
    batch_size = 3
    num_anchors = 10
    code_size = 4
    def graph_fn():
      prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
      target_tensor = tf.zeros([batch_size, num_anchors, code_size])
      weights = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], tf.float32)
      loss_op = losses.WeightedL2LocalizationLoss()
      loss = tf.reduce_sum(loss_op(prediction_tensor, target_tensor,
                                   weights=weights))
      return loss
    expected_loss = (3 * 5 * 4) / 2.0
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectAnchorwiseLoss(self):
    batch_size = 3
    num_anchors = 16
    code_size = 4
    def graph_fn():
      prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
      target_tensor = tf.zeros([batch_size, num_anchors, code_size])
      weights = tf.ones([batch_size, num_anchors])
      loss_op = losses.WeightedL2LocalizationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      return loss
    expected_loss = np.ones((batch_size, num_anchors)) * 2
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectNanLoss(self):
    batch_size = 3
    num_anchors = 10
    code_size = 4
    def graph_fn():
      prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
      target_tensor = tf.concat([
          tf.zeros([batch_size, num_anchors, code_size / 2]),
          tf.ones([batch_size, num_anchors, code_size / 2]) * np.nan
      ], axis=2)
      weights = tf.ones([batch_size, num_anchors])
      loss_op = losses.WeightedL2LocalizationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                     ignore_nan_targets=True)
      loss = tf.reduce_sum(loss)
      return loss
    expected_loss = (3 * 5 * 4) / 2.0
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectWeightedLossWithLossesMask(self):
    batch_size = 4
    num_anchors = 10
    code_size = 4
    def graph_fn():
      prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
      target_tensor = tf.zeros([batch_size, num_anchors, code_size])
      weights = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], tf.float32)
      losses_mask = tf.constant([True, False, True, True], tf.bool)
      loss_op = losses.WeightedL2LocalizationLoss()
      loss = tf.reduce_sum(loss_op(prediction_tensor, target_tensor,
                                   weights=weights, losses_mask=losses_mask))
      return loss
    expected_loss = (3 * 5 * 4) / 2.0
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, expected_loss)


class WeightedSmoothL1LocalizationLossTest(test_case.TestCase):

  def testReturnsCorrectLoss(self):
    batch_size = 2
    num_anchors = 3
    code_size = 4
    def graph_fn():
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
      loss = tf.reduce_sum(loss)
      return loss
    exp_loss = 7.695
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithLossesMask(self):
    batch_size = 3
    num_anchors = 3
    code_size = 4
    def graph_fn():
      prediction_tensor = tf.constant([[[2.5, 0, .4, 0],
                                        [0, 0, 0, 0],
                                        [0, 2.5, 0, .4]],
                                       [[3.5, 0, 0, 0],
                                        [0, .4, 0, .9],
                                        [0, 0, 1.5, 0]],
                                       [[3.5, 7., 0, 0],
                                        [0, .4, 0, .9],
                                        [2.2, 2.2, 1.5, 0]]], tf.float32)
      target_tensor = tf.zeros([batch_size, num_anchors, code_size])
      weights = tf.constant([[2, 1, 1],
                             [0, 3, 0],
                             [4, 3, 0]], tf.float32)
      losses_mask = tf.constant([True, True, False], tf.bool)
      loss_op = losses.WeightedSmoothL1LocalizationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                     losses_mask=losses_mask)
      loss = tf.reduce_sum(loss)
      return loss
    exp_loss = 7.695
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)


class WeightedIOULocalizationLossTest(test_case.TestCase):

  def testReturnsCorrectLoss(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                        [0, 0, 1, 1],
                                        [0, 0, .5, .25]]])
      target_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                    [0, 0, 1, 1],
                                    [50, 50, 500.5, 100.25]]])
      weights = [[1.0, .5, 2.0]]
      loss_op = losses.WeightedIOULocalizationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss)
      return loss
    exp_loss = 2.0
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithNoLabels(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                        [0, 0, 1, 1],
                                        [0, 0, .5, .25]]])
      target_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                    [0, 0, 1, 1],
                                    [50, 50, 500.5, 100.25]]])
      weights = [[1.0, .5, 2.0]]
      losses_mask = tf.constant([False], tf.bool)
      loss_op = losses.WeightedIOULocalizationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                     losses_mask=losses_mask)
      loss = tf.reduce_sum(loss)
      return loss
    exp_loss = 0.0
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)


class WeightedGIOULocalizationLossTest(test_case.TestCase):

  def testReturnsCorrectLoss(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                        [0, 0, 1, 1],
                                        [0, 0, 0, 0]]])
      target_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                    [0, 0, 1, 1],
                                    [5, 5, 10, 10]]])
      weights = [[1.0, .5, 2.0]]
      loss_op = losses.WeightedGIOULocalizationLoss()
      loss = loss_op(prediction_tensor,
                     target_tensor,
                     weights=weights)
      loss = tf.reduce_sum(loss)
      return loss
    exp_loss = 3.5
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithNoLabels(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                        [0, 0, 1, 1],
                                        [0, 0, .5, .25]]])
      target_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                    [0, 0, 1, 1],
                                    [50, 50, 500.5, 100.25]]])
      weights = [[1.0, .5, 2.0]]
      losses_mask = tf.constant([False], tf.bool)
      loss_op = losses.WeightedGIOULocalizationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                     losses_mask=losses_mask)
      loss = tf.reduce_sum(loss)
      return loss
    exp_loss = 0.0
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)


class WeightedSigmoidClassificationLossTest(test_case.TestCase):

  def testReturnsCorrectLoss(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      loss_op = losses.WeightedSigmoidClassificationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss)
      return loss

    exp_loss = -2 * math.log(.5)
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      loss_op = losses.WeightedSigmoidClassificationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss, axis=2)
      return loss

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithClassIndices(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]],
                             [[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [0, 0, 0, 0]]], tf.float32)
      # Ignores the last class.
      class_indices = tf.constant([0, 1, 2], tf.int32)
      loss_op = losses.WeightedSigmoidClassificationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                     class_indices=class_indices)
      loss = tf.reduce_sum(loss, axis=2)
      return loss

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithLossesMask(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[-100, 100, -100],
                                        [100, -100, -100],
                                        [100, 0, -100],
                                        [-100, -100, 100]],
                                       [[-100, 0, 100],
                                        [-100, 100, -100],
                                        [100, 100, 100],
                                        [0, 0, -1]],
                                       [[-100, 0, 100],
                                        [-100, 100, -100],
                                        [100, 100, 100],
                                        [0, 0, -100]]], tf.float32)
      target_tensor = tf.constant([[[0, 1, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]],
                                   [[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 1, 1],
                                    [1, 0, 0]],
                                   [[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]]], tf.float32)
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], tf.float32)
      losses_mask = tf.constant([True, True, False], tf.bool)

      loss_op = losses.WeightedSigmoidClassificationLoss()
      loss_per_anchor = loss_op(prediction_tensor, target_tensor,
                                weights=weights,
                                losses_mask=losses_mask)
      loss = tf.reduce_sum(loss_per_anchor)
      return loss

    exp_loss = -2 * math.log(.5)
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)


def _logit(probability):
  return math.log(probability / (1. - probability))


class SigmoidFocalClassificationLossTest(test_case.TestCase):

  def testEasyExamplesProduceSmallLossComparedToSigmoidXEntropy(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[_logit(0.97)],
                                        [_logit(0.91)],
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
      weights = tf.constant([[[1], [1], [1], [1], [1], [1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0,
                                                            alpha=None)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights), axis=2)
      sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(prediction_tensor,
                                                   target_tensor,
                                                   weights=weights), axis=2)
      return sigmoid_loss, focal_loss

    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    order_of_ratio = np.power(10,
                              np.floor(np.log10(sigmoid_loss / focal_loss)))
    self.assertAllClose(order_of_ratio, [[1000, 100, 10, 10, 100, 1000]])

  def testHardExamplesProduceLossComparableToSigmoidXEntropy(self):
    def graph_fn():
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
      weights = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0,
                                                            alpha=None)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights), axis=2)
      sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(prediction_tensor,
                                                   target_tensor,
                                                   weights=weights), axis=2)
      return sigmoid_loss, focal_loss
    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    order_of_ratio = np.power(10,
                              np.floor(np.log10(sigmoid_loss / focal_loss)))
    self.assertAllClose(order_of_ratio, [[1., 1., 1., 1., 1.]])

  def testNonAnchorWiseOutputComparableToSigmoidXEntropy(self):
    def graph_fn():
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
      weights = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0,
                                                            alpha=None)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights))
      sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(prediction_tensor,
                                                   target_tensor,
                                                   weights=weights))
      return sigmoid_loss, focal_loss
    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    order_of_ratio = np.power(10,
                              np.floor(np.log10(sigmoid_loss / focal_loss)))
    self.assertAlmostEqual(order_of_ratio, 1.)

  def testIgnoreNegativeExampleLossViaAlphaMultiplier(self):
    def graph_fn():
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
      weights = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0,
                                                            alpha=1.0)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights), axis=2)
      sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(prediction_tensor,
                                                   target_tensor,
                                                   weights=weights), axis=2)
      return sigmoid_loss, focal_loss

    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(focal_loss[0][3:], [0., 0.])
    order_of_ratio = np.power(10,
                              np.floor(np.log10(sigmoid_loss[0][:3] /
                                                focal_loss[0][:3])))
    self.assertAllClose(order_of_ratio, [1., 1., 1.])

  def testIgnorePositiveExampleLossViaAlphaMultiplier(self):
    def graph_fn():
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
      weights = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0,
                                                            alpha=0.0)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights), axis=2)
      sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(prediction_tensor,
                                                   target_tensor,
                                                   weights=weights), axis=2)
      return sigmoid_loss, focal_loss
    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(focal_loss[0][:3], [0., 0., 0.])
    order_of_ratio = np.power(10,
                              np.floor(np.log10(sigmoid_loss[0][3:] /
                                                focal_loss[0][3:])))
    self.assertAllClose(order_of_ratio, [1., 1.])

  def testSimilarToSigmoidXEntropyWithHalfAlphaAndZeroGammaUpToAScale(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=0.5,
                                                            gamma=0.0)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                                 weights=weights)
      sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                     weights=weights)
      return sigmoid_loss, focal_loss
    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(sigmoid_loss, focal_loss * 2)

  def testSameAsSigmoidXEntropyWithNoAlphaAndZeroGamma(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=None,
                                                            gamma=0.0)
      sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
      focal_loss = focal_loss_op(prediction_tensor, target_tensor,
                                 weights=weights)
      sigmoid_loss = sigmoid_loss_op(prediction_tensor, target_tensor,
                                     weights=weights)
      return sigmoid_loss, focal_loss
    sigmoid_loss, focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(sigmoid_loss, focal_loss)

  def testExpectedLossWithAlphaOneAndZeroGamma(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=1.0,
                                                            gamma=0.0)

      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights))
      return focal_loss
    focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(
        (-math.log(.5) *  # x-entropy per class per anchor
         1.0 *            # alpha
         8),              # positives from 8 anchors
        focal_loss)

  def testExpectedLossWithAlpha75AndZeroGamma(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], tf.float32)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=0.75,
                                                            gamma=0.0)

      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights))
      return focal_loss
    focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(
        (-math.log(.5) *  # x-entropy per class per anchor.
         ((0.75 *         # alpha for positives.
           8) +           # positives from 8 anchors.
          (0.25 *         # alpha for negatives.
           8 * 2))),      # negatives from 8 anchors for two classes.
        focal_loss)

  def testExpectedLossWithLossesMask(self):
    def graph_fn():
      # All zeros correspond to 0.5 probability.
      prediction_tensor = tf.constant([[[0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]],
                                       [[0, 0, 0],
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
                                    [1, 0, 0]],
                                   [[1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0]]], tf.float32)
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], tf.float32)
      losses_mask = tf.constant([True, True, False], tf.bool)
      focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=0.75,
                                                            gamma=0.0)

      focal_loss = tf.reduce_sum(focal_loss_op(prediction_tensor, target_tensor,
                                               weights=weights,
                                               losses_mask=losses_mask))
      return focal_loss
    focal_loss = self.execute(graph_fn, [])
    self.assertAllClose(
        (-math.log(.5) *  # x-entropy per class per anchor.
         ((0.75 *         # alpha for positives.
           8) +           # positives from 8 anchors.
          (0.25 *         # alpha for negatives.
           8 * 2))),      # negatives from 8 anchors for two classes.
        focal_loss)


class WeightedSoftmaxClassificationLossTest(test_case.TestCase):

  def testReturnsCorrectLoss(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [0.5, 0.5, 0.5],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      loss_op = losses.WeightedSoftmaxClassificationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = - 1.5 * math.log(.5)
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [0.5, 0.5, 0.5],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      loss_op = losses.WeightedSoftmaxClassificationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = np.matrix([[0, 0, - 0.5 * math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLossWithHighLogitScaleSetting(self):
    """At very high logit_scale, all predictions will be ~0.33."""
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], tf.float32)
      loss_op = losses.WeightedSoftmaxClassificationLoss(
          logit_scale=logit_scale)
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      return loss
    uniform_distribution_loss = - math.log(.33333333333)
    exp_loss = np.matrix([[uniform_distribution_loss] * 4,
                          [uniform_distribution_loss] * 4])
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithLossesMask(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[-100, 100, -100],
                                        [100, -100, -100],
                                        [0, 0, -100],
                                        [-100, -100, 100]],
                                       [[-100, 0, 0],
                                        [-100, 100, -100],
                                        [-100, 100, -100],
                                        [100, -100, -100]],
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
                                    [1, 0, 0]],
                                   [[1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0]]], tf.float32)
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [0.5, 0.5, 0.5],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]], tf.float32)
      losses_mask = tf.constant([True, True, False], tf.bool)
      loss_op = losses.WeightedSoftmaxClassificationLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                     losses_mask=losses_mask)
      loss = tf.reduce_sum(loss)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = - 1.5 * math.log(.5)
    self.assertAllClose(loss_output, exp_loss)


class WeightedSoftmaxClassificationAgainstLogitsLossTest(test_case.TestCase):

  def testReturnsCorrectLoss(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[-100, 100, -100],
                                        [100, -100, -100],
                                        [0, 0, -100],
                                        [-100, -100, 100]],
                                       [[-100, 0, 0],
                                        [-100, 100, -100],
                                        [-100, 100, -100],
                                        [100, -100, -100]]], tf.float32)

      target_tensor = tf.constant([[[-100, 100, -100],
                                    [100, -100, -100],
                                    [100, -100, -100],
                                    [-100, -100, 100]],
                                   [[-100, -100, 100],
                                    [-100, 100, -100],
                                    [-100, 100, -100],
                                    [100, -100, -100]]], tf.float32)
      weights = tf.constant([[1, 1, .5, 1],
                             [1, 1, 1, 1]], tf.float32)
      weights_shape = tf.shape(weights)
      weights_multiple = tf.concat(
          [tf.ones_like(weights_shape), tf.constant([3])],
          axis=0)
      weights = tf.tile(tf.expand_dims(weights, 2), weights_multiple)
      loss_op = losses.WeightedSoftmaxClassificationAgainstLogitsLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = - 1.5 * math.log(.5)
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    def graph_fn():
      prediction_tensor = tf.constant([[[-100, 100, -100],
                                        [100, -100, -100],
                                        [0, 0, -100],
                                        [-100, -100, 100]],
                                       [[-100, 0, 0],
                                        [-100, 100, -100],
                                        [-100, 100, -100],
                                        [100, -100, -100]]], tf.float32)
      target_tensor = tf.constant([[[-100, 100, -100],
                                    [100, -100, -100],
                                    [100, -100, -100],
                                    [-100, -100, 100]],
                                   [[-100, -100, 100],
                                    [-100, 100, -100],
                                    [-100, 100, -100],
                                    [100, -100, -100]]], tf.float32)
      weights = tf.constant([[1, 1, .5, 1],
                             [1, 1, 1, 0]], tf.float32)
      weights_shape = tf.shape(weights)
      weights_multiple = tf.concat(
          [tf.ones_like(weights_shape), tf.constant([3])],
          axis=0)
      weights = tf.tile(tf.expand_dims(weights, 2), weights_multiple)
      loss_op = losses.WeightedSoftmaxClassificationAgainstLogitsLoss()
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = np.matrix([[0, 0, - 0.5 * math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLossWithLogitScaleSetting(self):
    def graph_fn():
      logit_scale = 100.
      prediction_tensor = tf.constant([[[-100, 100, -100],
                                        [100, -100, -100],
                                        [0, 0, -100],
                                        [-100, -100, 100]],
                                       [[-100, 0, 0],
                                        [-100, 100, -100],
                                        [-100, 100, -100],
                                        [100, -100, -100]]], tf.float32)
      target_tensor = tf.constant([[[-100, 100, -100],
                                    [100, -100, -100],
                                    [0, 0, -100],
                                    [-100, -100, 100]],
                                   [[-100, 0, 0],
                                    [-100, 100, -100],
                                    [-100, 100, -100],
                                    [100, -100, -100]]], tf.float32)
      weights = tf.constant([[1, 1, .5, 1],
                             [1, 1, 1, 0]], tf.float32)
      weights_shape = tf.shape(weights)
      weights_multiple = tf.concat(
          [tf.ones_like(weights_shape), tf.constant([3])],
          axis=0)
      weights = tf.tile(tf.expand_dims(weights, 2), weights_multiple)
      loss_op = losses.WeightedSoftmaxClassificationAgainstLogitsLoss(
          logit_scale=logit_scale)
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      return loss

    # find softmax of the two prediction types above
    softmax_pred1 = [np.exp(-1), np.exp(-1), np.exp(1)]
    softmax_pred1 /= sum(softmax_pred1)
    softmax_pred2 = [np.exp(0), np.exp(0), np.exp(-1)]
    softmax_pred2 /= sum(softmax_pred2)

    # compute the expected cross entropy for perfect matches
    exp_entropy1 = sum(
        [-x*np.log(x) for x in softmax_pred1])
    exp_entropy2 = sum(
        [-x*np.log(x) for x in softmax_pred2])

    # weighted expected losses
    exp_loss = np.matrix(
        [[exp_entropy1, exp_entropy1, exp_entropy2*.5, exp_entropy1],
         [exp_entropy2, exp_entropy1, exp_entropy1, 0.]])
    loss_output = self.execute(graph_fn, [])
    self.assertAllClose(loss_output, exp_loss)


class BootstrappedSigmoidClassificationLossTest(test_case.TestCase):

  def testReturnsCorrectLossSoftBootstrapping(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      alpha = tf.constant(.5, tf.float32)
      loss_op = losses.BootstrappedSigmoidClassificationLoss(
          alpha, bootstrap_type='soft')
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = -math.log(.5)
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossHardBootstrapping(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      alpha = tf.constant(.5, tf.float32)
      loss_op = losses.BootstrappedSigmoidClassificationLoss(
          alpha, bootstrap_type='hard')
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = -math.log(.5)
    self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    def graph_fn():
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
      weights = tf.constant([[[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]],
                             [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [0, 0, 0]]], tf.float32)
      alpha = tf.constant(.5, tf.float32)
      loss_op = losses.BootstrappedSigmoidClassificationLoss(
          alpha, bootstrap_type='hard')
      loss = loss_op(prediction_tensor, target_tensor, weights=weights)
      loss = tf.reduce_sum(loss, axis=2)
      return loss
    loss_output = self.execute(graph_fn, [])
    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    self.assertAllClose(loss_output, exp_loss)


class HardExampleMinerTest(test_case.TestCase):

  def testHardMiningWithSingleLossType(self):
    def graph_fn():
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
      return loc_loss, cls_loss
    loc_loss_output, cls_loss_output = self.execute(graph_fn, [])
    exp_loc_loss = 100 + 3
    exp_cls_loss = 0 + 0
    self.assertAllClose(loc_loss_output, exp_loc_loss)
    self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testHardMiningWithBothLossType(self):
    def graph_fn():
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
      return loc_loss, cls_loss
    loc_loss_output, cls_loss_output = self.execute(graph_fn, [])
    exp_loc_loss = 80 + 0
    exp_cls_loss = 50 + 9
    self.assertAllClose(loc_loss_output, exp_loc_loss)
    self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testHardMiningNMS(self):
    def graph_fn():
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
      return loc_loss, cls_loss
    loc_loss_output, cls_loss_output = self.execute(graph_fn, [])
    exp_loc_loss = 0 + 90 + 0 + 1
    exp_cls_loss = 110 + 10 + 9 + 6

    self.assertAllClose(loc_loss_output, exp_loc_loss)
    self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testEnforceNegativesPerPositiveRatio(self):
    location_losses = np.array([[100, 90, 80, 0, 1, 2,
                                 3, 10, 20, 100, 20, 3]], np.float32)
    cls_losses = np.array([[0, 0, 100, 0, 90, 70,
                            0, 60, 0, 17, 13, 0]], np.float32)
    box_corners = np.array([[0.0, 0.0, 0.2, 0.1],
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
                            [0.0, 0.0, 0.2, 0.1]], np.float32)
    match_results = np.array([2, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 3],
                             np.int32)

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

    # pylint: disable=cell-var-from-loop
    for max_negatives_per_positive, exp_loc_loss, exp_cls_loss in zip(
        max_negatives_per_positive_list, exp_loc_loss_list, exp_cls_loss_list):
      def graph_fn():
        loss_op = losses.HardExampleMiner(
            num_hard_examples=None, iou_threshold=0.9999, loss_type='cls',
            cls_loss_weight=1, loc_loss_weight=1,
            max_negatives_per_positive=max_negatives_per_positive)
        match_list = [matcher.Match(tf.constant(match_results))]
        decoded_boxlist_list = [box_list.BoxList(tf.constant(box_corners))]
        (loc_loss, cls_loss) = loss_op(tf.constant(location_losses),
                                       tf.constant(cls_losses),
                                       decoded_boxlist_list, match_list)
        return loc_loss, cls_loss
      loc_loss_output, cls_loss_output = self.execute_cpu(graph_fn, [])
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)
    # pylint: enable=cell-var-from-loop

  def testEnforceNegativesPerPositiveRatioWithMinNegativesPerImage(self):
    location_losses = np.array([[100, 90, 80, 0, 1, 2,
                                 3, 10, 20, 100, 20, 3]], np.float32)
    cls_losses = np.array([[0, 0, 100, 0, 90, 70,
                            0, 60, 0, 17, 13, 0]], np.float32)
    box_corners = np.array([[0.0, 0.0, 0.2, 0.1],
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
                            [0.0, 0.0, 0.2, 0.1]], np.float32)
    match_results = np.array([-1] * 12, np.int32)

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

    # pylint: disable=cell-var-from-loop
    for min_negatives_per_image, exp_loc_loss, exp_cls_loss in zip(
        min_negatives_per_image_list, exp_loc_loss_list, exp_cls_loss_list):
      def graph_fn():
        loss_op = losses.HardExampleMiner(
            num_hard_examples=None, iou_threshold=0.9999, loss_type='cls',
            cls_loss_weight=1, loc_loss_weight=1,
            max_negatives_per_positive=3,
            min_negatives_per_image=min_negatives_per_image)
        match_list = [matcher.Match(tf.constant(match_results))]
        decoded_boxlist_list = [box_list.BoxList(tf.constant(box_corners))]
        (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                       decoded_boxlist_list, match_list)
        return loc_loss, cls_loss
      loc_loss_output, cls_loss_output = self.execute_cpu(graph_fn, [])
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)
    # pylint: enable=cell-var-from-loop


LOG_2 = np.log(2)
LOG_3 = np.log(3)


class PenaltyReducedLogisticFocalLossTest(test_case.TestCase):
  """Testing loss function from Equation (1) in [1].

  [1]: https://arxiv.org/abs/1904.07850
  """

  def setUp(self):
    super(PenaltyReducedLogisticFocalLossTest, self).setUp()
    self._prediction = np.array([
        # First batch
        [[1 / 2, 1 / 4, 3 / 4],
         [3 / 4, 1 / 3, 1 / 3]],
        # Second Batch
        [[0.0, 1.0, 1 / 2],
         [3 / 4, 2 / 3, 1 / 3]]], np.float32)
    self._prediction = np.log(self._prediction/(1 - self._prediction))

    self._target = np.array([
        # First batch
        [[1.0, 0.91, 1.0],
         [0.36, 0.84, 1.0]],
        # Second Batch
        [[0.01, 1.0, 0.75],
         [0.96, 1.0, 1.0]]], np.float32)

  def test_returns_correct_loss(self):
    def graph_fn(prediction, target):
      weights = tf.constant([
          [[1.0], [1.0]],
          [[1.0], [1.0]],
      ])
      loss = losses.PenaltyReducedLogisticFocalLoss(alpha=2.0, beta=0.5)
      computed_value = loss._compute_loss(prediction, target,
                                          weights)
      return computed_value
    computed_value = self.execute(graph_fn, [self._prediction, self._target])
    expected_value = np.array([
        # First batch
        [[1 / 4 * LOG_2,
          0.3 * 0.0625 * (2 * LOG_2 - LOG_3),
          1 / 16 * (2 * LOG_2 - LOG_3)],
         [0.8 * 9 / 16 * 2 * LOG_2,
          0.4 * 1 / 9 * (LOG_3 - LOG_2),
          4 / 9 * LOG_3]],
        # Second Batch
        [[0.0,
          0.0,
          1 / 2 * 1 / 4 * LOG_2],
         [0.2 * 9 / 16 * 2 * LOG_2,
          1 / 9 * (LOG_3 - LOG_2),
          4 / 9 * LOG_3]]])
    self.assertAllClose(computed_value, expected_value, rtol=1e-3, atol=1e-3)

  def test_returns_correct_loss_weighted(self):
    def graph_fn(prediction, target):
      weights = tf.constant([
          [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
          [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
      ])

      loss = losses.PenaltyReducedLogisticFocalLoss(alpha=2.0, beta=0.5)

      computed_value = loss._compute_loss(prediction, target,
                                          weights)
      return computed_value
    computed_value = self.execute(graph_fn, [self._prediction, self._target])
    expected_value = np.array([
        # First batch
        [[1 / 4 * LOG_2,
          0.0,
          1 / 16 * (2 * LOG_2 - LOG_3)],
         [0.0,
          0.0,
          4 / 9 * LOG_3]],
        # Second Batch
        [[0.0,
          0.0,
          1 / 2 * 1 / 4 * LOG_2],
         [0.0,
          0.0,
          0.0]]])

    self.assertAllClose(computed_value, expected_value, rtol=1e-3, atol=1e-3)


class L1LocalizationLossTest(test_case.TestCase):

  def test_returns_correct_loss(self):
    def graph_fn():
      loss = losses.L1LocalizationLoss()
      pred = [[0.1, 0.2], [0.7, 0.5]]
      target = [[0.9, 1.0], [0.1, 0.4]]

      weights = [[1.0, 0.0], [1.0, 1.0]]
      return loss._compute_loss(pred, target, weights)
    computed_value = self.execute(graph_fn, [])
    self.assertAllClose(computed_value, [[0.8, 0.0], [0.6, 0.1]], rtol=1e-6)


class WeightedDiceClassificationLoss(test_case.TestCase):

  def test_compute_weights_1(self):
    def graph_fn():
      loss = losses.WeightedDiceClassificationLoss(squared_normalization=False)
      pred = np.zeros((2, 3, 4), dtype=np.float32)
      target = np.zeros((2, 3, 4), dtype=np.float32)

      pred[0, 1, 0] = _logit(0.9)
      pred[0, 2, 0] = _logit(0.1)
      pred[0, 2, 2] = _logit(0.5)
      pred[0, 1, 3] = _logit(0.1)

      pred[1, 2, 3] = _logit(0.2)
      pred[1, 1, 1] = _logit(0.3)
      pred[1, 0, 2] = _logit(0.1)

      target[0, 1, 0] = 1.0
      target[0, 2, 2] = 1.0
      target[0, 1, 3] = 1.0

      target[1, 2, 3] = 1.0
      target[1, 1, 1] = 0.0
      target[1, 0, 2] = 0.0

      weights = np.ones_like(target)
      return loss._compute_loss(pred, target, weights)

    dice_coeff = np.zeros((2, 4))
    dice_coeff[0, 0] = 2 * 0.9 / 2.5
    dice_coeff[0, 2] = 2 * 0.5 / 2.5
    dice_coeff[0, 3] = 2 * 0.1 / 2.1
    dice_coeff[1, 3] = 2 * 0.2 / 2.2

    computed_value = self.execute(graph_fn, [])
    self.assertAllClose(computed_value, 1 - dice_coeff, rtol=1e-6)

  def test_compute_weights_set(self):

    def graph_fn():
      loss = losses.WeightedDiceClassificationLoss(squared_normalization=False)
      pred = np.zeros((2, 3, 4), dtype=np.float32)
      target = np.zeros((2, 3, 4), dtype=np.float32)

      pred[0, 1, 0] = _logit(0.9)
      pred[0, 2, 0] = _logit(0.1)
      pred[0, 2, 2] = _logit(0.5)
      pred[0, 1, 3] = _logit(0.1)

      pred[1, 2, 3] = _logit(0.2)
      pred[1, 1, 1] = _logit(0.3)
      pred[1, 0, 2] = _logit(0.1)

      target[0, 1, 0] = 1.0
      target[0, 2, 2] = 1.0
      target[0, 1, 3] = 1.0

      target[1, 2, 3] = 1.0
      target[1, 1, 1] = 0.0
      target[1, 0, 2] = 0.0

      weights = np.ones_like(target)
      weights[:, :, 0] = 0.0
      return loss._compute_loss(pred, target, weights)

    dice_coeff = np.zeros((2, 4))
    dice_coeff[0, 2] = 2 * 0.5 / 2.5
    dice_coeff[0, 3] = 2 * 0.1 / 2.1
    dice_coeff[1, 3] = 2 * 0.2 / 2.2

    computed_value = self.execute(graph_fn, [])
    self.assertAllClose(computed_value, 1 - dice_coeff, rtol=1e-6)

  def test_class_indices(self):
    def graph_fn():
      loss = losses.WeightedDiceClassificationLoss(squared_normalization=False)
      pred = np.zeros((2, 3, 4), dtype=np.float32)
      target = np.zeros((2, 3, 4), dtype=np.float32)

      pred[0, 1, 0] = _logit(0.9)
      pred[0, 2, 0] = _logit(0.1)
      pred[0, 2, 2] = _logit(0.5)
      pred[0, 1, 3] = _logit(0.1)

      pred[1, 2, 3] = _logit(0.2)
      pred[1, 1, 1] = _logit(0.3)
      pred[1, 0, 2] = _logit(0.1)

      target[0, 1, 0] = 1.0
      target[0, 2, 2] = 1.0
      target[0, 1, 3] = 1.0

      target[1, 2, 3] = 1.0
      target[1, 1, 1] = 0.0
      target[1, 0, 2] = 0.0

      weights = np.ones_like(target)
      return loss._compute_loss(pred, target, weights, class_indices=[0])

    dice_coeff = np.zeros((2, 4))
    dice_coeff[0, 0] = 2 * 0.9 / 2.5

    computed_value = self.execute(graph_fn, [])
    self.assertAllClose(computed_value, 1 - dice_coeff, rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
