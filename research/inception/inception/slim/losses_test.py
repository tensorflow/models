# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception.slim import losses


class LossesTest(tf.test.TestCase):

  def testL1Loss(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      weights = tf.constant(1.0, shape=shape)
      wd = 0.01
      loss = losses.l1_loss(weights, wd)
      self.assertEquals(loss.op.name, 'L1Loss/value')
      self.assertAlmostEqual(loss.eval(), num_elem * wd, 5)

  def testL2Loss(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      weights = tf.constant(1.0, shape=shape)
      wd = 0.01
      loss = losses.l2_loss(weights, wd)
      self.assertEquals(loss.op.name, 'L2Loss/value')
      self.assertAlmostEqual(loss.eval(), num_elem * wd / 2, 5)


class RegularizersTest(tf.test.TestCase):

  def testL1Regularizer(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      loss = losses.l1_regularizer()(tensor)
      self.assertEquals(loss.op.name, 'L1Regularizer/value')
      self.assertAlmostEqual(loss.eval(), num_elem, 5)

  def testL1RegularizerWithScope(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      loss = losses.l1_regularizer(scope='L1')(tensor)
      self.assertEquals(loss.op.name, 'L1/value')
      self.assertAlmostEqual(loss.eval(), num_elem, 5)

  def testL1RegularizerWithWeight(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      weight = 0.01
      loss = losses.l1_regularizer(weight)(tensor)
      self.assertEquals(loss.op.name, 'L1Regularizer/value')
      self.assertAlmostEqual(loss.eval(), num_elem * weight, 5)

  def testL2Regularizer(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      loss = losses.l2_regularizer()(tensor)
      self.assertEquals(loss.op.name, 'L2Regularizer/value')
      self.assertAlmostEqual(loss.eval(), num_elem / 2, 5)

  def testL2RegularizerWithScope(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      loss = losses.l2_regularizer(scope='L2')(tensor)
      self.assertEquals(loss.op.name, 'L2/value')
      self.assertAlmostEqual(loss.eval(), num_elem / 2, 5)

  def testL2RegularizerWithWeight(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      weight = 0.01
      loss = losses.l2_regularizer(weight)(tensor)
      self.assertEquals(loss.op.name, 'L2Regularizer/value')
      self.assertAlmostEqual(loss.eval(), num_elem * weight / 2, 5)

  def testL1L2Regularizer(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      loss = losses.l1_l2_regularizer()(tensor)
      self.assertEquals(loss.op.name, 'L1L2Regularizer/value')
      self.assertAlmostEqual(loss.eval(), num_elem + num_elem / 2, 5)

  def testL1L2RegularizerWithScope(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      loss = losses.l1_l2_regularizer(scope='L1L2')(tensor)
      self.assertEquals(loss.op.name, 'L1L2/value')
      self.assertAlmostEqual(loss.eval(), num_elem + num_elem / 2, 5)

  def testL1L2RegularizerWithWeights(self):
    with self.test_session():
      shape = [5, 5, 5]
      num_elem = 5 * 5 * 5
      tensor = tf.constant(1.0, shape=shape)
      weight_l1 = 0.01
      weight_l2 = 0.05
      loss = losses.l1_l2_regularizer(weight_l1, weight_l2)(tensor)
      self.assertEquals(loss.op.name, 'L1L2Regularizer/value')
      self.assertAlmostEqual(loss.eval(),
                             num_elem * weight_l1 + num_elem * weight_l2 / 2, 5)


class CrossEntropyLossTest(tf.test.TestCase):

  def testCrossEntropyLossAllCorrect(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
      loss = losses.cross_entropy_loss(logits, labels)
      self.assertEquals(loss.op.name, 'CrossEntropyLoss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testCrossEntropyLossAllWrong(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
      loss = losses.cross_entropy_loss(logits, labels)
      self.assertEquals(loss.op.name, 'CrossEntropyLoss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testCrossEntropyLossAllWrongWithWeight(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
      loss = losses.cross_entropy_loss(logits, labels, weight=0.5)
      self.assertEquals(loss.op.name, 'CrossEntropyLoss/value')
      self.assertAlmostEqual(loss.eval(), 5.0, 3)

if __name__ == '__main__':
  tf.test.main()
