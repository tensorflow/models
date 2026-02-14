# Copyright 2020 Google Research. All Rights Reserved.
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
# ======================================
"""Tests for iou_utils."""
from absl import logging
import tensorflow as tf
import iou_utils


class IouUtilsTest(tf.test.TestCase):
  """IoU test class."""

  def setUp(self):
    super(IouUtilsTest, self).setUp()
    self.pb = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                          dtype=tf.float32)
    self.tb = tf.constant(
        [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]], dtype=tf.float32)
    self.zeros = tf.constant([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.float32)

  def test_iou(self):
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.tb, 'iou'), [0.875, 1.])

  def test_ciou(self):
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.tb, 'ciou'), [1.408893, 1.548753])

  def test_diou(self):
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.tb, 'diou'), [1.406532, 1.531532])

  def test_giou(self):
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.tb, 'giou'), [1.075000, 1.933333])

  def test_iou_zero_target(self):
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.zeros, 'iou'), [0.0, 0.0])
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.zeros, 'ciou'), [0.0, 0.0])
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.zeros, 'diou'), [0.0, 0.0])
    self.assertAllClose(
        iou_utils.iou_loss(self.pb, self.zeros, 'giou'), [0.0, 0.0])

  def test_iou_multiple_anchors(self):
    pb = tf.tile(self.pb, [1, 2])
    tb = tf.tile(self.tb, [1, 2])
    self.assertAllClose(iou_utils.iou_loss(pb, tb, 'iou'), [1.75, 2.0])

  def test_iou_multiple_anchors_mixed(self):
    pb = tf.concat([self.pb, self.zeros], axis=-1)
    tb = tf.concat([self.tb, self.zeros], axis=-1)
    self.assertAllClose(iou_utils.iou_loss(pb, tb, 'iou'), [0.875, 1.0])

  def test_ciou_grad(self):
    pb = tf.concat([self.pb, self.zeros], axis=-1)
    tb = tf.concat([self.tb, self.zeros], axis=-1)
    with tf.GradientTape() as tape:
      tape.watch([pb, tb])
      loss = iou_utils.iou_loss(pb, tb, 'ciou')
    grad = tape.gradient(loss, [tb, pb])
    self.assertAlmostEqual(tf.reduce_sum(grad[0]).numpy(), 0.16687772)
    self.assertAlmostEqual(tf.reduce_sum(grad[1]).numpy(), -0.16687769)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()