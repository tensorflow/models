# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for maskrcnn_losses."""

from absl.testing import parameterized
import tensorflow as tf

from official.vision.losses import maskrcnn_losses


class MaskrcnnLossesTest(parameterized.TestCase, tf.test.TestCase):

  def testRpnScoreLoss(self):
    batch_size = 2
    height = 32
    width = 32
    num_anchors = 10
    score_outputs = {
        '1': tf.random.uniform([batch_size, height, width, num_anchors])
    }
    score_targets = {
        '1':
            tf.random.uniform([batch_size, height, width, num_anchors],
                              minval=-1,
                              maxval=2,
                              dtype=tf.int32)
    }
    loss_fn = maskrcnn_losses.RpnScoreLoss(rpn_batch_size_per_im=8)
    self.assertEqual(tf.rank(loss_fn(score_outputs, score_targets)), 0)

  def testRpnBoxLoss(self):
    batch_size = 2
    height = 32
    width = 32
    num_anchors = 10
    box_outputs = {
        '1': tf.random.uniform([batch_size, height, width, num_anchors * 4])
    }
    box_targets = {
        '1': tf.random.uniform([batch_size, height, width, num_anchors * 4])
    }
    loss_fn = maskrcnn_losses.RpnBoxLoss(huber_loss_delta=1. / 9.)
    self.assertEqual(tf.rank(loss_fn(box_outputs, box_targets)), 0)

  def testRpnBoxLossValidBox(self):
    box_outputs = {'1': tf.constant([[[[0.2, 0.2, 1.4, 1.4]]]])}
    box_targets = {'1': tf.constant([[[[0., 0., 1., 1.]]]])}
    loss_fn = maskrcnn_losses.RpnBoxLoss(huber_loss_delta=1. / 9.)
    self.assertAllClose(loss_fn(box_outputs, box_targets), 0.027093, atol=1e-4)

  def testRpnBoxLossInvalidBox(self):
    box_outputs = {'1': tf.constant([[[[0.2, 0.2, 1.4, 1.4]]]])}
    box_targets = {'1': tf.constant([[[[0., 0., 0., 0.]]]])}
    loss_fn = maskrcnn_losses.RpnBoxLoss(huber_loss_delta=1. / 9.)
    self.assertAllClose(loss_fn(box_outputs, box_targets), 0., atol=1e-4)

  @parameterized.parameters(True, False)
  def testFastrcnnClassLoss(self, use_binary_cross_entropy):
    batch_size = 2
    num_boxes = 10
    num_classes = 5
    class_outputs = tf.random.uniform([batch_size, num_boxes, num_classes])
    class_targets = tf.random.uniform([batch_size, num_boxes],
                                      minval=0,
                                      maxval=num_classes + 1,
                                      dtype=tf.int32)
    loss_fn = maskrcnn_losses.FastrcnnClassLoss(use_binary_cross_entropy)
    self.assertEqual(tf.rank(loss_fn(class_outputs, class_targets)), 0)

  def testFastrcnnClassLossTopK(self):
    class_targets = tf.constant([[0, 0, 0, 2]])
    class_outputs = tf.constant([[
        [100.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]])
    self.assertAllClose(
        maskrcnn_losses.FastrcnnClassLoss(top_k_percent=0.5)(
            class_outputs, class_targets
        ),
        0.775718,
        atol=1e-4,
    )
    self.assertAllClose(
        maskrcnn_losses.FastrcnnClassLoss(top_k_percent=1.0)(
            class_outputs, class_targets
        ),
        0.387861,
        atol=1e-4,
    )

  def testFastrcnnBoxLoss(self):
    batch_size = 2
    num_boxes = 10
    num_classes = 5
    box_outputs = tf.random.uniform([batch_size, num_boxes, num_classes * 4])
    box_targets = tf.random.uniform([batch_size, num_boxes, 4])
    class_targets = tf.random.uniform([batch_size, num_boxes],
                                      minval=0,
                                      maxval=num_classes + 1,
                                      dtype=tf.int32)
    loss_fn = maskrcnn_losses.FastrcnnBoxLoss(huber_loss_delta=1.)
    self.assertEqual(
        tf.rank(loss_fn(box_outputs, class_targets, box_targets)), 0)

  def testMaskrcnnLoss(self):
    batch_size = 2
    num_masks = 10
    mask_height = 16
    mask_width = 16
    num_classes = 5
    mask_outputs = tf.random.uniform(
        [batch_size, num_masks, mask_height, mask_width])
    mask_targets = tf.cast(
        tf.random.uniform([batch_size, num_masks, mask_height, mask_width],
                          minval=0,
                          maxval=2,
                          dtype=tf.int32), tf.float32)
    select_class_targets = tf.random.uniform([batch_size, num_masks],
                                             minval=0,
                                             maxval=num_classes + 1,
                                             dtype=tf.int32)
    loss_fn = maskrcnn_losses.MaskrcnnLoss()
    self.assertEqual(
        tf.rank(loss_fn(mask_outputs, mask_targets, select_class_targets)), 0)


if __name__ == '__main__':
  tf.test.main()
