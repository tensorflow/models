# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for segmentation_losses."""

from absl.testing import parameterized
import tensorflow as tf

from official.vision.losses import segmentation_losses


class SegmentationLossTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True, False, 1.),
      (True, True, 0.5),
      (False, True, 1.),
  )
  def testSegmentationLoss(self, use_groundtruth_dimension,
                           use_binary_cross_entropy, top_k_percent_pixels):
    # [batch, height, width, num_layers]: [2, 3, 4, 1]
    labels = tf.random.uniform([2, 3, 4, 1], minval=0, maxval=6, dtype=tf.int32)
    # [batch, height, width, num_classes]: [2, 3, 4, 6]
    logits = tf.random.uniform([2, 3, 4, 6],
                               minval=-1,
                               maxval=1,
                               dtype=tf.float32)
    loss = segmentation_losses.SegmentationLoss(
        label_smoothing=0.,
        class_weights=[],
        ignore_label=255,
        use_groundtruth_dimension=use_groundtruth_dimension,
        use_binary_cross_entropy=use_binary_cross_entropy,
        top_k_percent_pixels=top_k_percent_pixels)(logits, labels)
    self.assertEqual(tf.rank(loss), 0)

  def testSegmentationLossGroundTruthIsMattingMap(self):
    # [batch, height, width, num_layers]: [2, 3, 4, 1]
    labels = tf.random.uniform([2, 3, 4, 1],
                               minval=0,
                               maxval=1,
                               dtype=tf.float32)
    # [batch, height, width, num_classes]: [2, 3, 4, 2]
    logits = tf.random.uniform([2, 3, 4, 2],
                               minval=-1,
                               maxval=1,
                               dtype=tf.float32)
    loss = segmentation_losses.SegmentationLoss(
        label_smoothing=0.,
        class_weights=[],
        ignore_label=255,
        use_groundtruth_dimension=True,
        use_binary_cross_entropy=False,
        top_k_percent_pixels=1.)(logits, labels)
    self.assertEqual(tf.rank(loss), 0)
