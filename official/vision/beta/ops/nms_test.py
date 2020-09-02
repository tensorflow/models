# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for nms.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.ops import nms


class SortedNonMaxSuppressionTest(tf.test.TestCase):

  def setUp(self):
    super(SortedNonMaxSuppressionTest, self).setUp()
    self.boxes_data = [[[0, 0, 1, 1], [0, 0.2, 1, 1.2], [0, 0.4, 1, 1.4],
                        [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                       [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                        [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    self.scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    self.max_output_size = 6
    self.iou_threshold = 0.5

  def testSortedNonMaxSuppressionOnTPU(self):
    boxes_np = np.array(self.boxes_data, dtype=np.float32)
    scores_np = np.array(self.scores_data, dtype=np.float32)
    iou_threshold_np = np.array(self.iou_threshold, dtype=np.float32)

    boxes = tf.constant(boxes_np)
    scores = tf.constant(scores_np)
    iou_threshold = tf.constant(iou_threshold_np)

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      scores_tpu, boxes_tpu = nms.sorted_non_max_suppression_padded(
          boxes=boxes,
          scores=scores,
          max_output_size=self.max_output_size,
          iou_threshold=iou_threshold)

    self.assertEqual(boxes_tpu.numpy().shape, (2, self.max_output_size, 4))
    self.assertAllClose(scores_tpu.numpy(),
                        [[0.9, 0.6, 0.4, 0.3, 0., 0.],
                         [0.8, 0.7, 0.5, 0.3, 0., 0.]])

  def testSortedNonMaxSuppressionOnCPU(self):
    boxes_np = np.array(self.boxes_data, dtype=np.float32)
    scores_np = np.array(self.scores_data, dtype=np.float32)
    iou_threshold_np = np.array(self.iou_threshold, dtype=np.float32)

    boxes = tf.constant(boxes_np)
    scores = tf.constant(scores_np)
    iou_threshold = tf.constant(iou_threshold_np)

    # Runs on CPU.
    scores_cpu, boxes_cpu = nms.sorted_non_max_suppression_padded(
        boxes=boxes,
        scores=scores,
        max_output_size=self.max_output_size,
        iou_threshold=iou_threshold)

    self.assertEqual(boxes_cpu.numpy().shape, (2, self.max_output_size, 4))
    self.assertAllClose(scores_cpu.numpy(),
                        [[0.9, 0.6, 0.4, 0.3, 0., 0.],
                         [0.8, 0.7, 0.5, 0.3, 0., 0.]])

  def testSortedNonMaxSuppressionOnTPUSpeed(self):
    boxes_np = np.random.rand(2, 12000, 4).astype(np.float32)
    scores_np = np.random.rand(2, 12000).astype(np.float32)
    iou_threshold_np = np.array(0.7, dtype=np.float32)

    boxes = tf.constant(boxes_np)
    scores = tf.constant(scores_np)
    iou_threshold = tf.constant(iou_threshold_np)

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      scores_tpu, boxes_tpu = nms.sorted_non_max_suppression_padded(
          boxes=boxes,
          scores=scores,
          max_output_size=2000,
          iou_threshold=iou_threshold)

    self.assertEqual(scores_tpu.numpy().shape, (2, 2000))
    self.assertEqual(boxes_tpu.numpy().shape, (2, 2000, 4))


if __name__ == '__main__':
  tf.test.main()
