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
"""Tests for box_matcher.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.layers import box_matcher


class BoxMatcherTest(tf.test.TestCase):

  def test_box_matcher(self):
    boxes_np = np.array(
        [[
            [0, 0, 1, 1],
            [5, 0, 10, 5],
        ]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)

    gt_boxes_np = np.array(
        [[
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ]])
    gt_boxes = tf.constant(gt_boxes_np, dtype=tf.float32)
    gt_classes_np = np.array([[2, 10, 3, -1]])
    gt_classes = tf.constant(gt_classes_np, dtype=tf.int32)

    fg_threshold = 0.5
    bg_thresh_hi = 0.2
    bg_thresh_lo = 0.0

    matcher = box_matcher.BoxMatcher(fg_threshold, bg_thresh_hi, bg_thresh_lo)

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      (matched_gt_boxes_tpu, matched_gt_classes_tpu, matched_gt_indices_tpu,
       positive_matches_tpu, negative_matches_tpu, ignored_matches_tpu) = (
           matcher(boxes, gt_boxes, gt_classes))

    # Runs on CPU.
    (matched_gt_boxes_cpu, matched_gt_classes_cpu, matched_gt_indices_cpu,
     positive_matches_cpu, negative_matches_cpu, ignored_matches_cpu) = (
         matcher(boxes, gt_boxes, gt_classes))

    # correctness
    self.assertNDArrayNear(
        matched_gt_boxes_cpu.numpy(),
        [[[0, 0, 0, 0], [5, 0, 10, 5]]], 1e-4)
    self.assertAllEqual(
        matched_gt_classes_cpu.numpy(), [[0, 3]])
    self.assertAllEqual(
        matched_gt_indices_cpu.numpy(), [[-1, 2]])
    self.assertAllEqual(
        positive_matches_cpu.numpy(), [[False, True]])
    self.assertAllEqual(
        negative_matches_cpu.numpy(), [[True, False]])
    self.assertAllEqual(
        ignored_matches_cpu.numpy(), [[False, False]])

    # consistency.
    self.assertNDArrayNear(
        matched_gt_boxes_cpu.numpy(), matched_gt_boxes_tpu.numpy(), 1e-4)
    self.assertAllEqual(
        matched_gt_classes_cpu.numpy(), matched_gt_classes_tpu.numpy())
    self.assertAllEqual(
        matched_gt_indices_cpu.numpy(), matched_gt_indices_tpu.numpy())
    self.assertAllEqual(
        positive_matches_cpu.numpy(), positive_matches_tpu.numpy())
    self.assertAllEqual(
        negative_matches_cpu.numpy(), negative_matches_tpu.numpy())
    self.assertAllEqual(
        ignored_matches_cpu.numpy(), ignored_matches_tpu.numpy())

  def test_serialize_deserialize(self):
    kwargs = dict(
        foreground_iou_threshold=0.5,
        background_iou_high_threshold=0.5,
        background_iou_low_threshold=0.5,
    )
    matcher = box_matcher.BoxMatcher(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(matcher.get_config(), expected_config)

    new_matcher = box_matcher.BoxMatcher.from_config(matcher.get_config())

    self.assertAllEqual(matcher.get_config(), new_matcher.get_config())


if __name__ == '__main__':
  tf.test.main()
