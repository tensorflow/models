# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for ops."""
import numpy as np
import tensorflow as tf, tf_keras
from official.vision.utils.object_detection import ops


class OpsTest(tf.test.TestCase):

  def test_merge_boxes_with_multiple_labels(self):
    boxes = tf.constant(
        [
            [0.25, 0.25, 0.75, 0.75],
            [0.0, 0.0, 0.5, 0.75],
            [0.25, 0.25, 0.75, 0.75],
        ],
        dtype=tf.float32,
    )
    class_indices = tf.constant([0, 4, 2], dtype=tf.int32)
    class_confidences = tf.constant([0.8, 0.2, 0.1], dtype=tf.float32)
    num_classes = 5
    merged_boxes, merged_classes, merged_confidences, merged_box_indices = (
        ops.merge_boxes_with_multiple_labels(
            boxes, class_indices, class_confidences, num_classes
        )
    )

    expected_merged_boxes = np.array(
        [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=np.float32
    )
    expected_merged_classes = np.array(
        [[1, 0, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int32
    )
    expected_merged_confidences = np.array(
        [[0.8, 0, 0.1, 0, 0], [0, 0, 0, 0, 0.2]], dtype=np.float32
    )
    expected_merged_box_indices = np.array([0, 1], dtype=np.int32)

    self.assertAllClose(merged_boxes.numpy(), expected_merged_boxes)
    self.assertAllClose(merged_classes.numpy(), expected_merged_classes)
    self.assertAllClose(merged_confidences.numpy(), expected_merged_confidences)
    self.assertAllClose(merged_box_indices.numpy(), expected_merged_box_indices)

  def test_merge_boxes_with_multiple_labels_corner_case(self):
    boxes = tf.constant(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=tf.float32,
    )
    class_indices = tf.constant([0, 1, 2, 3, 2, 1, 0, 3], dtype=tf.int32)
    class_confidences = tf.constant(
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=tf.float32
    )
    num_classes = 4
    merged_boxes, merged_classes, merged_confidences, merged_box_indices = (
        ops.merge_boxes_with_multiple_labels(
            boxes, class_indices, class_confidences, num_classes
        )
    )
    expected_merged_boxes = np.array(
        [[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
        dtype=np.float32,
    )
    expected_merged_classes = np.array(
        [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.int32
    )
    expected_merged_confidences = np.array(
        [
            [0.1, 0, 0, 0.6],
            [0.4, 0.9, 0, 0],
            [0, 0.7, 0.2, 0],
            [0, 0, 0.3, 0.8],
        ],
        dtype=np.float32,
    )
    expected_merged_box_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    self.assertAllClose(merged_boxes.numpy(), expected_merged_boxes)
    self.assertAllClose(merged_classes.numpy(), expected_merged_classes)
    self.assertAllClose(merged_confidences.numpy(), expected_merged_confidences)
    self.assertAllClose(merged_box_indices.numpy(), expected_merged_box_indices)

  def test_merge_boxes_with_empty_inputs(self):
    boxes = tf.zeros([0, 4], dtype=tf.float32)
    class_indices = tf.constant([], dtype=tf.int32)
    class_confidences = tf.constant([], dtype=tf.float32)
    num_classes = 5
    merged_boxes, merged_classes, merged_confidences, merged_box_indices = (
        ops.merge_boxes_with_multiple_labels(
            boxes, class_indices, class_confidences, num_classes
        )
    )
    self.assertAllEqual(merged_boxes.shape, [0, 4])
    self.assertAllEqual(merged_classes.shape, [0, 5])
    self.assertAllEqual(merged_confidences.shape, [0, 5])
    self.assertAllEqual(merged_box_indices.shape, [0])


if __name__ == '__main__':
  tf.test.main()
