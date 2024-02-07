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

"""box_ops tests."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.yolo.ops import box_ops


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((1), (4))
  def test_box_conversions(self, num_boxes):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    expected_shape = np.array([num_boxes, 4])
    xywh_box = box_ops.yxyx_to_xcycwh(boxes)
    yxyx_box = box_ops.xcycwh_to_yxyx(boxes)
    self.assertAllEqual(tf.shape(xywh_box).numpy(), expected_shape)
    self.assertAllEqual(tf.shape(yxyx_box).numpy(), expected_shape)

  @parameterized.parameters((1), (5), (7))
  def test_ious(self, num_boxes):
    boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
    expected_shape = np.array([
        num_boxes,
    ])
    expected_iou = np.ones([
        num_boxes,
    ])
    iou = box_ops.compute_iou(boxes, boxes)
    _, giou = box_ops.compute_giou(boxes, boxes)
    _, ciou = box_ops.compute_ciou(boxes, boxes)
    _, diou = box_ops.compute_diou(boxes, boxes)
    self.assertAllEqual(tf.shape(iou).numpy(), expected_shape)
    self.assertArrayNear(iou, expected_iou, 0.001)
    self.assertArrayNear(giou, expected_iou, 0.001)
    self.assertArrayNear(ciou, expected_iou, 0.001)
    self.assertArrayNear(diou, expected_iou, 0.001)

if __name__ == '__main__':
  tf.test.main()
