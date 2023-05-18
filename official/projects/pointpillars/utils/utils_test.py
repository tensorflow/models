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

"""Tests for pointpillars utils."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.projects.pointpillars.utils import utils


class UtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([2, 1], [2, 1]),
      ([1, 1], [4, 3]),
      ([2, 2, 4], [2, 1, 5]),
  )
  def test_pad_or_trim_to_shape(self, original_shape, expected_shape):
    x = np.ones(shape=original_shape)
    x = utils.pad_or_trim_to_shape(x, expected_shape)
    self.assertAllEqual(x.shape, expected_shape)

  @parameterized.parameters(
      ([[1.1, 1.1, 2.2, 2.2]], 10.0, 5.0),
      ([[1.1, 10.1, 2.2, 10.2]], 10.0, 10.0),
      ([[-1.1, 10.1, -2.2, 10.2]], 5.0, 2.0),
  )
  def test_clip_boxes(self, boxes, height, width):
    boxes = np.array(boxes)
    boxes = utils.clip_boxes(boxes, height, width)
    self.assertGreaterEqual(boxes[:, 0], 0.0)
    self.assertGreaterEqual(boxes[:, 1], 0.0)
    self.assertLessEqual(boxes[:, 2], height)
    self.assertLessEqual(boxes[:, 3], width)

  def test_get_vehicle_xy(self):
    vehicle_xy = utils.get_vehicle_xy(10, 10, (-50, 50), (-50, 50))
    self.assertEqual(vehicle_xy, (5, 5))

  @parameterized.parameters(
      ([[1.0, 1.0]]),
      ([[-2.2, 4.2]]),
      ([[3.7, -10.3]]),
  )
  def test_frame_to_image_and_image_to_frame(self, frame_xy):
    frame_xy = np.array(frame_xy)
    vehicle_xy = (0, 0)
    resolution = 1.0
    image_xy = utils.frame_to_image_coord(frame_xy, vehicle_xy, 1 / resolution)
    frame_xy_1 = utils.image_to_frame_coord(image_xy, vehicle_xy, resolution)
    self.assertAllEqual(frame_xy_1, np.floor(frame_xy))

  @parameterized.parameters(
      ([[1.0, 1.0, 2.0, 2.0]]),
      ([[-2.2, -4.2, 2.2, 4.2]]),
  )
  def test_frame_to_image_boxes_and_image_to_frame_boxes(self, frame_boxes):
    frame_boxes = np.array(frame_boxes)
    vehicle_xy = (0, 0)
    resolution = 1.0
    image_boxes = utils.frame_to_image_boxes(frame_boxes, vehicle_xy,
                                             1 / resolution)
    frame_boxes_1 = utils.image_to_frame_boxes(image_boxes, vehicle_xy,
                                               resolution)
    self.assertAllClose(frame_boxes_1, frame_boxes)

  def test_generate_anchors(self):
    min_level = 1
    max_level = 3
    image_size = [16, 16]
    anchor_sizes = [(2.0, 1.0)]
    all_anchors = utils.generate_anchors(min_level, max_level, image_size,
                                         anchor_sizes)
    for level in range(min_level, max_level + 1):
      anchors = all_anchors[str(level)]
      stride = 2**level
      self.assertAllEqual(anchors.shape.as_list(),
                          [image_size[0] / stride, image_size[1] / stride, 4])


if __name__ == '__main__':
  tf.test.main()
