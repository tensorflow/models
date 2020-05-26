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

"""Tests for object_detection.np_mask_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.utils import np_mask_ops


class MaskOpsTests(tf.test.TestCase):

  def setUp(self):
    masks1_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks1_1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks1 = np.stack([masks1_0, masks1_1])
    masks2_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks2_1 = np.array([[1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks2_2 = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0]],
                        dtype=np.uint8)
    masks2 = np.stack([masks2_0, masks2_1, masks2_2])
    self.masks1 = masks1
    self.masks2 = masks2

  def testArea(self):
    areas = np_mask_ops.area(self.masks1)
    expected_areas = np.array([8.0, 10.0], dtype=np.float32)
    self.assertAllClose(expected_areas, areas)

  def testIntersection(self):
    intersection = np_mask_ops.intersection(self.masks1, self.masks2)
    expected_intersection = np.array(
        [[8.0, 0.0, 8.0], [0.0, 9.0, 7.0]], dtype=np.float32)
    self.assertAllClose(intersection, expected_intersection)

  def testIOU(self):
    iou = np_mask_ops.iou(self.masks1, self.masks2)
    expected_iou = np.array(
        [[1.0, 0.0, 8.0/25.0], [0.0, 9.0 / 16.0, 7.0 / 28.0]], dtype=np.float32)
    self.assertAllClose(iou, expected_iou)

  def testIOA(self):
    ioa21 = np_mask_ops.ioa(self.masks1, self.masks2)
    expected_ioa21 = np.array([[1.0, 0.0, 8.0/25.0],
                               [0.0, 9.0/15.0, 7.0/25.0]],
                              dtype=np.float32)
    self.assertAllClose(ioa21, expected_ioa21)


if __name__ == '__main__':
  tf.test.main()
