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

"""Tests for object_detection.utils.test_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.utils import test_utils


class TestUtilsTest(tf.test.TestCase):

  def test_diagonal_gradient_image(self):
    """Tests if a good pyramid image is created."""
    pyramid_image = test_utils.create_diagonal_gradient_image(3, 4, 2)

    # Test which is easy to understand.
    expected_first_channel = np.array([[3, 2, 1, 0],
                                       [4, 3, 2, 1],
                                       [5, 4, 3, 2]], dtype=np.float32)
    self.assertAllEqual(np.squeeze(pyramid_image[:, :, 0]),
                        expected_first_channel)

    # Actual test.
    expected_image = np.array([[[3, 30],
                                [2, 20],
                                [1, 10],
                                [0, 0]],
                               [[4, 40],
                                [3, 30],
                                [2, 20],
                                [1, 10]],
                               [[5, 50],
                                [4, 40],
                                [3, 30],
                                [2, 20]]], dtype=np.float32)

    self.assertAllEqual(pyramid_image, expected_image)

  def test_random_boxes(self):
    """Tests if valid random boxes are created."""
    num_boxes = 1000
    max_height = 3
    max_width = 5
    boxes = test_utils.create_random_boxes(num_boxes,
                                           max_height,
                                           max_width)

    true_column = np.ones(shape=(num_boxes)) == 1
    self.assertAllEqual(boxes[:, 0] < boxes[:, 2], true_column)
    self.assertAllEqual(boxes[:, 1] < boxes[:, 3], true_column)

    self.assertTrue(boxes[:, 0].min() >= 0)
    self.assertTrue(boxes[:, 1].min() >= 0)
    self.assertTrue(boxes[:, 2].max() <= max_height)
    self.assertTrue(boxes[:, 3].max() <= max_width)

  def test_first_rows_close_as_set(self):
    a = [1, 2, 3, 0, 0]
    b = [3, 2, 1, 0, 0]
    k = 3
    self.assertTrue(test_utils.first_rows_close_as_set(a, b, k))

    a = [[1, 2], [1, 4], [0, 0]]
    b = [[1, 4 + 1e-9], [1, 2], [0, 0]]
    k = 2
    self.assertTrue(test_utils.first_rows_close_as_set(a, b, k))

    a = [[1, 2], [1, 4], [0, 0]]
    b = [[1, 4 + 1e-9], [2, 2], [0, 0]]
    k = 2
    self.assertFalse(test_utils.first_rows_close_as_set(a, b, k))


if __name__ == '__main__':
  tf.test.main()
