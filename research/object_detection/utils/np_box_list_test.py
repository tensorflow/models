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

"""Tests for object_detection.utils.np_box_list_test."""

import numpy as np
import tensorflow as tf

from object_detection.utils import np_box_list


class BoxListTest(tf.test.TestCase):

  def test_invalid_box_data(self):
    with self.assertRaises(ValueError):
      np_box_list.BoxList([0, 0, 1, 1])

    with self.assertRaises(ValueError):
      np_box_list.BoxList(np.array([[0, 0, 1, 1]], dtype=int))

    with self.assertRaises(ValueError):
      np_box_list.BoxList(np.array([0, 1, 1, 3, 4], dtype=float))

    with self.assertRaises(ValueError):
      np_box_list.BoxList(np.array([[0, 1, 1, 3], [3, 1, 1, 5]], dtype=float))

  def test_has_field_with_existed_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    self.assertTrue(boxlist.has_field('boxes'))

  def test_has_field_with_nonexisted_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    self.assertFalse(boxlist.has_field('scores'))

  def test_get_field_with_existed_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    self.assertTrue(np.allclose(boxlist.get_field('boxes'), boxes))

  def test_get_field_with_nonexited_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    with self.assertRaises(ValueError):
      boxlist.get_field('scores')


class AddExtraFieldTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    self.boxlist = np_box_list.BoxList(boxes)

  def test_add_already_existed_field(self):
    with self.assertRaises(ValueError):
      self.boxlist.add_field('boxes', np.array([[0, 0, 0, 1, 0]], dtype=float))

  def test_add_invalid_field_data(self):
    with self.assertRaises(ValueError):
      self.boxlist.add_field('scores', np.array([0.5, 0.7], dtype=float))
    with self.assertRaises(ValueError):
      self.boxlist.add_field('scores',
                             np.array([0.5, 0.7, 0.9, 0.1], dtype=float))

  def test_add_single_dimensional_field_data(self):
    boxlist = self.boxlist
    scores = np.array([0.5, 0.7, 0.9], dtype=float)
    boxlist.add_field('scores', scores)
    self.assertTrue(np.allclose(scores, self.boxlist.get_field('scores')))

  def test_add_multi_dimensional_field_data(self):
    boxlist = self.boxlist
    labels = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
                      dtype=int)
    boxlist.add_field('labels', labels)
    self.assertTrue(np.allclose(labels, self.boxlist.get_field('labels')))

  def test_get_extra_fields(self):
    boxlist = self.boxlist
    self.assertItemsEqual(boxlist.get_extra_fields(), [])

    scores = np.array([0.5, 0.7, 0.9], dtype=float)
    boxlist.add_field('scores', scores)
    self.assertItemsEqual(boxlist.get_extra_fields(), ['scores'])

    labels = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
                      dtype=int)
    boxlist.add_field('labels', labels)
    self.assertItemsEqual(boxlist.get_extra_fields(), ['scores', 'labels'])

  def test_get_coordinates(self):
    y_min, x_min, y_max, x_max = self.boxlist.get_coordinates()

    expected_y_min = np.array([3.0, 14.0, 0.0], dtype=float)
    expected_x_min = np.array([4.0, 14.0, 0.0], dtype=float)
    expected_y_max = np.array([6.0, 15.0, 20.0], dtype=float)
    expected_x_max = np.array([8.0, 15.0, 20.0], dtype=float)

    self.assertTrue(np.allclose(y_min, expected_y_min))
    self.assertTrue(np.allclose(x_min, expected_x_min))
    self.assertTrue(np.allclose(y_max, expected_y_max))
    self.assertTrue(np.allclose(x_max, expected_x_max))

  def test_num_boxes(self):
    boxes = np.array([[0., 0., 100., 100.], [10., 30., 50., 70.]], dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    expected_num_boxes = 2
    self.assertEquals(boxlist.num_boxes(), expected_num_boxes)


if __name__ == '__main__':
  tf.test.main()
