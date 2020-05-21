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

"""Tests for object_detection.core.box_list."""
import numpy as np
import tensorflow as tf

from object_detection.core import box_list
from object_detection.utils import test_case


class BoxListTest(test_case.TestCase):
  """Tests for BoxList class."""

  def test_num_boxes(self):
    def graph_fn():
      data = tf.constant([[0, 0, 1, 1], [1, 1, 2, 3], [3, 4, 5, 5]], tf.float32)
      boxes = box_list.BoxList(data)
      return boxes.num_boxes()
    num_boxes_out = self.execute(graph_fn, [])
    self.assertEqual(num_boxes_out, 3)

  def test_get_correct_center_coordinates_and_sizes(self):
    boxes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    def graph_fn(boxes):
      boxes = box_list.BoxList(boxes)
      centers_sizes = boxes.get_center_coordinates_and_sizes()
      return centers_sizes
    centers_sizes_out = self.execute(graph_fn, [boxes])
    expected_centers_sizes = [[15, 0.35], [12.5, 0.25], [10, 0.3], [5, 0.3]]
    self.assertAllClose(centers_sizes_out, expected_centers_sizes)

  def test_create_box_list_with_dynamic_shape(self):
    def graph_fn():
      data = tf.constant([[0, 0, 1, 1], [1, 1, 2, 3], [3, 4, 5, 5]], tf.float32)
      indices = tf.reshape(tf.where(tf.greater([1, 0, 1], 0)), [-1])
      data = tf.gather(data, indices)
      assert data.get_shape().as_list() == [None, 4]
      boxes = box_list.BoxList(data)
      return boxes.num_boxes()
    num_boxes = self.execute(graph_fn, [])
    self.assertEqual(num_boxes, 2)

  def test_transpose_coordinates(self):
    boxes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    def graph_fn(boxes):
      boxes = box_list.BoxList(boxes)
      boxes.transpose_coordinates()
      return boxes.get()
    transpoded_boxes = self.execute(graph_fn, [boxes])
    expected_corners = [[10.0, 10.0, 15.0, 20.0], [0.1, 0.2, 0.4, 0.5]]
    self.assertAllClose(transpoded_boxes, expected_corners)

  def test_box_list_invalid_inputs(self):
    data0 = tf.constant([[[0, 0, 1, 1], [3, 4, 5, 5]]], tf.float32)
    data1 = tf.constant([[0, 0, 1], [1, 1, 2], [3, 4, 5]], tf.float32)
    data2 = tf.constant([[0, 0, 1], [1, 1, 2], [3, 4, 5]], tf.int32)

    with self.assertRaises(ValueError):
      _ = box_list.BoxList(data0)
    with self.assertRaises(ValueError):
      _ = box_list.BoxList(data1)
    with self.assertRaises(ValueError):
      _ = box_list.BoxList(data2)

  def test_num_boxes_static(self):
    box_corners = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    boxes = box_list.BoxList(tf.constant(box_corners))
    self.assertEqual(boxes.num_boxes_static(), 2)
    self.assertEqual(type(boxes.num_boxes_static()), int)

  def test_as_tensor_dict(self):
    boxes = tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]],
                        tf.float32)
    boxlist = box_list.BoxList(boxes)
    classes = tf.constant([0, 1])
    boxlist.add_field('classes', classes)
    scores = tf.constant([0.75, 0.2])
    boxlist.add_field('scores', scores)
    tensor_dict = boxlist.as_tensor_dict()

    self.assertDictEqual(tensor_dict, {'scores': scores, 'classes': classes,
                                       'boxes': boxes})

  def test_as_tensor_dict_with_features(self):
    boxes = tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]],
                        tf.float32)
    boxlist = box_list.BoxList(boxes)
    classes = tf.constant([0, 1])
    boxlist.add_field('classes', classes)
    scores = tf.constant([0.75, 0.2])
    boxlist.add_field('scores', scores)
    tensor_dict = boxlist.as_tensor_dict(['scores', 'classes'])

    self.assertDictEqual(tensor_dict, {'scores': scores, 'classes': classes})

  def test_as_tensor_dict_missing_field(self):
    boxlist = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]], tf.float32))
    boxlist.add_field('classes', tf.constant([0, 1]))
    boxlist.add_field('scores', tf.constant([0.75, 0.2]))
    with self.assertRaises(ValueError):
      boxlist.as_tensor_dict(['foo', 'bar'])


if __name__ == '__main__':
  tf.test.main()
