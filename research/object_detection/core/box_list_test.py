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

import tensorflow as tf

from object_detection.core import box_list


class BoxListTest(tf.test.TestCase):
  """Tests for BoxList class."""

  def test_num_boxes(self):
    data = tf.constant([[0, 0, 1, 1], [1, 1, 2, 3], [3, 4, 5, 5]], tf.float32)
    expected_num_boxes = 3

    boxes = box_list.BoxList(data)
    with self.test_session() as sess:
      num_boxes_output = sess.run(boxes.num_boxes())
      self.assertEquals(num_boxes_output, expected_num_boxes)

  def test_get_correct_center_coordinates_and_sizes(self):
    boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    boxes = box_list.BoxList(tf.constant(boxes))
    centers_sizes = boxes.get_center_coordinates_and_sizes()
    expected_centers_sizes = [[15, 0.35], [12.5, 0.25], [10, 0.3], [5, 0.3]]
    with self.test_session() as sess:
      centers_sizes_out = sess.run(centers_sizes)
      self.assertAllClose(centers_sizes_out, expected_centers_sizes)

  def test_create_box_list_with_dynamic_shape(self):
    data = tf.constant([[0, 0, 1, 1], [1, 1, 2, 3], [3, 4, 5, 5]], tf.float32)
    indices = tf.reshape(tf.where(tf.greater([1, 0, 1], 0)), [-1])
    data = tf.gather(data, indices)
    assert data.get_shape().as_list() == [None, 4]
    expected_num_boxes = 2

    boxes = box_list.BoxList(data)
    with self.test_session() as sess:
      num_boxes_output = sess.run(boxes.num_boxes())
      self.assertEquals(num_boxes_output, expected_num_boxes)

  def test_transpose_coordinates(self):
    boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    boxes = box_list.BoxList(tf.constant(boxes))
    boxes.transpose_coordinates()
    expected_corners = [[10.0, 10.0, 15.0, 20.0], [0.1, 0.2, 0.4, 0.5]]
    with self.test_session() as sess:
      corners_out = sess.run(boxes.get())
      self.assertAllClose(corners_out, expected_corners)

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
    self.assertEquals(boxes.num_boxes_static(), 2)
    self.assertEquals(type(boxes.num_boxes_static()), int)

  def test_num_boxes_static_for_uninferrable_shape(self):
    placeholder = tf.placeholder(tf.float32, shape=[None, 4])
    boxes = box_list.BoxList(placeholder)
    self.assertEquals(boxes.num_boxes_static(), None)

  def test_as_tensor_dict(self):
    boxlist = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]], tf.float32))
    boxlist.add_field('classes', tf.constant([0, 1]))
    boxlist.add_field('scores', tf.constant([0.75, 0.2]))
    tensor_dict = boxlist.as_tensor_dict()

    expected_boxes = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]]
    expected_classes = [0, 1]
    expected_scores = [0.75, 0.2]

    with self.test_session() as sess:
      tensor_dict_out = sess.run(tensor_dict)
      self.assertAllEqual(3, len(tensor_dict_out))
      self.assertAllClose(expected_boxes, tensor_dict_out['boxes'])
      self.assertAllEqual(expected_classes, tensor_dict_out['classes'])
      self.assertAllClose(expected_scores, tensor_dict_out['scores'])

  def test_as_tensor_dict_with_features(self):
    boxlist = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]], tf.float32))
    boxlist.add_field('classes', tf.constant([0, 1]))
    boxlist.add_field('scores', tf.constant([0.75, 0.2]))
    tensor_dict = boxlist.as_tensor_dict(['boxes', 'classes', 'scores'])

    expected_boxes = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]]
    expected_classes = [0, 1]
    expected_scores = [0.75, 0.2]

    with self.test_session() as sess:
      tensor_dict_out = sess.run(tensor_dict)
      self.assertAllEqual(3, len(tensor_dict_out))
      self.assertAllClose(expected_boxes, tensor_dict_out['boxes'])
      self.assertAllEqual(expected_classes, tensor_dict_out['classes'])
      self.assertAllClose(expected_scores, tensor_dict_out['scores'])

  def test_as_tensor_dict_missing_field(self):
    boxlist = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]], tf.float32))
    boxlist.add_field('classes', tf.constant([0, 1]))
    boxlist.add_field('scores', tf.constant([0.75, 0.2]))
    with self.assertRaises(ValueError):
      boxlist.as_tensor_dict(['foo', 'bar'])


if __name__ == '__main__':
  tf.test.main()
