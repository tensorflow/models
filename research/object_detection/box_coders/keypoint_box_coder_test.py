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

"""Tests for object_detection.box_coder.keypoint_box_coder."""

import tensorflow as tf

from object_detection.box_coders import keypoint_box_coder
from object_detection.core import box_list
from object_detection.core import standard_fields as fields


class KeypointBoxCoderTest(tf.test.TestCase):

  def test_get_correct_relative_codes_after_encoding(self):
    boxes = [[10., 10., 20., 15.],
             [0.2, 0.1, 0.5, 0.4]]
    keypoints = [[[15., 12.], [10., 15.]],
                 [[0.5, 0.3], [0.2, 0.4]]]
    num_keypoints = len(keypoints[0])
    anchors = [[15., 12., 30., 18.],
               [0.1, 0.0, 0.7, 0.9]]
    expected_rel_codes = [
        [-0.5, -0.416666, -0.405465, -0.182321,
         -0.5, -0.5, -0.833333, 0.],
        [-0.083333, -0.222222, -0.693147, -1.098612,
         0.166667, -0.166667, -0.333333, -0.055556]
    ]
    boxes = box_list.BoxList(tf.constant(boxes))
    boxes.add_field(fields.BoxListFields.keypoints, tf.constant(keypoints))
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = keypoint_box_coder.KeypointBoxCoder(num_keypoints)
    rel_codes = coder.encode(boxes, anchors)
    with self.test_session() as sess:
      rel_codes_out, = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_get_correct_relative_codes_after_encoding_with_scaling(self):
    boxes = [[10., 10., 20., 15.],
             [0.2, 0.1, 0.5, 0.4]]
    keypoints = [[[15., 12.], [10., 15.]],
                 [[0.5, 0.3], [0.2, 0.4]]]
    num_keypoints = len(keypoints[0])
    anchors = [[15., 12., 30., 18.],
               [0.1, 0.0, 0.7, 0.9]]
    scale_factors = [2, 3, 4, 5]
    expected_rel_codes = [
        [-1., -1.25, -1.62186, -0.911608,
         -1.0, -1.5, -1.666667, 0.],
        [-0.166667, -0.666667, -2.772588, -5.493062,
         0.333333, -0.5, -0.666667, -0.166667]
    ]
    boxes = box_list.BoxList(tf.constant(boxes))
    boxes.add_field(fields.BoxListFields.keypoints, tf.constant(keypoints))
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = keypoint_box_coder.KeypointBoxCoder(
        num_keypoints, scale_factors=scale_factors)
    rel_codes = coder.encode(boxes, anchors)
    with self.test_session() as sess:
      rel_codes_out, = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_get_correct_boxes_after_decoding(self):
    anchors = [[15., 12., 30., 18.],
               [0.1, 0.0, 0.7, 0.9]]
    rel_codes = [
        [-0.5, -0.416666, -0.405465, -0.182321,
         -0.5, -0.5, -0.833333, 0.],
        [-0.083333, -0.222222, -0.693147, -1.098612,
         0.166667, -0.166667, -0.333333, -0.055556]
    ]
    expected_boxes = [[10., 10., 20., 15.],
                      [0.2, 0.1, 0.5, 0.4]]
    expected_keypoints = [[[15., 12.], [10., 15.]],
                          [[0.5, 0.3], [0.2, 0.4]]]
    num_keypoints = len(expected_keypoints[0])
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = keypoint_box_coder.KeypointBoxCoder(num_keypoints)
    boxes = coder.decode(rel_codes, anchors)
    with self.test_session() as sess:
      boxes_out, keypoints_out = sess.run(
          [boxes.get(), boxes.get_field(fields.BoxListFields.keypoints)])
      self.assertAllClose(boxes_out, expected_boxes)
      self.assertAllClose(keypoints_out, expected_keypoints)

  def test_get_correct_boxes_after_decoding_with_scaling(self):
    anchors = [[15., 12., 30., 18.],
               [0.1, 0.0, 0.7, 0.9]]
    rel_codes = [
        [-1., -1.25, -1.62186, -0.911608,
         -1.0, -1.5, -1.666667, 0.],
        [-0.166667, -0.666667, -2.772588, -5.493062,
         0.333333, -0.5, -0.666667, -0.166667]
    ]
    scale_factors = [2, 3, 4, 5]
    expected_boxes = [[10., 10., 20., 15.],
                      [0.2, 0.1, 0.5, 0.4]]
    expected_keypoints = [[[15., 12.], [10., 15.]],
                          [[0.5, 0.3], [0.2, 0.4]]]
    num_keypoints = len(expected_keypoints[0])
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = keypoint_box_coder.KeypointBoxCoder(
        num_keypoints, scale_factors=scale_factors)
    boxes = coder.decode(rel_codes, anchors)
    with self.test_session() as sess:
      boxes_out, keypoints_out = sess.run(
          [boxes.get(), boxes.get_field(fields.BoxListFields.keypoints)])
      self.assertAllClose(boxes_out, expected_boxes)
      self.assertAllClose(keypoints_out, expected_keypoints)

  def test_very_small_width_nan_after_encoding(self):
    boxes = [[10., 10., 10.0000001, 20.]]
    keypoints = [[[10., 10.], [10.0000001, 20.]]]
    anchors = [[15., 12., 30., 18.]]
    expected_rel_codes = [[-0.833333, 0., -21.128731, 0.510826,
                           -0.833333, -0.833333, -0.833333, 0.833333]]
    boxes = box_list.BoxList(tf.constant(boxes))
    boxes.add_field(fields.BoxListFields.keypoints, tf.constant(keypoints))
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = keypoint_box_coder.KeypointBoxCoder(2)
    rel_codes = coder.encode(boxes, anchors)
    with self.test_session() as sess:
      rel_codes_out, = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)


if __name__ == '__main__':
  tf.test.main()
