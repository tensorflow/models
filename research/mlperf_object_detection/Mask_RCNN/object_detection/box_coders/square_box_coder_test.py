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

"""Tests for object_detection.box_coder.square_box_coder."""

import tensorflow as tf

from object_detection.box_coders import square_box_coder
from object_detection.core import box_list


class SquareBoxCoderTest(tf.test.TestCase):

  def test_correct_relative_codes_with_default_scale(self):
    boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    anchors = [[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]]
    scale_factors = None
    expected_rel_codes = [[-0.790569, -0.263523, -0.293893],
                          [-0.068041, -0.272166, -0.89588]]

    boxes = box_list.BoxList(tf.constant(boxes))
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
    rel_codes = coder.encode(boxes, anchors)
    with self.test_session() as sess:
      (rel_codes_out,) = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_correct_relative_codes_with_non_default_scale(self):
    boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    anchors = [[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]]
    scale_factors = [2, 3, 4]
    expected_rel_codes = [[-1.581139, -0.790569, -1.175573],
                          [-0.136083, -0.816497, -3.583519]]
    boxes = box_list.BoxList(tf.constant(boxes))
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
    rel_codes = coder.encode(boxes, anchors)
    with self.test_session() as sess:
      (rel_codes_out,) = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_correct_relative_codes_with_small_width(self):
    boxes = [[10.0, 10.0, 10.0000001, 20.0]]
    anchors = [[15.0, 12.0, 30.0, 18.0]]
    scale_factors = None
    expected_rel_codes = [[-1.317616, 0., -20.670586]]
    boxes = box_list.BoxList(tf.constant(boxes))
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
    rel_codes = coder.encode(boxes, anchors)
    with self.test_session() as sess:
      (rel_codes_out,) = sess.run([rel_codes])
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def test_correct_boxes_with_default_scale(self):
    anchors = [[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]]
    rel_codes = [[-0.5, -0.416666, -0.405465],
                 [-0.083333, -0.222222, -0.693147]]
    scale_factors = None
    expected_boxes = [[14.594306, 7.884875, 20.918861, 14.209432],
                      [0.155051, 0.102989, 0.522474, 0.470412]]
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
    boxes = coder.decode(rel_codes, anchors)
    with self.test_session() as sess:
      (boxes_out,) = sess.run([boxes.get()])
      self.assertAllClose(boxes_out, expected_boxes)

  def test_correct_boxes_with_non_default_scale(self):
    anchors = [[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]]
    rel_codes = [[-1., -1.25, -1.62186], [-0.166667, -0.666667, -2.772588]]
    scale_factors = [2, 3, 4]
    expected_boxes = [[14.594306, 7.884875, 20.918861, 14.209432],
                      [0.155051, 0.102989, 0.522474, 0.470412]]
    anchors = box_list.BoxList(tf.constant(anchors))
    coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
    boxes = coder.decode(rel_codes, anchors)
    with self.test_session() as sess:
      (boxes_out,) = sess.run([boxes.get()])
      self.assertAllClose(boxes_out, expected_boxes)


if __name__ == '__main__':
  tf.test.main()
