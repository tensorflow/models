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
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.box_coders import square_box_coder
from object_detection.core import box_list
from object_detection.utils import test_case


class SquareBoxCoderTest(test_case.TestCase):

  def test_correct_relative_codes_with_default_scale(self):
    boxes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    expected_rel_codes = [[-0.790569, -0.263523, -0.293893],
                          [-0.068041, -0.272166, -0.89588]]
    def graph_fn(boxes, anchors):
      scale_factors = None
      boxes = box_list.BoxList(boxes)
      anchors = box_list.BoxList(anchors)
      coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)

  def test_correct_relative_codes_with_non_default_scale(self):
    boxes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    expected_rel_codes = [[-1.581139, -0.790569, -1.175573],
                          [-0.136083, -0.816497, -3.583519]]
    def graph_fn(boxes, anchors):
      scale_factors = [2, 3, 4]
      boxes = box_list.BoxList(boxes)
      anchors = box_list.BoxList(anchors)
      coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-03,
                        atol=1e-03)

  def test_correct_relative_codes_with_small_width(self):
    boxes = np.array([[10.0, 10.0, 10.0000001, 20.0]], np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0]], np.float32)
    expected_rel_codes = [[-1.317616, 0., -20.670586]]
    def graph_fn(boxes, anchors):
      scale_factors = None
      boxes = box_list.BoxList(boxes)
      anchors = box_list.BoxList(anchors)
      coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)

  def test_correct_boxes_with_default_scale(self):
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    rel_codes = np.array([[-0.5, -0.416666, -0.405465],
                          [-0.083333, -0.222222, -0.693147]], np.float32)
    expected_boxes = [[14.594306, 7.884875, 20.918861, 14.209432],
                      [0.155051, 0.102989, 0.522474, 0.470412]]
    def graph_fn(rel_codes, anchors):
      scale_factors = None
      anchors = box_list.BoxList(anchors)
      coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
      boxes = coder.decode(rel_codes, anchors).get()
      return boxes
    boxes_out = self.execute(graph_fn, [rel_codes, anchors])
    self.assertAllClose(boxes_out, expected_boxes, rtol=1e-04,
                        atol=1e-04)

  def test_correct_boxes_with_non_default_scale(self):
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    rel_codes = np.array(
        [[-1., -1.25, -1.62186], [-0.166667, -0.666667, -2.772588]], np.float32)
    expected_boxes = [[14.594306, 7.884875, 20.918861, 14.209432],
                      [0.155051, 0.102989, 0.522474, 0.470412]]
    def graph_fn(rel_codes, anchors):
      scale_factors = [2, 3, 4]
      anchors = box_list.BoxList(anchors)
      coder = square_box_coder.SquareBoxCoder(scale_factors=scale_factors)
      boxes = coder.decode(rel_codes, anchors).get()
      return boxes
    boxes_out = self.execute(graph_fn, [rel_codes, anchors])
    self.assertAllClose(boxes_out, expected_boxes, rtol=1e-04,
                        atol=1e-04)


if __name__ == '__main__':
  tf.test.main()
