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

"""Tests for object_detection.box_coder.faster_rcnn_box_coder."""
import numpy as np
import tensorflow as tf

from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.core import box_list
from object_detection.utils import test_case


class FasterRcnnBoxCoderTest(test_case.TestCase):

  def test_get_correct_relative_codes_after_encoding(self):
    boxes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    expected_rel_codes = [[-0.5, -0.416666, -0.405465, -0.182321],
                          [-0.083333, -0.222222, -0.693147, -1.098612]]
    def graph_fn(boxes, anchors):
      boxes = box_list.BoxList(boxes)
      anchors = box_list.BoxList(anchors)
      coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)

  def test_get_correct_relative_codes_after_encoding_with_scaling(self):
    boxes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    expected_rel_codes = [[-1., -1.25, -1.62186, -0.911608],
                          [-0.166667, -0.666667, -2.772588, -5.493062]]
    def graph_fn(boxes, anchors):
      scale_factors = [2, 3, 4, 5]
      boxes = box_list.BoxList(boxes)
      anchors = box_list.BoxList(anchors)
      coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
          scale_factors=scale_factors)
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)

  def test_get_correct_boxes_after_decoding(self):
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    rel_codes = np.array([[-0.5, -0.416666, -0.405465, -0.182321],
                          [-0.083333, -0.222222, -0.693147, -1.098612]],
                         np.float32)
    expected_boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    def graph_fn(rel_codes, anchors):
      anchors = box_list.BoxList(anchors)
      coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
      boxes = coder.decode(rel_codes, anchors)
      return boxes.get()
    boxes_out = self.execute(graph_fn, [rel_codes, anchors])
    self.assertAllClose(boxes_out, expected_boxes, rtol=1e-04,
                        atol=1e-04)

  def test_get_correct_boxes_after_decoding_with_scaling(self):
    anchors = np.array([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]],
                       np.float32)
    rel_codes = np.array([[-1., -1.25, -1.62186, -0.911608],
                          [-0.166667, -0.666667, -2.772588, -5.493062]],
                         np.float32)
    expected_boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    def graph_fn(rel_codes, anchors):
      scale_factors = [2, 3, 4, 5]
      anchors = box_list.BoxList(anchors)
      coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
          scale_factors=scale_factors)
      boxes = coder.decode(rel_codes, anchors).get()
      return boxes
    boxes_out = self.execute(graph_fn, [rel_codes, anchors])
    self.assertAllClose(expected_boxes, boxes_out, rtol=1e-04,
                        atol=1e-04)

  def test_very_small_Width_nan_after_encoding(self):
    boxes = np.array([[10.0, 10.0, 10.0000001, 20.0]], np.float32)
    anchors = np.array([[15.0, 12.0, 30.0, 18.0]], np.float32)
    expected_rel_codes = [[-0.833333, 0., -21.128731, 0.510826]]
    def graph_fn(boxes, anchors):
      boxes = box_list.BoxList(boxes)
      anchors = box_list.BoxList(anchors)
      coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)


if __name__ == '__main__':
  tf.test.main()
