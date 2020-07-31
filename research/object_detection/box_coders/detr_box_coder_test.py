# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.box_coder.detr_box_coder."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.box_coders import detr_box_coder
from object_detection.core import box_list
from object_detection.utils import test_case


class DETRBoxCoder(test_case.TestCase):

  def test_get_correct_relative_codes_after_encoding(self):
    boxes = np.array([[0.0, 2.5, 20.0, 17.5], [-0.05, -0.1, 0.45, 0.3]],
                     np.float32)
    expected_rel_codes = [[10.0, 10.0, 20.0, 15.0],
                          [0.2, 0.1, 0.5, 0.4]]
    def graph_fn(boxes):
      boxes = box_list.BoxList(boxes)
      coder = detr_box_coder.DETRBoxCoder()
      rel_codes = coder.encode(boxes, None)
      return rel_codes
    rel_codes_out = self.execute(graph_fn, [boxes])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)

  def test_get_correct_boxes_after_decoding(self):
    rel_codes = np.array([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]],
                     np.float32)
    expected_boxes = [[0.0, 2.5, 20.0, 17.5], [-0.05, -0.1, 0.45, 0.3]]
    def graph_fn(rel_codes):
      coder = detr_box_coder.DETRBoxCoder()
      boxes = coder.decode(rel_codes, None)
      return boxes.get()
    boxes_out = self.execute(graph_fn, [rel_codes])
    self.assertAllClose(boxes_out, expected_boxes, rtol=1e-04,
                        atol=1e-04)

if __name__ == '__main__':
  tf.test.main()
