# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import unittest
import numpy as np
import torch
from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection import models_utils


class UtilsTest(unittest.TestCase):
  """Tests for the utility functions."""

  def test_filter_boxes_keep_smaller_on_given_data(self):
    boxes = [
        [402.24, 0.54, 1343.04, 350.46],
        [402.24, 0.54, 955.20, 333.18],
        [930.24, 0.54, 1343.04, 351.54],
        [611.52, 0.54, 955.20, 334.26],
        [402.24, 0.54, 751.68, 305.10],
        [749.76, 592.38, 1055.04, 875.34],
        [941.76, 1012.50, 1039.68, 1078.38],
    ]
    masks = [f"mask_{i}" for i in range(len(boxes))]
    data = {"boxes": boxes, "masks": masks}

    expected_boxes = [
        [941.76, 1012.50, 1039.68, 1078.38],
        [749.76, 592.38, 1055.04, 875.34],
        [402.24, 0.54, 751.68, 305.10],
        [611.52, 0.54, 955.20, 334.26],
        [930.24, 0.54, 1343.04, 351.54],
    ]

    result = models_utils.filter_boxes_keep_smaller(data)
    actual_boxes = result["boxes"]

    # Sort both lists for comparison (optional depending on importance of order)
    actual_sorted = sorted(actual_boxes)
    expected_sorted = sorted(expected_boxes)

    self.assertEqual(len(actual_sorted), len(expected_sorted))
    for box1, box2 in zip(actual_sorted, expected_sorted):
      np.testing.assert_almost_equal(box1, box2, decimal=2)

  def test_convert_boxes_cxcywh_to_xyxy_withsinglebox_returnscorrectcoordinates(
      self,
  ):
    """Tests that a single box is converted correctly."""
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])  # cx, cy, w, h
    image_shape = (100, 200, 3)  # h, w, c
    expected_boxes = np.array([[80, 30, 120, 70]])  # x1, y1, x2, y2

    converted_boxes = models_utils.convert_boxes_cxcywh_to_xyxy(
        boxes, image_shape
    )
    np.testing.assert_array_equal(converted_boxes, expected_boxes)

  def test_convert_boxes_cxcywh_to_xyxy_withmultipleboxes_returnscorrectcoordinates(
      self,
  ):
    """Tests that multiple boxes are converted correctly."""
    boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.4],
        [0.25, 0.25, 0.1, 0.1],
    ])
    image_shape = (100, 200, 3)
    expected_boxes = np.array([
        [80, 30, 120, 70],
        [40, 20, 60, 30],
    ])
    converted_boxes = models_utils.convert_boxes_cxcywh_to_xyxy(
        boxes, image_shape
    )
    np.testing.assert_array_equal(converted_boxes, expected_boxes)

    boxes = torch.empty((0, 4))
    image_shape = (100, 200, 3)
    expected_boxes = np.empty((0, 4), dtype=int)
    converted_boxes = models_utils.convert_boxes_cxcywh_to_xyxy(
        boxes, image_shape
    )
    np.testing.assert_array_equal(converted_boxes, expected_boxes)


if __name__ == "__main__":
  unittest.main()
