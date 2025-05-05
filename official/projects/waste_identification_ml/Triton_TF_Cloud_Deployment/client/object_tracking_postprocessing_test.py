# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import pandas as pd
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import object_tracking_postprocessing

TEST_DATA = pd.DataFrame({
    "particle": [1, 1, 2, 2, 3, 3, 3, 4],
    "source_name": [
        "src1",
        "src1",
        "src2",
        "src2",
        "src3",
        "src3",
        "src3",
        "src4",
    ],
    "image_name": [
        "img1",
        "img1",
        "img2",
        "img2",
        "img3",
        "img3",
        "img3",
        "img4",
    ],
    "detection_scores": [0.8, 0.9, 0.5, 0.7, 0.6, 0.7, 0.8, 0.9],
    "detection_classes": ["A", "A", "B", "C", "D", "D", "E", "A"],
    "detection_classes_names": [
        "ClassA",
        "ClassA",
        "ClassB",
        "ClassC",
        "ClassD",
        "ClassD",
        "ClassE",
        "ClassA",
    ],
    "color": [
        "red",
        "red",
        "blue",
        "blue",
        "green",
        "green",
        "green",
        "orange",
    ],
    "creation_time": [
        "2024-01-01",
        "2024-01-01",
        "2024-01-02",
        "2024-01-02",
        "2024-01-03",
        "2024-01-03",
        "2024-01-03",
        "2024-01-04",
    ],
    "bbox_0": [10, 10, 20, 20, 30, 30, 30, 40],
    "bbox_1": [15, 15, 25, 25, 35, 35, 35, 45],
    "bbox_2": [50, 50, 60, 60, 70, 70, 70, 80],
    "bbox_3": [55, 55, 65, 65, 75, 75, 75, 85],
})


class TestProcessTrackingResult(unittest.TestCase):

  def test_particle_grouping(self):
    """Test if the function correctly aggregates the tracking data."""
    result = object_tracking_postprocessing.process_tracking_result(
        TEST_DATA.copy()
    )

    self.assertEqual(len(result), 4)

  def test_single_class_selection(self):
    """Test if the function correctly selects a class when there's only one."""
    result = object_tracking_postprocessing.process_tracking_result(
        TEST_DATA.copy()
    )

    self.assertEqual(
        result[result["particle"] == 4]["detection_classes"].values, "A"
    )

  def test_modal_class_selection(self):
    """Test if the function correctly selects the most common class."""
    result = object_tracking_postprocessing.process_tracking_result(
        TEST_DATA.copy()
    )

    self.assertEqual(
        result[result["particle"] == 3]["detection_classes"].values, "D"
    )

  def test_single_class_selection_with_tie(self):
    """Test if the function correctly selects a class when there's a modal tie."""
    result = object_tracking_postprocessing.process_tracking_result(
        TEST_DATA.copy()
    )

    self.assertEqual(
        result[result["particle"] == 1]["detection_classes"].values, "A"
    )

  def test_tie_multiple_class_selection(self):
    """Test class selection using scores for modal tie with multiple classes."""
    result = object_tracking_postprocessing.process_tracking_result(
        TEST_DATA.copy()
    )

    self.assertEqual(
        result[result["particle"] == 2]["detection_classes"].values, "C"
    )

if __name__ == "__main__":
  unittest.main()
