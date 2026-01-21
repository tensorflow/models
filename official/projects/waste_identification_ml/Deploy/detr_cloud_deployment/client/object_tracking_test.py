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
from unittest import mock

import numpy as np
import pandas as pd

from official.projects.waste_identification_ml.Deploy.detr_cloud_deployment.client import object_tracking

MODULE_PATH = object_tracking.__name__


class ObjectTrackerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracker = object_tracking.ObjectTracker(
        search_range=(10, 10), memory=2
    )

  def test_init(self):
    # Assert
    self.assertEqual(self.tracker.search_range, (10, 10))
    self.assertEqual(self.tracker.memory, 2)
    self.assertEqual(self.tracker.all_detections, [])

  @mock.patch(f"{MODULE_PATH}.cv2.resize")
  @mock.patch(f"{MODULE_PATH}.skimage.measure.regionprops_table")
  def test_extract_features_for_tracking(self, mock_regionprops, mock_resize):
    image = np.zeros((100, 100), dtype=np.uint8)
    masks = np.random.randint(0, 2, size=(2, 50, 50), dtype=np.uint8)
    results = {
        "masks": masks,
        "confidence": [0.9, 0.8],
        "labels": [1, 2],
        "class_names": ["class1", "class2"],
    }
    tracking_image_size = (20, 20)
    image_path = "/path/to/image_dir/image1.png"
    creation_time = "2025-01-01"
    frame_idx = 0
    colors = ["red", "blue"]

    mock_resize.return_value = np.ones((20, 20), dtype=np.uint8)
    mock_regionprops.return_value = {
        "centroid-0": [10.0],
        "centroid-1": [10.0],
        "bbox-0": [5],
        "bbox-1": [5],
        "bbox-2": [15],
        "bbox-3": [15],
        "area": [100],
        "convex_area": [100],
        "bbox_area": [100],
        "major_axis_length": [10],
        "minor_axis_length": [10],
        "eccentricity": [0],
        "label": [1],
        "mean_intensity": [0],
        "max_intensity": [0],
        "min_intensity": [0],
        "perimeter": [40],
    }

    self.tracker.extract_features_for_tracking(
        image,
        results,
        tracking_image_size,
        image_path,
        creation_time,
        frame_idx,
        colors,
    )

    self.assertEqual(mock_resize.call_count, 2)
    self.assertEqual(mock_regionprops.call_count, 2)
    self.assertEqual(len(self.tracker.all_detections), 1)
    df = self.tracker.all_detections[0]
    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(len(df), 2)
    self.assertIn("y", df.columns)
    self.assertEqual(df["frame"].iloc[0], 0)
    self.assertEqual(df["color"].tolist(), ["red", "blue"])

  @mock.patch(f"{MODULE_PATH}.cv2.resize")
  @mock.patch(f"{MODULE_PATH}.skimage.measure.regionprops_table")
  def test_extract_features_for_tracking_no_detections(
      self, mock_regionprops, mock_resize
  ):
    image = np.zeros((100, 100), dtype=np.uint8)
    results = {
        "masks": np.empty((0, 50, 50)),  # No masks
        "confidence": [],
        "labels": [],
        "class_names": [],
    }
    tracking_image_size = (20, 20)
    image_path = "/path/to/image_dir/image1.png"
    creation_time = "2025-01-01"
    frame_idx = 0
    colors = []

    self.tracker.extract_features_for_tracking(
        image,
        results,
        tracking_image_size,
        image_path,
        creation_time,
        frame_idx,
        colors,
    )

    mock_resize.assert_not_called()
    mock_regionprops.assert_not_called()
    self.assertEqual(len(self.tracker.all_detections), 1)
    df = self.tracker.all_detections[0]
    self.assertTrue(df.empty)
    self.assertListEqual(list(df.columns), list(self.tracker._properties))

  def test_run_tracking_empty(self):
    self.tracker.all_detections = []

    result_df = self.tracker.run_tracking()

    self.assertTrue(result_df.empty)

  @mock.patch(f"{MODULE_PATH}.tp.link_df")
  def test_run_tracking(self, mock_link_df):

    data1 = {
        "x": [10],
        "y": [10],
        "frame": [0],
        "bbox_0": [5],
        "bbox_1": [5],
        "bbox_2": [15],
        "bbox_3": [15],
        "major_axis_length": [10],
        "minor_axis_length": [10],
        "perimeter": [40],
        "source_name": ["d1"],
        "image_name": ["i1"],
        "detection_scores": [0.9],
        "detection_classes_names": ["c1"],
        "detection_classes": [1],
        "color": ["red"],
        "creation_time": ["t1"],
    }
    data2 = {
        "x": [12],
        "y": [12],
        "frame": [1],
        "bbox_0": [7],
        "bbox_1": [7],
        "bbox_2": [17],
        "bbox_3": [17],
        "major_axis_length": [10],
        "minor_axis_length": [10],
        "perimeter": [40],
        "source_name": ["d1"],
        "image_name": ["i2"],
        "detection_scores": [0.95],
        "detection_classes_names": ["c1"],
        "detection_classes": [1],
        "color": ["red"],
        "creation_time": ["t2"],
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    self.tracker.all_detections = [df1, df2]

    linked_df = pd.concat([df1, df2], ignore_index=True)
    linked_df["particle"] = 0
    mock_link_df.return_value = linked_df

    result_df = self.tracker.run_tracking()

    mock_link_df.assert_called_once()
    self.assertIsInstance(result_df, pd.DataFrame)
    self.assertIn("particle", result_df.columns)
    self.assertNotIn("frame", result_df.columns)
    self.assertEqual(len(result_df), 2)
    self.assertEqual(result_df["particle"].iloc[0], 0)

  def test_process_tracking_results(self):

    track_data = {
        "x": [10, 12],
        "y": [10, 12],
        "bbox_0": [5, 7],
        "bbox_1": [5, 7],
        "bbox_2": [15, 17],
        "bbox_3": [15, 17],
        "major_axis_length": [10, 10],
        "minor_axis_length": [10, 10],
        "perimeter": [40, 40],
        "particle": [0, 0],
        "source_name": ["d1", "d1"],
        "image_name": ["i1", "i2"],
        "detection_scores": [0.9, 0.8],
        "detection_classes_names": ["apple", "banana"],
        "detection_classes": [1, 2],
        "color": ["red", "yellow"],
        "creation_time": ["t1", "t2"],
    }
    track_df = pd.DataFrame(track_data)

    final_df = self.tracker.process_tracking_results(track_df)

    self.assertIsInstance(final_df, pd.DataFrame)
    self.assertEqual(len(final_df), 1)  # 1 particle
    self.assertEqual(final_df["particle"].tolist(), [0])
    particle0 = final_df[final_df["particle"] == 0].iloc[0]
    self.assertEqual(particle0["detected_classes"], 1)
    self.assertEqual(particle0["detected_classes_names"], "apple")
    self.assertEqual(particle0["detected_colors"], "red")
    self.assertEqual(particle0["detection_scores"], 0.9)

  def test_select_class_with_scores_tie_break(self):
    group_data = {
        "detection_classes": [1, 2, 1, 2],
        "detection_scores": [0.8, 0.9, 0.7, 0.85],
        "detection_classes_names": ["c1", "c2", "c1", "c2"],
        "color": ["red", "blue", "red", "blue"],
    }
    group_df = pd.DataFrame(group_data)
    result = self.tracker._select_class_with_scores(group_df)

    self.assertEqual(result["class_name"], "c2")

  def test_select_class_with_higher_frequency(self):
    group_data = {
        "detection_classes": [1, 2, 1, 1],
        "detection_scores": [0.8, 0.9, 0.7, 0.85],
        "detection_classes_names": ["c1", "c2", "c1", "c1"],
        "color": ["red", "blue", "red", "red"],
    }
    group_df = pd.DataFrame(group_data)
    result = self.tracker._select_class_with_scores(group_df)

    self.assertEqual(result["class_name"], "c1")


if __name__ == "__main__":
  unittest.main()
