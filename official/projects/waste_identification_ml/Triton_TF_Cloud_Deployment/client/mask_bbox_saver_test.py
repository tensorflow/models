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

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import mask_bbox_saver


class VisualizeTrackingResultsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tracking_folder = "mock_tracking_output"
    self.image_name = "img1.png"
    self.mock_input_dir = "/mock/path/images"
    self.output_suffix = "_cropped_objects"
    self.mock_image = np.ones((200, 200, 3), dtype=np.uint8) * 255

    self.agg_features = pd.DataFrame({
        "detection_classes_names": ["Bottle"],
        "image_name": ["img1.png"],
        "bbox_0": [0.1],
        "bbox_1": [0.1],
        "bbox_2": [0.5],
        "bbox_3": [0.5],
        "particle": [1],
        "detection_scores": [0.9],
    })

    self.tracking_df = pd.DataFrame({
        "image_name": [self.image_name, self.image_name],
        "x": [15, 45],
        "y": [25, 65],
        "particle": [1, 2],
    })

    self.tracking_images = {self.image_name: self.mock_image}

  def test_dataframe_validity(self):
    self.assertFalse(
        self.tracking_df.isnull().values.any(),
        msg="DataFrame contains NaN values",
    )

    expected_cols = {"image_name", "x", "y", "particle"}
    actual_cols = set(self.tracking_df.columns)
    self.assertEqual(
        expected_cols, actual_cols, msg="DataFrame missing required columns"
    )

  @mock.patch("cv2.imwrite")
  def test_visualize_tracking_function_runs(self, mock_imwrite):
    result_folder = mask_bbox_saver.visualize_tracking_results(
        tracking_features=self.tracking_df,
        tracking_images=self.tracking_images,
        tracking_folder=self.tracking_folder,
    )

    self.assertEqual(result_folder, self.tracking_folder)
    mock_imwrite.assert_called()  # At least one image was attempted to be saved

    # Optional: Check that the correct file path was constructed
    expected_output_path = os.path.join(self.tracking_folder, self.image_name)
    mock_imwrite.assert_any_call(expected_output_path, mock.ANY)

  def test_dataframe_is_valid(self):
    self.assertFalse(self.agg_features.empty)
    required_columns = {
        "detection_classes_names",
        "image_name",
        "bbox_0",
        "bbox_1",
        "bbox_2",
        "bbox_3",
        "particle",
        "detection_scores",
    }
    self.assertTrue(required_columns.issubset(set(self.agg_features.columns)))

  @mock.patch("cv2.imwrite")
  @mock.patch(
      "cv2.cvtColor", return_value=np.ones((200, 200, 3), dtype=np.uint8)
  )
  @mock.patch("cv2.imread", return_value=np.ones((200, 200, 3), dtype=np.uint8))
  @mock.patch("os.makedirs")
  def test_save_cropped_objects_success(
      self, mock_makedirs, mock_imread, mock_cvtcolor, mock_imwrite  # pylint: disable=unused-argument
  ):

    agg_features = pd.DataFrame({
        "detection_classes_names": ["Bottle"],
        "image_name": ["img1.png"],
        "bbox_0": [0.1],
        "bbox_1": [0.1],
        "bbox_2": [0.5],
        "bbox_3": [0.5],
        "particle": [1],
        "detection_scores": [0.9],
    })

    input_directory = "/mock/path/images"
    output_suffix = "_cropped_objects"
    mock_bbox = (10, 10, 100, 100)

    def mock_resize_bbox(bbox, old_size, new_size):  # pylint: disable=unused-argument
      return mock_bbox

    output_path = mask_bbox_saver.save_cropped_objects(
        agg_features=agg_features,
        input_directory=input_directory,
        height_tracking=100,
        width_tracking=100,
        resize_bbox=mock_resize_bbox,
        output_suffix=output_suffix,
    )

    expected_output = os.path.basename(input_directory) + output_suffix
    self.assertEqual(output_path, expected_output)
    mock_makedirs.assert_any_call(expected_output, exist_ok=True)
    mock_imread.assert_called_once()
    mock_imwrite.assert_called_once()

  @mock.patch("os.makedirs")
  def test_empty_dataframe_returns_early(self, mock_makedirs):
    agg_features = pd.DataFrame()  # Empty DataFrame
    input_directory = "/mock/path/images"
    output_suffix = "_cropped_objects"

    def mock_resize_bbox(bbox, old_size, new_size):  # pylint: disable=unused-argument
      return (10, 10, 100, 100)

    output_path = mask_bbox_saver.save_cropped_objects(
        agg_features=agg_features,
        input_directory=input_directory,
        height_tracking=100,
        width_tracking=100,
        resize_bbox=mock_resize_bbox,
        output_suffix=output_suffix,
    )

    expected_output = os.path.basename(input_directory) + output_suffix
    self.assertEqual(output_path, expected_output)
    mock_makedirs.assert_called_once_with(expected_output, exist_ok=True)


if __name__ == "__main__":
  unittest.main()
