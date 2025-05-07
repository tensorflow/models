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
from unittest import mock
import numpy as np
# Import the functions to be tested
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import triton_server_prediction


class TestTritonPrediction(unittest.TestCase):

  @mock.patch("cv2.imread")
  @mock.patch("cv2.cvtColor")
  @mock.patch("cv2.resize")
  def test_model_input(self, mock_resize, mock_convert_color, mock_imread):
    """Test the model_input function."""

    # Mocking image loading and processing
    mock_imread.return_value = np.ones((500, 500, 3), dtype=np.uint8)
    mock_convert_color.return_value = np.ones((500, 500, 3), dtype=np.uint8)
    mock_resize.return_value = np.ones((224, 224, 3), dtype=np.uint8)

    _, image, image_resized = triton_server_prediction.model_input(
        "dummy_path.jpg", 224, 224
    )

    self.assertEqual(image.shape, (500, 500, 3))
    self.assertEqual(image_resized.shape, (224, 224, 3))

  @mock.patch("your_module._query_model")
  def test_prediction(self, mock_query_model):
    """Test the prediction function."""

    mock_result = mock.MagicMock()
    mock_result.as_numpy.side_effect = lambda key: np.array(
        [1]
    )
    mock_query_model.return_value = mock_result

    mock_inputs = mock.MagicMock()
    model_name = "dummy_model"

    result = triton_server_prediction.prediction(model_name, mock_inputs)

    for key in (
        "detection_classes",
        "detection_masks",
        "detection_boxes",
        "image_info",
        "num_detections",
        "detection_scores",
    ):
      self.assertIn(key, result)
      self.assertIsInstance(result[key], np.ndarray)


if __name__ == "__main__":
  unittest.main()
