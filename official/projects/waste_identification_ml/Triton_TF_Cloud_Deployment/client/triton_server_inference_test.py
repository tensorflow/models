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
import cv2
import numpy as np
from tritonclient import grpc as triton_grpc
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import triton_server_inference

# Create a small 1x4 BGR (open cv default) test image
BGR_TEST_IMAGE = np.zeros((1, 4, 3), dtype=np.uint8)
BGR_TEST_IMAGE[0, 0] = [0, 0, 255]  # Red in BGR
BGR_TEST_IMAGE[0, 1] = [0, 255, 0]
BGR_TEST_IMAGE[0, 2] = [255, 0, 0]  # Blue in BGR
BGR_TEST_IMAGE[0, 3] = [0, 255, 255]


class TestTritonPrediction(unittest.TestCase):

  @mock.patch.object(cv2, 'imread')
  def test_input_conversion_to_rgb(self, mock_imread):
    mock_imread.return_value = BGR_TEST_IMAGE

    _, test_image, _ = (
        triton_server_inference.prepare_image('/path/test_img.jpg', 5, 5)
    )

    # Check that a single BRG pixel is converted to RGB
    self.assertEqual(test_image[0, 0].tolist(), [255, 0, 0])

  @mock.patch.object(cv2, 'imread')
  def test_input_image_resized(self, mock_imread):
    mock_imread.return_value = BGR_TEST_IMAGE

    _, _, test_image_resized = (
        triton_server_inference.prepare_image('/path/test_img.jpg', 5, 5)
    )

    self.assertEqual(test_image_resized.shape, (5, 5, 3))

  @mock.patch.object(cv2, 'imread')
  def test_batch_dimension_prepended_to_triton_input(self, mock_imread):
    mock_imread.return_value = BGR_TEST_IMAGE

    test_triton_input, _, _ = (
        triton_server_inference.prepare_image('/path/test_img.jpg', 5, 5)
    )

    self.assertEqual(test_triton_input.shape(), [1, 5, 5, 3])

  @mock.patch.object(cv2, 'imread')
  def test_image_converted_to_infer_input(self, mock_imread):
    mock_imread.return_value = BGR_TEST_IMAGE

    test_triton_input, _, _ = (
        triton_server_inference.prepare_image('/path/test_img.jpg', 5, 5)
    )

    self.assertIsInstance(test_triton_input, triton_grpc.InferInput)

  @mock.patch.object(triton_grpc.InferInput, 'set_data_from_numpy')
  @mock.patch.object(cv2, 'imread')
  def test_infer_input_set(self, mock_imread, mock_set_data_from_numpy):
    mock_imread.return_value = BGR_TEST_IMAGE

    triton_server_inference.prepare_image('/path/test_img.jpg', 5, 5)

    # Check that the set_data_from_numpy method is called once. Triton
    # InferInput data is a black-box, so we just check that it was set.
    mock_set_data_from_numpy.assert_called_once()

  @mock.patch.object(triton_grpc.InferenceServerClient, 'infer')
  def test_inference_output_converted_to_dict(self, mock_query_model):
    test_output_data = np.array([[1, 0]])
    mock_infer_result = mock.create_autospec(
        triton_grpc.InferResult, instance=True
    )
    mock_infer_result.as_numpy = lambda key: test_output_data
    mock_query_model.return_value = mock_infer_result

    result = triton_server_inference.infer('test_model', mock.MagicMock())

    for key in triton_server_inference._OUTPUT_KEYS:
      self.assertIn(key, result)
      self.assertIsInstance(result[key], np.ndarray)


if __name__ == '__main__':
  unittest.main()
