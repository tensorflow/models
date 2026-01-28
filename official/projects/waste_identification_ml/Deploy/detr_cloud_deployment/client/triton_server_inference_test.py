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

from official.projects.waste_identification_ml.Deploy.detr_cloud_deployment.client import triton_server_inference

MODULE_PATH = triton_server_inference.__name__


class TritonObjectDetectorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock triton client
    self.mock_http_client_patch = mock.patch(
        f"{MODULE_PATH}.httpclient.InferenceServerClient"
    )
    self.mock_http_client = self.mock_http_client_patch.start()
    self.mock_client_instance = self.mock_http_client.return_value

    # Mock pd.read_csv for labels
    self.mock_pd_read_csv_patch = mock.patch(f"{MODULE_PATH}.pd.read_csv")
    self.mock_pd_read_csv = self.mock_pd_read_csv_patch.start()
    self.mock_pd_read_csv.return_value = pd.DataFrame(
        {"id": [1, 2], "names": ["Class1", "Class2"]}
    )

    # Mock os.path.join to avoid filesystem access for labels
    self.mock_os_path_join_patch = mock.patch(f"{MODULE_PATH}.os.path.join")
    self.mock_os_path_join = self.mock_os_path_join_patch.start()
    self.mock_os_path_join.return_value = "dummy_path/labels50.csv"

    self.detector = triton_server_inference.TritonObjectDetector(
        server_url="test:8000", model_name="test_model", input_size=(100, 100)
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_init(self):
    # Assert
    self.mock_http_client.assert_called_once_with(
        url="test:8000", verbose=False
    )
    self.assertEqual(self.detector.model_name, "test_model")
    self.assertEqual(self.detector.input_size, (100, 100))

  def test_sigmoid(self):
    # Arrange
    x = np.array([-1.0, 0.0, 1.0])

    # Act
    result = triton_server_inference._sigmoid(x)

    # Assert
    np.testing.assert_allclose(result, [0.26894142, 0.5, 0.73105858])

  def test_box_cxcywh_to_xyxyn(self):
    # Arrange
    x = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 1.0, 1.0]])

    # Act
    result = triton_server_inference._box_cxcywh_to_xyxyn(x)

    # Assert
    expected = np.array([[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 1.0, 1.0]])
    np.testing.assert_allclose(result, expected)

  @mock.patch(f"{MODULE_PATH}.cv2.resize")
  def test_scale_bbox_and_masks(self, mock_cv2_resize):
    # Arrange
    mock_cv2_resize.return_value = np.ones(
        (20, 10)
    )  # h=20, w=10. cv2 resize returns h, w

    results = {
        "xyxy": np.array([[0.1, 0.1, 0.5, 0.5]]),
        "masks": np.ones((1, 5, 5), dtype=np.uint8),
    }
    target_dims = (10, 20)  # w=10, h=20

    # Act
    scaled_results = self.detector._scale_bbox_and_masks(results, target_dims)

    # Assert
    np.testing.assert_allclose(scaled_results["xyxy"], [[1.0, 2.0, 5.0, 10.0]])
    self.assertEqual(scaled_results["masks"].shape, (1, 20, 10))
    self.assertEqual(scaled_results["masks"].dtype, bool)
    mock_cv2_resize.assert_called_once()
    args, kwargs = mock_cv2_resize.call_args
    np.testing.assert_array_equal(args[0], np.ones((5, 5, 1)))
    self.assertEqual(args[1], (10, 20))
    self.assertEqual(
        kwargs["interpolation"], triton_server_inference.cv2.INTER_NEAREST
    )

  @mock.patch(f"{MODULE_PATH}.cv2.imread")
  @mock.patch(f"{MODULE_PATH}.cv2.cvtColor")
  @mock.patch(f"{MODULE_PATH}.cv2.resize")
  def test_get_input_batch_for_inference_success(
      self, mock_resize, mock_cvtcolor, mock_imread
  ):
    # Arrange
    mock_imread.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = "dummy.jpg"

    # Act
    processed_image = self.detector._get_input_batch_for_inference(image_path)

    # Assert
    mock_imread.assert_called_once_with(image_path)
    mock_cvtcolor.assert_called_once()
    mock_resize.assert_called_once_with(
        mock_cvtcolor.return_value, (100, 100), interpolation=mock.ANY
    )
    self.assertEqual(processed_image.shape, (1, 3, 100, 100))
    self.assertEqual(processed_image.dtype, np.float32)

  @mock.patch(f"{MODULE_PATH}.cv2.imread")
  def test_get_input_batch_for_inference_filenotfound(self, mock_imread):
    # Arrange
    mock_imread.return_value = None
    image_path = "nonexistent.jpg"

    # Act & Assert
    with self.assertRaises(FileNotFoundError):
      self.detector._get_input_batch_for_inference(image_path)

  def test_reformat_triton_output_to_dict(self):

    raw_boxes = np.array([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    raw_probs_logits = np.array([[[-1.0, 3.0, 0.0], [-2.0, -3.0, -4.0]]])
    masks = np.zeros((1, 2, 10, 10))
    outputs = [raw_boxes, raw_probs_logits, masks]

    # Act
    # max_boxes=2, threshold=0.5 -> box 0 should be kept, box 1 filtered
    results = self.detector._reformat_triton_output_to_dict(
        outputs, confidence_threshold=0.5, max_boxes=2
    )

    # Assert
    expected_results = {
        "confidence": np.array([0.95257413]),
        "labels": np.array([1]),
        "xyxy": np.array([[0.4, 0.4, 0.6, 0.6]]),
        "masks": np.zeros((1, 10, 10)),
    }
    self.assertCountEqual(results.keys(), expected_results.keys())
    np.testing.assert_allclose(
        results["confidence"], expected_results["confidence"]
    )
    np.testing.assert_array_equal(results["labels"], expected_results["labels"])
    np.testing.assert_allclose(results["xyxy"], expected_results["xyxy"])
    np.testing.assert_array_equal(results["masks"], expected_results["masks"])

  @mock.patch(
      f"{MODULE_PATH}.TritonObjectDetector._get_input_batch_for_inference"
  )
  @mock.patch(
      f"{MODULE_PATH}.TritonObjectDetector._reformat_triton_output_to_dict"
  )
  @mock.patch(f"{MODULE_PATH}.TritonObjectDetector._scale_bbox_and_masks")
  @mock.patch(f"{MODULE_PATH}.httpclient.InferInput")
  def test_predict(
      self,
      mock_infer_input,
      mock_scale,
      mock_reformat_triton_output_to_dict,
      mock_get_input_batch_for_inference,
  ):
    # Arrange
    image_path = "dummy.jpg"
    mock_get_input_batch_for_inference.return_value = np.zeros(
        (1, 3, 100, 100), dtype=np.float32
    )
    mock_input_instance = mock.Mock()
    mock_infer_input.return_value = mock_input_instance

    mock_response = mock.Mock()
    mock_response.as_numpy.side_effect = [
        np.array([1]),
        np.array([2]),
        np.array([3]),
    ]
    self.mock_client_instance.infer.return_value = mock_response

    post_process_result = {"xyxy": np.array([[1, 1, 2, 2]]), "masks": None}
    mock_reformat_triton_output_to_dict.return_value = post_process_result
    scale_result = {
        "xyxy": np.array([[10, 10, 20, 20]]),
        "masks": None,
        "some_key": 1,
    }
    mock_scale.return_value = scale_result

    # Act
    results = self.detector.predict(
        image_path,
        confidence_threshold=0.7,
        max_boxes=50,
        output_dims=(200, 200),
    )

    # Assert
    mock_get_input_batch_for_inference.assert_called_once_with(image_path)
    mock_infer_input.assert_called_once_with(
        "input", (1, 3, 100, 100), datatype="FP32"
    )
    mock_input_instance.set_data_from_numpy.assert_called_once_with(
        mock_get_input_batch_for_inference.return_value, binary_data=True
    )
    self.mock_client_instance.infer.assert_called_once_with(
        model_name="test_model", inputs=[mock_input_instance]
    )
    mock_reformat_triton_output_to_dict.assert_called_once_with(
        [np.array([1]), np.array([2]), np.array([3])], 0.7, 50
    )
    mock_scale.assert_called_once_with(post_process_result, (200, 200))
    self.assertEqual(results, scale_result)

  def test_get_class_id_to_class_name_mapping(self):
    mapper = self.detector._get_class_id_to_class_name_mapping()

    self.mock_pd_read_csv.assert_called_once_with("dummy_path/labels50.csv")

    self.assertEqual(mapper, {1: "Class1", 2: "Class2"})

  def test_get_class_names(self):
    results = {"labels": np.array([0, 1, 99])}

    class_names = self.detector.get_class_names(results)

    np.testing.assert_array_equal(class_names, ["Class1", "Class2", "None"])


if __name__ == "__main__":
  unittest.main()
