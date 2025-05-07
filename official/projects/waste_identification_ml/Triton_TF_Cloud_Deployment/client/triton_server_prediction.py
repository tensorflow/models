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

"""Prediction from the Triton server."""

from typing import Any
import cv2
import numpy as np
import tritonclient

_API_URL = 'localhost:8000'
_OUTPUT_KEYS = (
    'detection_classes',
    'detection_masks',
    'detection_boxes',
    'image_info',
    'num_detections',
    'detection_scores',
)

# Setting up the Triton client
_TRITON_CLIENT = tritonclient.http.InferenceServerClient(
    url=_API_URL, network_timeout=1200, connection_timeout=1200
)

# Outputs setup based on constants
_OUTPUTS = [
    tritonclient.http.InferRequestedOutput(key, binary_data=True)
    for key in _OUTPUT_KEYS
]


def model_input(
    path: str, height: int, width: int
) -> tritonclient.http.InferInput:
  """Prepares an image for input to a Triton model server.

  It reads it from a path, resizes it, normalizes it, and converts it to the
  format required by the server.

  Args:
      path: The file path to the image that needs to be processed.
      height: The height of the image to be resized.
      width: The width of the image to be resized.

  Returns:
      A Triton inference server input object containing the processed image.
  """
  original_image = cv2.imread(path)
  image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
  image_resized = cv2.resize(
      image, (width, height), interpolation=cv2.INTER_AREA
  )
  expanded_image = np.expand_dims(image_resized, axis=0)
  inputs = tritonclient.http.InferInput(
      'inputs', expanded_image.shape, datatype='UINT8'
  )
  inputs.set_data_from_numpy(expanded_image, binary_data=True)
  return inputs, image, image_resized


def _query_model(
    client: tritonclient.http.InferenceServerClient,
    model_name: str,
    inputs: tritonclient.http.InferInput,
) -> tritonclient.http.InferResult:
  """Sends an inference request to the Triton server.

  Args:
      client: The Triton server client.
      model_name: Name of the model for which inference is requested.
      inputs: The input data for inference.

  Returns:
      The result of the inference request.
  """
  return client.infer(model_name=model_name, inputs=[inputs], outputs=_OUTPUTS)


def prediction(
    model_name: str, inputs: tritonclient.http.InferInput
) -> dict[str, Any]:
  """Model name for prediction.

  Args:
      model_name: Model name in Triton Server.
      inputs: The input data for inference.

  Returns:
      prediction output from the model.
  """
  result = _query_model(_TRITON_CLIENT, model_name, inputs)
  result_dict = {key: result.as_numpy(key) for key in _OUTPUT_KEYS}
  return result_dict
