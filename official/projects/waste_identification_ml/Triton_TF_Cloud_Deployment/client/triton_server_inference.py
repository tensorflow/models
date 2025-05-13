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
from tritonclient import grpc as triton_grpc


_OUTPUT_KEYS = (
    'detection_classes',
    'detection_masks',
    'detection_boxes',
    'image_info',
    'num_detections',
    'detection_scores',
)
_OUTPUTS = tuple(triton_grpc.InferRequestedOutput(key) for key in _OUTPUT_KEYS)


def prepare_image(
    path: str, height: int, width: int
) -> tuple[triton_grpc.InferInput, np.ndarray, np.ndarray]:
  """Prepares an image and converts it to an input for a Triton model server.

  Args:
      path: The file path to the image that needs to be processed.
      height: The height of the image to be resized.
      width: The width of the image to be resized.

  Returns:
      A tuple with the triton InferInput and both the original and resized
      image.
  """
  image_bgr = cv2.imread(path)
  image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  image_resized = cv2.resize(
      image, (width, height), interpolation=cv2.INTER_AREA
  )
  expanded_image = np.expand_dims(image_resized, axis=0)
  inputs = triton_grpc.InferInput(
      'inputs', expanded_image.shape, datatype='UINT8'
  )
  inputs.set_data_from_numpy(expanded_image)
  return inputs, image, image_resized


def infer(
    model_name: str, inputs: triton_grpc.InferInput
) -> dict[str, Any]:
  """Wraps inference and converts the result to a dictionary of output keys.

  Args:
      model_name: Model name in Triton Server.
      inputs: The input data for inference.

  Returns:
      A dictionary of output keys and their corresponding values from
      InferResult.
  """
  result = triton_grpc.InferenceServerClient(url='localhost:8001').infer(
      model_name=model_name, inputs=[inputs], outputs=_OUTPUTS
  )
  if result:
    return {key: result.as_numpy(key) for key in _OUTPUT_KEYS}

