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

"""This is a prediction script.

For sending images to a FastAPI server and receiving predictions from a Mask
R-CNN model. The FastAPI server, powered by Uvicorn, hosts the Mask R-CNN model
which performs instance segmentation.

The script leverages the 'requests' library to send HTTP POST requests, carrying
images to the FastAPI server. The server processes these images using the
Mask R-CNN model and returns prediction results which are then postprocessed.
"""

import json
from absl import flags
import numpy as np
import requests

_IMAGE_PATH = flags.DEFINE_string(
    'image_path', None, 'The path to an image for prediction purpose'
)

_PORT = flags.DEFINE_integer(
    'port', None, 'The port number to send the image to'
)


def send_image_for_prediction(
    image_path: str,
    port: int,
) -> tuple[list[dict[str, np.ndarray]], int]:
  """Send an image to a local prediction service and retrieve the predictions.

  Args:
    image_path: Path to the image to be predicted.
    port: Port number on the server end for sending an image for prediction.

  Returns:
    A list containing the list of prediction results and the HTTP status
    code.
  """

  url = f'http://localhost:{port}/predict'
  response = None
  try:
    with open(image_path, 'rb') as image_file:
      files = {'image': (image_path, image_file, 'image/png')}
      response = requests.post(url, files=files)
      response.raise_for_status()
      result = json.loads(response.json())
      result = result.get('predictions', [])[:2]
      return result, response.status_code
  except (requests.RequestException, json.JSONDecodeError) as e:
    print(f'An error occurred: {e}')
    return [], response.status_code if response else 500


if __name__ == '__main__':
  results, status_code = send_image_for_prediction(
      _IMAGE_PATH.value, _PORT.value
  )
  print(f'HTTP Status Code: {status_code}')
  print('Predictions from material model:', results[0]['num_detections'][0])
  print(
      'predictions from material form model:', results[1]['num_detections'][0]
  )
