# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""FastAPI Server for Image Predictions with Uvicorn.

This script sets up a FastAPI server that uses 2 trained Mask RCNN instance
segmentation models to predict objects present in uploaded images.
The results of the predictions are serialized into a JSON format and returned
to the client.

The server utilizes Uvicorn, an ASGI server, to serve FastAPI applications.
The setup is intended to be containerized using Docker and subsequently deployed
on a VM instance at the client's side.
"""

import io
import json
import fastapi
import PIL
import tensorflow as tf, tf_keras
import uvicorn
from official.projects.waste_identification_ml.docker_solution.prediction_api import app_utils


HEIGHT, WIDTH = 512, 1024

app = fastapi.FastAPI()
model_manager = app_utils.ModelManager()


@app.on_event('startup')
def startup_event():
  model_manager.load_all_models()


@app.post('/predict')
async def predict(
    image: fastapi.UploadFile = fastapi.File(default=None),
) -> fastapi.responses.JSONResponse:
  """Predicts objects in the uploaded image.

  Args:
    image: Image from which to generate predictions.

  Returns:
    A JSON encoded list of detections.
  """
  image_data = await image.read()
  try:
    p_image = PIL.Image.open(io.BytesIO(image_data))
  except (OSError, PIL.UnidentifiedImageError):
    return fastapi.responses.JSONResponse(
        content={'message': 'Could not open image_data as an image.'},
        status_code=400,
    )  # Bad Request

  try:
    tf_image = tf.image.resize(
        p_image, (HEIGHT, WIDTH), method=tf.image.ResizeMethod.AREA
    )
    image_cp = tf.cast(tf_image, tf.uint8)
    image = app_utils.preprocess_image(image_cp)
    detections = list(
        map(
            lambda model: app_utils.perform_detection(model, image),
            model_manager.detection_fns,
        )
    )

    json_dump = json.dumps(
        {'predictions': detections}, cls=app_utils.NumpyEncoder
    )
    return fastapi.responses.JSONResponse(content=json_dump)

  except TypeError:
    return fastapi.responses.JSONResponse(
        content={'message': 'Image data is not in the correct format.'},
        status_code=422,
    )  # Unprocessable Entity


if __name__ == '__main__':
  uvicorn.run(app, host='0.0.0.0', port=5000)
