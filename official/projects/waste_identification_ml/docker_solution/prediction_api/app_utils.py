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

"""Model manager for the server."""

import json
import logging
import logging.config
import sys
import types
from typing import Any, Callable
import numpy as np
import tensorflow as tf, tf_keras

# sys.path.append is used as preprocessing.py. Will be imported after cloning.
# 'tensorflow_models' project from github
sys.path.append(
    'models/official/projects/waste_identification_ml/model_inference/'
)
from official.projects.waste_identification_ml.model_inference import preprocessing  # pylint: disable=g-import-not-at-top,g-bad-import-order

MODELS_DIR_PATH = types.MappingProxyType({
    'material_model': 'material/saved_model/',
    'material_form_model': 'material_form/saved_model/',
})


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        'absl': {'level': 'ERROR'},
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        }
    },
    'root': {'level': 'INFO', 'propagate': False, 'handlers': ['default']},
})

logger = logging.getLogger(__name__)


class ModelManager:
  """Manages all models for the server.

  This class is responsible for loading and managing TensorFlow models that are
  used for object detection. It provides mechanisms to load models and perform
  detections with them.

  Attributes:
    detection_fns: The detection functions loaded from TensorFlow SavedModels.
  """

  def __init__(self):
    # Initializes an empty list to hold the model detection functions.
    self.detection_fns = []

  def load_model(
      self,
      model_handle: str,
  ) -> Callable[[tf.Tensor], dict[str, np.ndarray]]:
    """Loads a TensorFlow SavedModel and returns a function for predictions.

    Args:
      model_handle: A path to a TensorFlow SavedModel.

    Returns:
      A function that can be used to make predictions.
    """
    with tf.device('GPU:0'):
      logger.info('loading model...')
      model = tf.saved_model.load(model_handle)
      logger.info('model loaded!')
      detection_fn = model.signatures['serving_default']
      return detection_fn

  def load_all_models(self):
    logger.info('Loading all models!')
    self.detection_fns = [
        self.load_model(value) for value in MODELS_DIR_PATH.values()
    ]
    logger.info('Models loaded!')


class NumpyEncoder(json.JSONEncoder):
  """JSON Encoder for NumPy types.

  This encoder can be used with json.dump() and json.dumps()
  to serialize NumPy arrays that the standard JSON encoder
  cannot handle.
  """

  def default(self, o: Any) -> Any:
    """Override the default() method to handle NumPy arrays.

    Args:
      o: The object to be encoded.

    Returns:
      The encoded object.
    """
    if isinstance(o, np.ndarray):
      return o.tolist()
    return super().default(o)


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
  """Builds segmentation model inputs for serving.

  Args:
    image: Image to be normalized

  Returns:
    A normalized image
  """
  image = preprocessing.normalize_image(image)
  image = tf.expand_dims(image, axis=0)
  return image


def perform_detection(
    model: Callable[[tf.Tensor], dict[str, np.ndarray]], image: tf.Tensor
) -> dict[str, np.ndarray]:
  """Performs Mask RCNN on an image using the specified model.

  Args:
    model: A function that can be used to make predictions.
    image: Image to be detected.

  Returns:
     detection: A dictionary that contains detection information such as
     bounding boxes, classes, and scores, etc.
  """
  detection = model(image)
  detection = {key: value.numpy() for key, value in detection.items()}
  return detection

