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

"""This script is tailored for processing outputs from two Mask R-CNN models.

It is designed to handle object detection and segmentation tasks, combines
outputs from two Mask R-CNN models. This involves aggregating detected objects
and their respective masks and bounding boxes. Identifies and removes duplicate
detections in the merged result, ensuring each detected object is unique.
Extracts and compiles features of the detected objects, which may include
aspects like size, area, color, or other model-specific attributes.
"""

import sys
import numpy as np

sys.path.append(
    'models/official/projects/waste_identification_ml/model_inference/'
)
from official.projects.waste_identification_ml.model_inference import postprocessing  # pylint: disable=g-import-not-at-top,g-bad-import-order

HEIGHT, WIDTH = 512, 1024


def merge_predictions(
    results: list[dict[str, np.ndarray]],
    score: float,
    category_indices: list[list[str]],
    category_index: dict[int, dict[str, str]],
    max_detection: int,
) -> dict[str, np.ndarray]:
  """Merges and refines prediction results.

  This function takes the prediction results from two models, reframes masks to
  the original image size, and aligns similar masks from both model outputs. It
  then merges these masks into a single result based on the given threshold
  criteria. The criteria include a minimum score threshold, an area threshold,
  and category alignment using provided indices and dictionary.

  Args:
    results: Outputs from 2 Mask RCNN models.
    score: The minimum score threshold for filtering out the detections.
    category_indices: Class labels of 2 models.
    category_index: A dictionary mapping class IDs to class labels.
    max_detection: Maximum number of detections from both models.

  Returns:
    Merged and filtered detection results.
  """
  # This threshold will be used to eliminate all the detected objects whose
  # area is greater than the 'area_threshold'.
  area_threshold = 0.3 * HEIGHT * WIDTH

  # Reframe the masks from the output of the model to its original size.
  results_reframed = [
      postprocessing.reframing_masks(detection, HEIGHT, WIDTH)
      for detection in results
  ]

  # Align similar masks from both the model outputs and merge all the
  # properties into a single mask. Function will only compare first
  # 'max_detection' objects. All the objects which have less than
  # 'score' probability will be eliminated. All objects whose area is
  # more than 'area_threshold' will be eliminated. 'category_dict' and
  # 'category_index' are used to find the label from the combinations of
  # labels from both individual models. The output should include masks
  # appearing in either of the models if they qualify the criteria.
  final_result = postprocessing.find_similar_masks(
      results_reframed[0],
      results_reframed[1],
      max_detection,
      score,
      category_indices,
      category_index,
      area_threshold,
  )
  return final_result


def _transform_bounding_boxes(
    results: dict[str, np.ndarray]
) -> list[list[int]]:
  """Transforms normalized bounding box coordinates to their original format.

  This function takes a dictionary containing normalized bounding box
  coordinates and transforms these coordinates to their original scale based on
  the provided image height and width.

  Args:
    results: A dictionary containing detection results. Expected to have a key
      'detection_boxes' with a numpy array of normalized coordinates.

  Returns:
    A list of transformed bounding boxes, each represented as [ymin, xmin, ymax,
    xmax] in the original image scale.
  """
  transformed_boxes = []
  for bb in results['detection_boxes'][0]:
    ymin = int(bb[0] * HEIGHT)
    xmin = int(bb[1] * WIDTH)
    ymax = int(bb[2] * HEIGHT)
    xmax = int(bb[3] * WIDTH)
    transformed_boxes.append([ymin, xmin, ymax, xmax])
  return transformed_boxes
