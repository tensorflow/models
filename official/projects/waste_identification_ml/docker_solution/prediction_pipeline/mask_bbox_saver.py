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

"""Processing and saving images with annotations from Mask R-CNN outputs.

It handles the extraction of bounding boxes and segmentation masks from the
output and saves these annotations in a visually interpretable format.
Additionally, it saves binary masks for each detected object.

The script performs two main functions:
1) It overlays bounding boxes and segmentation masks on the original image and
saves this annotated image.
2) It extracts and saves binary masks for each detected object in the image.
"""

import os
import sys
from typing import Any
import cv2
import numpy as np

sys.path.append('models/research/')
from object_detection.utils import visualization_utils as viz_utils  # pylint: disable=g-import-not-at-top,g-bad-import-order


def save_bbox_masks_labels(
    result: dict[Any, np.ndarray],
    image: np.ndarray,
    file_name: str,
    folder: str,
    category_index: dict[int, dict[str, str]],
    threshold: float,
) -> None:
  """Saves an image with visualized bounding boxes, labels, and masks.

  This function takes the output from Mask R-CNN, copies the original image,
  and applies visualizations of detection boxes, classes, and scores.
  If available, it also applies segmentation masks. The result is an image that
  juxtaposes the original with the annotated version, saved to the specified
  folder.

  Args:
    result: The output from theMask RCNN model, expected to contain detection
      boxes, classes, scores, reframed detection masks, etc.
    image: The original image as a numpy array.
    file_name: The filename for saving the output image.
    folder: The folder path where the output image will be saved.
    category_index: A dictionary mapping class IDs to class labels.
    threshold: Value between 0 and 1 to filter out the prediction results.
  """
  image_new = image.copy()
  if 'detection_masks_reframed' in result:
    result['detection_masks_reframed'] = result[
        'detection_masks_reframed'
    ].astype(np.uint8)

  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_new,
      result['detection_boxes'][0],
      (result['detection_classes'] + 0).astype(int),
      result['detection_scores'][0],
      category_index=category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=100,
      min_score_thresh=threshold,
      agnostic_mode=False,
      instance_masks=result.get('detection_masks_reframed', None),
      line_thickness=2,
  )

  cv2.imwrite(
      os.path.join(folder, file_name),
      np.concatenate((image, image_new), axis=1),
  )


def save_binary_masks(
    result: dict[Any, np.ndarray], file_name: str, folder: str
) -> None:
  """Saves binary masks generated from object detection results.

  This function processes the binary mask data extracted from the results of
  Mask RCNN model and saves the combined binary masks as an image file. It
  creates a mask image by layering each individual mask found in the result and
  saves the final mask image to the specified location.

  Args:
    result: The output from the Mask RCNN model, expected to contain key
      'detection_masks_reframed'.
    file_name: The filename for saving the output mask image.
    folder: The folder path where the output mask image will be saved.
  """
  mask = np.zeros_like(result['detection_masks_reframed'][0])
  result['detection_masks_reframed'] = result[
      'detection_masks_reframed'
  ].astype(np.uint8)
  for i in result['detection_masks_reframed']:
    i = i * 255
    mask += i

  cv2.imwrite(os.path.join(folder, file_name), mask)
