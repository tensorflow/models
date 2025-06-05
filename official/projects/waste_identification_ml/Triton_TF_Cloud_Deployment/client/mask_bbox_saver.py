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

"""Processing and saving images with annotations from Mask R-CNN outputs.

It handles the extraction of bounding boxes and segmentation masks from the
output and saves these annotations in a visually interpretable format.
Additionally, it saves binary masks for each detected object.

The script performs two main functions:
1) It overlays bounding boxes and segmentation masks on the original image and
saves this annotated image.
2) It extracts and saves binary masks for each detected object in the image.
"""

from collections.abc import Mapping
import dataclasses
import os
from typing import Any, Callable, Dict

import cv2
import numpy as np
import pandas as pd

from official.vision.utils.object_detection import visualization_utils as viz_utils

CIRCLE_RADIUS = 7
CIRCLE_COLOR = (255, 133, 233)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.0
TEXT_COLOR = (255, 0, 0)
TEXT_THICKNESS = 2
TEXT_LINE_TYPE = cv2.LINE_AA


@dataclasses.dataclass
class BoundingBox:
  y1: int | float
  x1: int | float
  y2: int | float
  x2: int | float


@dataclasses.dataclass
class ImageSize:
  height: int
  width: int


def save_bbox_masks_labels(
    *,
    result: Mapping[Any, np.ndarray],
    image: np.ndarray,
    file_name: str,
    folder: str,
    category_index: Dict[int, Dict[str, str]],
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
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_new,
      result['normalized_boxes'][0],
      (result['detection_classes'][0] + 0).astype(int),
      result['detection_scores'][0],
      category_index=category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=100,
      min_score_thresh=threshold,
      agnostic_mode=False,
      instance_masks=result.get('detection_masks_reframed', None),
      line_thickness=4,
  )

  cv2.imwrite(
      os.path.join(folder, file_name),
      np.concatenate((image, image_new), axis=1),
  )


def save_binary_masks(
    result: Dict[Any, np.ndarray], file_name: str, folder: str
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
  mask = np.zeros_like(result['detection_masks_reframed'][0], dtype=np.uint8)

  # Process and accumulate binary masks
  for single_mask in result['detection_masks_reframed']:
    mask += single_mask.astype(np.uint8) * 255  # Convert to 0-255 range

  cv2.imwrite(os.path.join(folder, file_name), mask)


# TODO(umairsabir): Add helper function to remove nested loop.
def visualize_tracking_results(
    tracking_features: pd.DataFrame,
    tracking_images: Mapping[str, np.ndarray],
    tracking_folder: str,
) -> str:
  """Draws tracking results on images and saves them to an output folder.

  Args:
      tracking_features: DataFrame with columns ['image_name', 'x', 'y',
        'particle'].
      tracking_images: Mapping from image_name to image (numpy array).
      tracking_folder: Directory where tracking results are saved.

  Returns:
      str:Path to the output folder where annotated images are saved.
  """
  groups = tracking_features.groupby('image_name')
  for name, group in groups:
    img = tracking_images[name].copy()
    for k in range(len(group)):
      x, y = int(group.iloc[k]['x']), int(group.iloc[k]['y'])
      cv2.circle(img, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, -1)
      cv2.putText(
          img,
          str(int(group.iloc[k]['particle'])),
          (x, y),
          TEXT_FONT,
          TEXT_SCALE,
          TEXT_COLOR,
          TEXT_THICKNESS,
          TEXT_LINE_TYPE,
      )
    cv2.imwrite(os.path.join(tracking_folder, str(name)), img)
  return tracking_folder


def save_cropped_objects(
    agg_features: pd.DataFrame,
    input_directory: str,
    height_tracking: int,
    width_tracking: int,
    resize_bbox: Callable[
        [BoundingBox, ImageSize, ImageSize], tuple[int, int, int, int]
    ],
    output_suffix: str = '_cropped_objects',
) -> str:
  """Saves cropped object images by category from tracking results.

  Args:
      agg_features: DataFrame containing grouped tracking results.
      input_directory: The directory with original images.
      height_tracking: Height used during tracking.
      width_tracking: Width used during tracking.
      resize_bbox: Function to resize bounding box.
      output_suffix: Suffix for cropped objects folder.

  Returns:
      str: Path to the output folder where cropped images are saved.
  """
  cropped_obj_folder = os.path.basename(input_directory) + output_suffix
  os.makedirs(cropped_obj_folder, exist_ok=True)

  if agg_features.empty:
    return cropped_obj_folder

  for group_name, df in agg_features.groupby('detection_classes_names'):
    class_folder = os.path.join(cropped_obj_folder, str(group_name))
    os.makedirs(class_folder, exist_ok=True)

    for row in df.itertuples(index=False):
      image = cv2.imread(
          os.path.join(os.path.basename(input_directory), row.image_name)
      )
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      new_h, new_w = image_rgb.shape[0], image_rgb.shape[1]

      y1, x1, y2, x2 = row.bbox_0, row.bbox_1, row.bbox_2, row.bbox_3
      bbox = BoundingBox(y1, x1, y2, x2)
      new_bbox = resize_bbox(
          bbox,
          ImageSize(height=height_tracking, width=width_tracking),
          ImageSize(height=new_h, width=new_w),
      )

      score = getattr(row, 'detection_scores', 0.0)
      name = f'{os.path.splitext(row.image_name)[0]}_{row.particle}_{score:.2f}.png'
      crop = image_rgb[new_bbox[0] : new_bbox[2], new_bbox[1] : new_bbox[3]]

      cv2.imwrite(os.path.join(class_folder, name), crop)

  return cropped_obj_folder
