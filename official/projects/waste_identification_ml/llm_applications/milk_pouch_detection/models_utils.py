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

"""Utility functions for milk pouch detection."""

from collections.abc import Mapping
import dataclasses
import pathlib
from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision


@dataclasses.dataclass(frozen=True)
class _BoundingBox:
  """A class representing a bounding box."""
  x1: float
  y1: float
  x2: float
  y2: float


def _box_area(box: _BoundingBox) -> float:
  """Calculates the area of a bounding box.

  Args:
    box: A _BoundingBox object.

  Returns:
    The area of the bounding box.
  """
  return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def _calculate_iou(
    box1: _BoundingBox,
    box2: _BoundingBox
) -> float:
  """Calculates the Intersection over Union (IoU) of two bounding boxes.

  Args:
    box1: The first bounding box in (x1, y1, x2, y2) format.
    box2: The second bounding box in (x1, y1, x2, y2) format.

  Returns:
    The IoU score, a float between 0.0 and 1.0.
  """
  # Determine the coordinates of the intersection rectangle
  x1 = max(box1.x1, box2.x1)
  y1 = max(box1.y1, box2.y1)
  x2 = min(box1.x2, box2.x2)
  y2 = min(box1.y2, box2.y2)

  # Calculate the area of intersection
  inter_area = max(0, x2 - x1) * max(0, y2 - y1)

  # Calculate the area of both bounding boxes
  box1_area = _box_area(box1)
  box2_area = _box_area(box2)

  # Calculate the area of the union
  union_area = box1_area + box2_area - inter_area

  # Compute the IoU score
  return inter_area / union_area if union_area != 0 else 0.0


def _is_contained(
    inner_box: _BoundingBox,
    outer_box: _BoundingBox,
    margin: int = 5,
) -> bool:
  """Checks if one bounding box is contained within another, with a margin.

  Args:
    inner_box: The bounding box that is potentially inside.
    outer_box: The bounding box that is potentially surrounding.
    margin: An optional pixel margin to allow for slight inaccuracies.

  Returns:
    True if the inner box is contained within the outer box, False
    otherwise.
  """
  return (
      inner_box.x1 >= outer_box.x1 - margin
      and inner_box.y1 >= outer_box.y1 - margin
      and inner_box.x2 <= outer_box.x2 + margin
      and inner_box.y2 <= outer_box.y2 + margin
  )


def filter_boxes_keep_smaller(
    data: Mapping[str, list[Any]],
    iou_threshold: float = 0.8,
    area_threshold: int | None = None,
    min_area: int = 1000,
    margin: int = 5,
) -> dict[str, list[Any]]:
  """Filters overlapping bounding boxes, preferentially keeping smaller ones.

  This function sorts boxes by area and iterates through them, discarding any
  box that has a high IoU with an already-kept box or is contained within one.
  This is useful for eliminating duplicate or redundant detections.

  Args:
      data: A dictionary containing 'boxes' and 'masks' lists.
      iou_threshold: The IoU value above which a box is considered an overlap.
      area_threshold: An optional maximum area to consider for a box.
      min_area: The minimum area required for a box to be kept.
      margin: The pixel margin used for the containment check.

  Returns:
      A dictionary with the filtered 'boxes' and their corresponding 'masks'.
  """
  # Check if the input data is valid
  bounding_boxes = [_BoundingBox(*b) for b in data['boxes']]

  areas = ([_box_area(b) for b in bounding_boxes])

  # Sort boxes from smallest to largest area
  sorted_indices = np.argsort(areas)
  sorted_bounding_boxes = [bounding_boxes[i] for i in sorted_indices]

  masks = np.array(data['masks'])
  sorted_masks = masks[sorted_indices]

  kept_boxes = []
  kept_masks = []
  kept_bounding_boxes_for_check = []

  for i, box in enumerate(sorted_bounding_boxes):
    current_area = _box_area(box)
    if (
        area_threshold is not None and current_area > area_threshold
    ) or current_area < min_area:
      continue

    keep = True
    for kept_box in kept_bounding_boxes_for_check:
      if _calculate_iou(box, kept_box) > iou_threshold or _is_contained(
          kept_box, box, margin
      ):
        keep = False
        break

    if keep:
      kept_boxes.append([box.x1, box.y1, box.x2, box.y2])
      kept_masks.append(sorted_masks[i])
      kept_bounding_boxes_for_check.append(box)

  return {'boxes': kept_boxes, 'masks': kept_masks}


def convert_boxes_cxcywh_to_xyxy(
    boxes: torch.Tensor, image_shape: tuple[int, int, int]
) -> np.ndarray:
  """Converts bounding boxes from center-based to corner-based format.

  Args:
    boxes: A tensor of bounding boxes in (cx, cy, w, h) format.
    image_shape: A tuple representing the image dimensions (h, w, c).

  Returns:
    A NumPy array of bounding boxes in (x1, y1, x2, y2) format.
  """
  h, w, _ = image_shape
  scale_factors = torch.tensor([w, h, w, h], device=boxes.device)
  scaled_boxes = boxes * scale_factors
  xyxy_boxes = torchvision.ops.box_convert(
      boxes=scaled_boxes, in_fmt='cxcywh', out_fmt='xyxy'
  )
  return xyxy_boxes.cpu().numpy().astype(int)


def initialize_coco_output(category_name: str) -> dict[str, list[Any]]:
  """Initializes the COCO format output structure.

  Args:
    category_name: Name of the object category.

  Returns:
    A dictionary with the COCO format structure.
  """
  return {
      'categories': [{
          'id': 1,
          'name': category_name,
          'supercategory': 'object',
      }],
      'images': [],
      'annotations': [],
  }


def plot_prediction(
    image_path: pathlib.Path, pred_class: str, pred_prob: float
):
  """Plots the original image with its prediction and probability.

  Args:
    image_path: Path to the input image file.
    pred_class: The predicted class name for the image.
    pred_prob: The predicted probability for the class.
  """
  img = Image.open(image_path)
  plt.figure()
  plt.imshow(img)
  plt.title(f'Pred: {pred_class} | Prob: {pred_prob:.3f}%')
  plt.axis(False)
  plt.show()


def extract_largest_contour_segmentation(mask: np.ndarray) -> list[float]:
  """Extracts the largest external contour from a binary mask.

  This function finds all external contours in a binary mask and returns
  the flattened coordinate list of the largest valid contour.

  Args:
      mask: A binary mask image (2D array) where the object is marked with 1s or
        255s. Shape should be (height, width).

  Returns:
      A list containing a single flattened contour with coordinates in the
      format [x1, y1, x2, y2, ...]. Returns an empty list if no valid
      contour is found.

  Examples:
      >>> mask = np.zeros((100, 100), dtype=np.uint8)
      >>> mask[20:80, 20:80] = 1
      >>> segmentation = extract_largest_contour_segmentation(mask)
  """
  mask_uint8 = mask.astype(np.uint8)

  contours, _ = cv2.findContours(
      mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  if not contours:
    return []

  valid_segmentations = []
  for contour in contours:
    flattened = contour.flatten().tolist()
    # Need at least 3 points (6 coordinates) for a valid polygon
    if len(flattened) >= 6:
      valid_segmentations.append(flattened)

  if not valid_segmentations:
    return []

  return [max(valid_segmentations, key=len)]


def get_bbox_details(box: list[int]) -> tuple[int, int, int]:
  """Calculates width, height, and area from bounding box coordinates.

  Args:
      box: Bounding box coordinates in the format [x1, y1, x2, y2], where (x1,
        y1) is the top-left corner and (x2, y2) is the bottom-right corner.

  Returns:
      A tuple containing (width, height, area) of the bounding box.

  Examples:
      >>> box = [10, 20, 50, 80]
      >>> width, height, area = get_bbox_details(box)
      >>> print(f"Width: {width}, Height: {height}, Area: {area}")
      Width: 40, Height: 60, Area: 2400
  """
  x1, y1, x2, y2 = box

  bbox_width = x2 - x1
  bbox_height = y2 - y1
  bbox_area = bbox_width * bbox_height

  return bbox_width, bbox_height, bbox_area
