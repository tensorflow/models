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

"""Handles COCO JSON file creation for training dataset preparation.

This module provides utilities for creating COCO JSON annotations
from detected objects, to be used for training or finetuning object detection
and segmentation models.
"""

import json
import os
from typing import List

import numpy as np

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import models_utils


class CocoAnnotationWriter:
  """Manages creation and writing of COCO-format annotations.

  This class handles the incremental building of COCO JSON annotations as
  images are processed, including image metadata and object annotations with
  bounding boxes and segmentation masks.

  Attributes:
    category_name: Name of the object category for annotations.
    coco_output: Dictionary containing COCO format data structure.
    annotation_id_counter: Counter for generating unique annotation IDs.
    image_id_counter: Counter for tracking processed images.
  """

  def __init__(self, category_name: str):
    """Initializes the COCO annotation writer.

    Args:
      category_name: Name of the category (e.g., 'packets', 'dairy').
    """
    self.category_name = category_name
    self.coco_output = models_utils.initialize_coco_output(category_name)
    self.annotation_id_counter = 0
    self.image_id_counter = 0

  def add_image(
      self,
      file_path: str,
      width: int,
      height: int,
  ) -> int:
    """Adds image metadata to COCO output.

    Args:
      file_path: Path to the image file.
      width: Image width in pixels.
      height: Image height in pixels.

    Returns:
      The image ID assigned to this image.
    """
    image_info = {
        "id": self.image_id_counter,
        "file_name": os.path.basename(file_path),
        "width": width,
        "height": height,
    }
    self.coco_output["images"].append(image_info)
    current_image_id = self.image_id_counter
    self.image_id_counter += 1
    return current_image_id

  def add_annotations(
      self,
      image_id: int,
      boxes: List[np.ndarray],
      masks: List[np.ndarray],
  ) -> int:
    """Adds object annotations for a single image.

    Args:
      image_id: ID of the image these annotations belong to.
      boxes: Bounding boxes in [x, y, width, height] format.
      masks: Binary segmentation masks.

    Returns:
      Number of annotations successfully added.
    """
    annotations_added = 0

    for box, mask in zip(boxes, masks):
      try:
        # Get the polygon points of masks
        segmentation = models_utils.extract_largest_contour_segmentation(mask)
        bbox_width, bbox_height, area = models_utils.get_bbox_details(box)

        # Create annotation in COCO format
        annotation_info = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [
                int(box[0]),
                int(box[1]),
                int(bbox_width),
                int(bbox_height),
            ],
            "area": int(area),
            "iscrowd": 0,
            "segmentation": segmentation,
        }
        self.coco_output["annotations"].append(annotation_info)
        self.annotation_id_counter += 1
        annotations_added += 1
      except (ValueError, SystemError) as e:
        print(f"[ERROR] Failed to create annotation: {e}")
        continue

    return annotations_added

  def save(self, output_path: str) -> None:
    """Saves COCO annotations to JSON file.

    Args:
      output_path: Full path where the JSON file should be saved.
    """
    with open(output_path, "w") as f:
      json.dump(self.coco_output, f, indent=4)

  def get_statistics(self) -> dict[str, int | str]:
    """Gets statistics about the annotations created.

    Returns:
      Stats on how many annotations were created for what category.
    """
    return {
        "num_images": self.image_id_counter,
        "num_annotations": self.annotation_id_counter,
        "category": self.category_name,
    }
