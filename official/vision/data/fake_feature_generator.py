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

"""Generates fake feature for testing and validation."""

import collections
from typing import Optional, Tuple, Union

import numpy as np

_RGB_CHANNELS = 3


def generate_image_np(height: int,
                      width: int,
                      num_channels: int = _RGB_CHANNELS) -> np.ndarray:
  """Returns a fake numpy image matrix array."""
  return np.reshape(
      np.mod(np.arange(height * width * num_channels), 255).astype(np.uint8),
      newshape=(height, width, num_channels))


def generate_normalized_boxes_np(num_boxes: int) -> np.ndarray:
  """Returns a fake numpy normalized boxes array."""
  xmins = np.reshape(np.arange(num_boxes) / (2 * num_boxes), (num_boxes, 1))
  ymins = np.reshape(np.arange(num_boxes) / (2 * num_boxes), (num_boxes, 1))
  xmaxs = xmins + .5
  ymaxs = ymins + .5
  return np.concatenate((ymins, xmins, ymaxs, xmaxs), axis=-1)


def generate_boxes_np(height: int, width: int, num_boxes: int) -> np.ndarray:
  """Returns a fake numpy absolute boxes array."""
  normalized_boxes = generate_normalized_boxes_np(num_boxes)
  normalized_boxes[:, 1::2] *= height
  normalized_boxes[:, 0::2] *= width
  return normalized_boxes


def generate_classes_np(num_classes: int,
                        size: Optional[int] = None) -> Union[int, np.ndarray]:
  """Returns a fake class or a fake numpy classes array."""
  if size is None:
    return num_classes - 1

  return np.arange(size) % num_classes


def generate_confidences_np(
    size: Optional[int] = None) -> Union[float, np.ndarray]:
  """Returns a fake confidence score or a fake numpy confidence score array."""
  if size is None:
    return 0.5

  return np.arange(size) / size


def generate_instance_masks_np(height: int,
                               width: int,
                               boxes_np: np.ndarray,
                               normalized: bool = True) -> np.ndarray:
  """Returns a fake numpy instance mask matrices array."""
  num_boxes = len(boxes_np)
  instance_masks_np = np.zeros((num_boxes, height, width, 1))
  if normalized:
    boxes_np[:, 1::2] *= height
    boxes_np[:, ::2] *= width
  xmins = boxes_np[:, 0].astype(int)
  ymins = boxes_np[:, 1].astype(int)
  box_widths = boxes_np[:, 2].astype(int) - xmins
  box_heights = boxes_np[:, 3].astype(int) - ymins

  for i, (x, y, w, h) in enumerate(zip(xmins, ymins, box_widths, box_heights)):
    instance_masks_np[i, y:y + h, x:x + w, :] = np.reshape(
        np.mod(np.arange(h * w), 2).astype(np.uint8), newshape=(h, w, 1))
  return instance_masks_np


def generate_semantic_mask_np(height: int, width: int,
                              num_classes: int) -> np.ndarray:
  """Returns a fake numpy semantic mask array."""
  return generate_image_np(height, width, num_channels=1) % num_classes


def generate_panoptic_masks_np(
    semantic_mask: np.ndarray, instance_masks: np.ndarray,
    instance_classes: np.ndarray,
    stuff_classes_offset: int) -> Tuple[np.ndarray, np.ndarray]:
  """Returns fake numpy panoptic category and instance mask arrays."""
  panoptic_category_mask = np.zeros_like(semantic_mask)
  panoptic_instance_mask = np.zeros_like(semantic_mask)
  instance_ids = collections.defaultdict(int)
  for instance_mask, instance_class in zip(instance_masks, instance_classes):
    if instance_class == 0:
      continue
    instance_ids[instance_class] += 1
    # If a foreground pixel is labelled previously, replace the old category
    # class and instance ID with the new one.
    foreground_indices = np.where(np.equal(instance_mask, 1))
    # Note that instance class start from index 1.
    panoptic_category_mask[foreground_indices] = instance_class + 1
    panoptic_instance_mask[foreground_indices] = instance_ids[instance_class]

  # If there are pixels remains unlablled (labelled as background), then the
  # semantic labels will be used (if it has one).
  # Note that in panoptic FPN, the panoptic labels are expected in this order,
  # 0 (background), 1 ..., N (stuffs), N + 1, ..., N + M - 2 (things)
  # N classes for stuff classes, without background class, and M classes for
  # thing classes, with 0 representing the background class and 1 representing
  # all stuff classes.
  background_indices = np.where(np.equal(panoptic_category_mask, 0))
  panoptic_category_mask[background_indices] = (
      semantic_mask[background_indices] + stuff_classes_offset)
  return panoptic_category_mask, panoptic_instance_mask
