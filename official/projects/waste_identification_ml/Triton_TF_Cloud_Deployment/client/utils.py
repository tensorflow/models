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

"""Utility functions for the pipeline."""

from collections.abc import Mapping, Sequence
import csv
import dataclasses
import logging
import os
from typing import Any, TypedDict
import cv2
import natsort
import numpy as np
import tensorflow as tf, tf_keras


class ItemDict(TypedDict):
  id: int
  name: str
  supercategory: str


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


def _reframe_image_corners_relative_to_boxes(boxes: tf.Tensor) -> tf.Tensor:
  """Reframe the image corners ([0, 0, 1, 1]) to be relative to boxes.

  The local coordinate frame of each box is assumed to be relative to
  its own for corners.

  Args:
    boxes: A float tensor of [num_boxes, 4] of (ymin, xmin, ymax, xmax)
      coordinates in relative coordinate space of each bounding box.

  Returns:
    reframed_boxes: Reframes boxes with same shape as input.
  """
  ymin, xmin, ymax, xmax = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])

  height = tf.maximum(ymax - ymin, 1e-4)
  width = tf.maximum(xmax - xmin, 1e-4)

  ymin_out = (0 - ymin) / height
  xmin_out = (0 - xmin) / width
  ymax_out = (1 - ymin) / height
  xmax_out = (1 - xmin) / width
  return tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)


def _reframe_box_masks_to_image_masks(
    box_masks: tf.Tensor,
    boxes: tf.Tensor,
    image_height: int,
    image_width: int,
    resize_method='bilinear'
) -> tf.Tensor:
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.
  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.
  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
  resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""

    num_boxes = tf.shape(box_masks)[0]
    box_masks_expanded = tf.expand_dims(box_masks, axis=3)

    resized_crops = tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=_reframe_image_corners_relative_to_boxes(boxes),
        box_indices=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        method=resize_method,
        extrapolation_value=0)
    return tf.cast(resized_crops, box_masks.dtype)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
  return tf.squeeze(image_masks, axis=3)


def _read_csv_to_list(file_path: str) -> Sequence[str]:
  """Reads a CSV file and returns its contents as a list.

  This function reads the given CSV file, skips the header, and assumes
  there is only one column in the CSV. It returns the contents as a list of
  strings.

  Args:
      file_path: The path to the CSV file.

  Returns:
      The contents of the CSV file as a list of strings.
  """
  data_list = []
  with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      data_list.append(row[0])  # Assuming there is only one column in the CSV
  return data_list


def _categories_dictionary(objects: Sequence[str]) -> Mapping[int, ItemDict]:
  """This function takes a list of objects and returns a dictionaries.

  A dictionary of objects, where each object is represented by a dictionary
  with the following keys:
    - id: The ID of the object.
    - name: The name of the object.
    - supercategory: The supercategory of the object.

  Args:
    objects: A list of strings, where each string is the name of an object.

  Returns:
    A tuple of two dictionaries, as described above.
  """
  category_index = {}

  for num, obj_name in enumerate(objects, start=1):
    obj_dict = {'id': num, 'name': obj_name, 'supercategory': 'objects'}
    category_index[num] = obj_dict
  return category_index


def load_labels(
    labels_path: str,
) -> tuple[Sequence[str], Mapping[int, ItemDict]]:
  """Loads labels from a CSV file and generates category mappings.

  Args:
      labels_path: Path to the CSV file containing label definitions.

  Returns:
    category_indices: A list of category indices.
    category_index: A dictionary mapping category indices to ItemDict objects.
  """
  category_indices = _read_csv_to_list(labels_path)
  category_index = _categories_dictionary(category_indices)
  return category_indices, category_index


def files_paths(folder_path):
  """List the full paths of image files in a folder and sort them.

  Args:
    folder_path: The path of the folder to list the image files from.

  Returns:
    A list of full paths of the image files in the folder, sorted in ascending
    order.
  """
  img_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
  image_files_full_path = []

  for entry in os.scandir(folder_path):
    if entry.is_file() and entry.name.lower().endswith(img_extensions):
      image_files_full_path.append(entry.path)

  # Sort the list of files by name
  image_files_full_path = natsort.natsorted(image_files_full_path)

  return image_files_full_path


def create_log_file(name: str, logs_folder_path: str) -> logging.Logger:
  """Creates a logger and a log file given the name of the video.

  Args:
    name: The name of the video.
    logs_folder_path: Path to the directory where logs should be saved.

  Returns:
    logging.Logger: Logger object configured to write logs to the file.
  """
  log_file_path = os.path.join(logs_folder_path, f'{name}.log')
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(log_file_path)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  return logger


def reframe_masks(
    results: Mapping[str, Any], boxes: str, height: int, width: int
) -> np.ndarray:
  """Reframe the masks to an image size.

  Args:
    results: The detection results from the model.
    boxes: The detection boxes.
    height: The height of the original image.
    width: The width of the original image.

  Returns:
    The reframed masks.
  """
  detection_masks = results['detection_masks'][0]
  detection_boxes = results[boxes][0]
  detection_masks_reframed = _reframe_box_masks_to_image_masks(
      detection_masks, detection_boxes, height, width
  )
  detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, np.uint8)
  detection_masks_reframed = detection_masks_reframed.numpy()
  return detection_masks_reframed


def _calculate_area(mask: np.ndarray) -> int:
  """Calculate the area of the mask.

  Args:
    mask: The mask to calculate the area of.

  Returns:
    The area of the mask.
  """
  return np.sum(mask)


def _calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
  """Calculate the intersection over union (IoU) between two masks.

  Args:
    mask1: The first mask.
    mask2: The second mask.

  Returns:
    The intersection over union (IoU) between the two masks.
  """
  intersection = np.logical_and(mask1, mask2).sum()
  union = np.logical_or(mask1, mask2).sum()
  return intersection / union if union != 0 else 0


def _is_contained(mask1: np.ndarray, mask2: np.ndarray) -> bool:
  """Check if mask1 is entirely contained within mask2.

  Args:
    mask1: The first mask.
    mask2: The second mask.

  Returns:
    True if mask1 is entirely contained within mask2, False otherwise.
  """
  return np.array_equal(np.logical_and(mask1, mask2), mask1)


# TODO: b/416838511 - Reduce the nesting statement in the for loop.
def filter_masks(
    masks: np.ndarray,
    iou_threshold: float = 0.8,
    area_threshold: int | None = None,
) -> Sequence[int]:
  """Filter the overlapping masks.

  Filter the masks based on the area and intersection over union (IoU).

  Args:
    masks: The masks to filter.
    iou_threshold: The threshold for the intersection over union (IoU) between
      two masks.
    area_threshold: The threshold for the area of the mask.

  Returns:
    The indices of the unique masks.
  """
  # Calculate the area for each mask
  areas = np.array([_calculate_area(mask) for mask in masks])

  # Sort the masks based on area in descending order
  sorted_indices = np.argsort(areas)[::-1]
  sorted_masks = masks[sorted_indices]
  sorted_areas = areas[sorted_indices]

  unique_indices = []

  for i, mask in enumerate(sorted_masks):
    if (
        area_threshold is not None and sorted_areas[i] > area_threshold
    ) or sorted_areas[i] < 4000:
      continue

    keep = True
    for j in range(i):
      if _calculate_iou(mask, sorted_masks[j]) > iou_threshold or _is_contained(
          mask, sorted_masks[j]
      ):
        keep = False
        break
    if keep:
      unique_indices.append(sorted_indices[i])

  return unique_indices


def resize_each_mask(
    masks: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
  """Resize each mask to the target height and width.

  Args:
    masks: The masks to resize.
    target_height: The target height of the resized masks.
    target_width: The target width of the resized masks.

  Returns:
    The resized masks.
  """
  combined_masks = []
  for i in masks:
    mask = cv2.resize(
        i, (target_width, target_height), interpolation=cv2.INTER_NEAREST
    )
    combined_masks.append(mask)
  return np.array(combined_masks)


def extract_and_resize_objects(
    results: Mapping[str, Any],
    masks: str,
    boxes: str,
    image: np.ndarray,
    resize_factor: float = 0.5,
) -> Sequence[np.ndarray]:
  """Extract and resize objects from the detection results.

  Args:
    results: The detection results from the model.
    masks: The masks to extract objects from.
    boxes: The bounding boxes of the objects.
    image: The image to extract objects from.
    resize_factor: The factor by which to resize the objects.

  Returns:
    A list of cropped objects.
  """
  cropped_objects = []

  for i, mask in enumerate(results[masks]):
    ymin, xmin, ymax, xmax = results[boxes][0][i]
    mask = np.expand_dims(mask, axis=-1)

    # Crop the object using the mask and bounding box
    cropped_object = np.where(
        mask[ymin:ymax, xmin:xmax], image[ymin:ymax, xmin:xmax], 0
    )

    # Calculate new dimensions
    new_width = int(cropped_object.shape[1] * resize_factor)
    new_height = int(cropped_object.shape[0] * resize_factor)
    cropped_object = cv2.resize(
        cropped_object, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    cropped_objects.append(cropped_object)

  return cropped_objects


def adjust_image_size(
    height: int, width: int, min_size: int
) -> tuple[int, int]:
  """Adjust the image size to ensure both dimensions are at least of min_size.

  Args:
    height: The height of the image.
    width: The width of the image.
    min_size: Minimum size of the image dimension needed.

  Returns:
    The adjusted height and width of the image.
  """
  if height < min_size or width < min_size:
    return height, width

  # Calculate the scale factor to ensure both dimensions remain at least 1024
  scale_factor = min(height / min_size, width / min_size)
  return int(height / scale_factor), int(width / scale_factor)


def filter_detections(
    results: Mapping[str, np.ndarray],
    valid_indices: Sequence[int] | Sequence[bool],
) -> Mapping[str, np.ndarray]:
  """Filter the detection results based on the valid indices.

  Args:
    results: The detection results from the model.
    valid_indices: The indices of the valid detections.

  Returns:
    The filtered detection results.
  """
  if np.array(valid_indices).dtype == bool:
    new_num_detections = int(np.sum(valid_indices))
  else:
    new_num_detections = len(valid_indices)

  # Define the keys to filter
  keys_to_filter = [
      'detection_masks',
      'detection_masks_resized',
      'detection_masks_reframed',
      'detection_classes',
      'detection_boxes',
      'normalized_boxes',
      'detection_scores',
  ]

  filtered_output = {}

  for key in keys_to_filter:
    if key in results:
      if key == 'detection_masks':
        filtered_output[key] = results[key][:, valid_indices, :, :]
      elif key in ['detection_masks_resized', 'detection_masks_reframed']:
        filtered_output[key] = results[key][valid_indices, :, :]
      elif key in ['detection_boxes', 'normalized_boxes']:
        filtered_output[key] = results[key][:, valid_indices, :]
      elif key in [
          'detection_classes',
          'detection_scores',
          'detection_classes_names',
      ]:
        filtered_output[key] = results[key][:, valid_indices]
  filtered_output['image_info'] = results['image_info']
  filtered_output['num_detections'] = np.array([new_num_detections])

  return filtered_output


def resize_bbox(
    bbox: BoundingBox, old_size: ImageSize, new_size: ImageSize
) -> tuple[int, int, int, int]:
  """Resize bounding box coordinates based on new image size.

  Args:
      bbox: BoundingBox with original coordinates.
      old_size: Original image size.
      new_size: New image size.

  Returns:
      Rescaled bounding box coordinates.
  """
  scale_x = new_size.width / old_size.width
  scale_y = new_size.height / old_size.height

  new_y1 = int(bbox.y1 * scale_y)
  new_x1 = int(bbox.x1 * scale_x)
  new_y2 = int(bbox.y2 * scale_y)
  new_x2 = int(bbox.x2 * scale_x)

  return new_y1, new_x1, new_y2, new_x2
