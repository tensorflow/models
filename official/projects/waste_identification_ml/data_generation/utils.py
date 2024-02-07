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

"""Utility functions for the automated mark generation script."""

import random
from typing import Any
import imantics
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image: np.ndarray):
  """Plots a mask image.

  Args:
    image: A numpy array of shape (height, width) representing a mask.
  """
  plt.figure(figsize=(24, 32))
  plt.imshow(image, cmap='gray')
  plt.show()


def _show_anns(anns: list[dict[str, Any]]):
  """Displays annotations on an image.

  Args:
    anns: A list of dictionaries representing annotations.

  Returns:
    None.
  """
  if not anns:
    return
  sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for ann in sorted_anns:
    m = ann['segmentation']
    img = np.ones((m.shape[0], m.shape[1], 3))
    random.seed()
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
      img[:, :, i] = color_mask[i]
    ax.imshow(np.dstack((img, m * 0.35)))


def display_image_with_annotations(
    image: np.ndarray, masks: list[dict[str, Any]]
):
  """Displays an image with annotations.

  Args:
    image: A numpy array of shape (height, width, 3) representing an image.
    masks: A list of dictionaries representing masks.

  Returns:
    None.
  """
  plt.figure(figsize=(24, 32))
  plt.imshow(image)
  _show_anns(masks)
  plt.axis('off')
  plt.show()


def plot_grid(images: list[np.ndarray], n_cols: int):
  """Plots a list of images in a grid with a given number of images per row.

  Args:
    images: A list of numpy arrays representing images.
    n_cols: The number of images per row.

  Returns:
    None.
  """
  images = [np.array(item['segmentation'], dtype=float) for item in images]
  n_rows = int(np.ceil(len(images) / n_cols))
  _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
  axes = axes.flatten()

  for i, image in enumerate(images):
    axes[i].imshow(image, cmap='gray')
    axes[i].set_axis_off()

  plt.tight_layout()
  plt.show()


def convert_bbox_format(coord: list[float]) -> list[float]:
  """Convert bounding box format.

  Convert bounding box coordinates from x, y, width, height to
  xmin, ymin, xmax, ymax format.

  Args:
    coord: A list or tuple containing the coordinates in (x, y, width, height)
      format.

  Returns:
    A list containing the coordinates in (xmin, ymin, xmax, ymax) format.
  """
  xmin, ymin, width, height = coord
  return [xmin, ymin, xmin + width, ymin + height]


def _aspect_ratio(bbox: list[float]) -> float:
  """Calculate the aspect ratio of a bounding box.

  Args:
    bbox: A list or tuple containing the coordinates in (xmin, ymin, xmax, ymax)
      format.

  Returns:
    The aspect ratio, defined as the length of the longer side
    divided by the length of the shorter side.
  """
  xmin, ymin, xmax, ymax = bbox
  width, height = xmax - xmin, ymax - ymin
  return max(width, height) / min(width, height)


def _calculate_area_bounds(
    elements: list[np.ndarray], upper_multiplier: int, lower_multiplier: int
) -> tuple[float, float]:
  """Calculate the upper and lower bounds for a specified key.

  Args:
    elements: A list of elements containing the specified key.
    upper_multiplier: Multiplier to calculate the upper bound of IQR.
    lower_multiplier: Mulitplier to calculate the lower bound of IQR.

  Returns:
    A tuple containing the upper and lower bounds.
  """
  leng = [i['area'] for i in elements]

  q1, _, q3 = np.percentile(leng, [25, 50, 75])
  iqr = q3 - q1
  upper_bound = q3 + upper_multiplier * iqr
  lower_bound = q1 * lower_multiplier
  return upper_bound, lower_bound


def filter_masks(
    image: np.ndarray,
    elements: list[np.ndarray],
    upper_multiplier: int,
    lower_multiplier: int,
    area_ratio_threshold: float,
) -> list[np.ndarray]:
  """Filter masks based on area bounds and aspect ratio.

  Args:
    image: Original image
    elements: List of elements with multiple attributes.
    upper_multiplier: Multiplier to calculate the upper bound of IQR.
    lower_multiplier: Mulitplier to calculate the lower bound of IQR.
    area_ratio_threshold: Threshold for the ratio of mask area to image area.

  Returns:
      List of filtered masks.
  """
  area_upper_bound, area_lower_bound = _calculate_area_bounds(
      elements, upper_multiplier, lower_multiplier
  )
  threshold = area_ratio_threshold * np.prod(image.shape[:-1])
  filtered_elements = []
  for element in elements:
    if (
        area_lower_bound <= element['area'] <= area_upper_bound
        and _aspect_ratio(element['bbox']) <= 2
        and element['area'] <= threshold
    ):
      filtered_elements.append(element)
  return filtered_elements


def _calculate_intersection_score(
    elem1: dict[str, Any], elem2: dict[str, Any]
) -> float:
  """Calculates the intersection score for two masks.

  Args:
    elem1: The first element.
    elem2: The second element.

  Returns:
    The intersection score calculated as the ratio of the intersection
    area to the area of the smaller mask.
  """

  # Check if the masks have the same dimensions.
  if elem1['segmentation'].shape != elem2['segmentation'].shape:
    raise ValueError('The masks must have the same dimensions.')

  min_elem = elem1 if elem1['area'] < elem2['area'] else elem2
  intersection = np.logical_and(elem1['segmentation'], elem2['segmentation'])
  score = np.sum(intersection) / np.sum(min_elem['segmentation'])
  return score


def filter_nested_similar_masks(
    elements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  """Filters out nested masks from a list of elements.

  Args:
    elements: A list of dictionaries representing elements.

  Returns:
    A list of dictionaries representing elements with nested masks filtered out.
  """
  retained_elements = []
  handled_indices = (
      set()
  )  # To keep track of indices that have already been handled

  for i, elem in enumerate(elements):
    if i in handled_indices:
      continue  # Skip elements that have already been handled

    matching_indices = [i]  # Start with the current element

    # Find all elements that match with the current element
    for j, other_elem in enumerate(elements):
      if i != j and _calculate_intersection_score(elem, other_elem) > 0.95:
        matching_indices.append(j)

    # If more than one element matched, find the one with the highest 'area'
    # and add it to retained_elements
    if len(matching_indices) > 1:
      highest_area_index = max(
          matching_indices, key=lambda idx: elements[idx]['area']
      )
      retained_elements.append(elements[highest_area_index])
      handled_indices.update(
          matching_indices
      )  # Mark all matching indices as handled
    else:
      # If no matches were found, retain the current element
      retained_elements.append(elem)
      handled_indices.add(i)  # Mark the current index as handled

  return retained_elements


def generate_coco_json(
    masks: list[np.ndarray],
    image: np.ndarray,
    category_name: str,
    file_name: str,
) -> dict[str, Any]:
  """Generates a COCO JSON annotation.

  Create a COCO formatted JSON file for the given masks, image, and
  category name.

  Args:
    masks: A list of masks.
    image: The image to which the masks correspond.
    category_name: The name of the category for the masks.
    file_name: The name of the file to save the COCO JSON to.

  Returns:
    A COCO JSON dictionary.
  """
  height, width = image.shape[:2]

  # Initialize variables
  mask = np.zeros((height, width), dtype=np.uint8)
  images_dict = [{}]
  categories_dict = [{}]
  annotations_dict = []
  annotation_id = 1

  # Process masks
  for sub_mask in masks:
    # Convert mask to numpy array
    mask_array = sub_mask.reshape(height, width).astype(np.uint8)

    # Create Mask object and add it to the imantics_Image
    mask_image = imantics.Mask(mask_array)

    # Create imantics_Image object
    imantics_image = imantics.Image(image)
    imantics_image.add(mask_image, category=imantics.Category(category_name))

    try:
      # Export imantics_Image as COCO JSON
      coco_json = imantics_image.export(style='coco')
    except imantics.ExportError as exc:
      print('Error:', exc)
      continue

    # Update images_dict and categories_dict
    images_dict[0] = coco_json['images'][0]
    categories_dict[0] = coco_json['categories'][0]

    # Retrieve annotation information and modify the segmentation field
    annotation = coco_json['annotations'][0]
    annotation['segmentation'] = [max(annotation['segmentation'], key=len)]

    # Check for valid segmentations and create annotation dictionary
    if len(annotation['segmentation']) >= 1:
      for segmentation in annotation['segmentation']:
        if len(segmentation) > 4:
          annotation_dict = {
              'id': annotation_id,
              'image_id': annotation['image_id'],
              'category_id': annotation['category_id'],
              'iscrowd': annotation['iscrowd'],
              'area': annotation['area'],
              'bbox': annotation['bbox'],
              'segmentation': [segmentation],
          }
          annotations_dict.append(annotation_dict)
          annotation_id += 1

    # Free up memory
    del mask_image, coco_json, imantics_image

    # Add mask_array to the overall mask
    mask += mask_array

  # assign file name
  images_dict[0]['file_name'] = file_name

  # Create final COCO dictionary
  coco_dict_final = {
      'images': images_dict,
      'categories': categories_dict,
      'annotations': annotations_dict,
  }

  return coco_dict_final
