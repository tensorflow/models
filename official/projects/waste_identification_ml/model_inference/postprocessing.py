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

"""Post process the results output from the ML model.

Given the output from the 2 mask RCNN models. The 3 main tasks are done by three
functions mentioned below -
1. reframing_masks : Reframe the masks according to the size of an image and
their respective positions within an image.
2. find_similar_masks : Given masks from the output of 2 models. Find masks 
which belong to the same object and combine all of their attributes like 
confidence score, bounding boxes, label names, etc. The masks are mapped to each
other if their score is above a threshold limit. Two outputs are combined into 
a single output.
3. filter_bounding_boxes : The combined output may have nested bounding boxes of 
the same object. The parent bounding boxes are removed in this step so that any 
object should not have more than a single bounding box. 
"""
import copy
from typing import Any, Optional, TypedDict
import numpy as np
import tensorflow as tf, tf_keras


class DetectionResult(TypedDict):
  num_detections: np.ndarray
  detection_classes: np.ndarray
  detection_scores: np.ndarray
  detection_boxes: np.ndarray
  detection_classes_names: np.ndarray
  detection_masks_reframed: np.ndarray


class ItemDict(TypedDict):
  id: int
  name: str
  supercategory: str


def reframe_image_corners_relative_to_boxes(boxes: tf.Tensor) -> tf.Tensor:
  """Reframe the image corners ([0, 0, 1, 1]) to be relative to boxes.

  The local coordinate frame of each box is assumed to be relative to
  its own for corners.

  Args:
    boxes: A float tensor of [num_boxes, 4] of (ymin, xmin, ymax, xmax)
      coordinates in relative coordinate space of each bounding box.

  Returns:
    reframed_boxes: Reframes boxes with same shape as input.
  """
  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)

  height = tf.maximum(ymax - ymin, 1e-4)
  width = tf.maximum(xmax - xmin, 1e-4)

  ymin_out = (0 - ymin) / height
  xmin_out = (0 - xmin) / width
  ymax_out = (1 - ymin) / height
  xmax_out = (1 - xmin) / width
  return tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)


def reframe_box_masks_to_image_masks(
    box_masks: tf.Tensor,
    boxes: tf.Tensor,
    image_height: int,
    image_width: int,
    resize_method: str = 'bilinear',
) -> tf.Tensor:
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
      corners. Row i contains [ymin, xmin, ymax, xmax] of the box corresponding
      to mask i. Note that the box corners are in normalized coordinates.
    image_height: Image height. The output mask will have the same height as the
      image height.
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
        boxes=reframe_image_corners_relative_to_boxes(boxes),
        box_indices=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        method=resize_method,
        extrapolation_value=0,
    )
    return tf.cast(resized_crops, box_masks.dtype)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype),
  )
  return tf.squeeze(image_masks, axis=3)


def reframing_masks(
    results: dict[str, np.ndarray], height: int, width: int
) -> dict[str, np.ndarray]:
  """Processes the output from Mask RCNN model to create a full size mask.

  Args:
    results: list of dictionaries containing the output of Mask RCNN.
    height: The height of the image.
    width: The width of the image.

  Returns:
    A processed list of dictionaries.
  """
  result = copy.deepcopy(results)
  result['detection_boxes'][0][:, [0, 2]] /= height
  result['detection_boxes'][0][:, [1, 3]] /= width

  detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
  detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])
  detection_masks_reframed = reframe_box_masks_to_image_masks(
      detection_masks, detection_boxes, height, width
  )
  detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8, np.uint8)
  result['detection_masks_reframed'] = detection_masks_reframed.numpy()
  return result


def find_id_by_name(
    dictionary: dict[int, ItemDict], name: str
) -> Optional[int]:
  """Finds the id of a dictionary given its value.

  Args:
    dictionary: The dictionary containing the data.
    name: The value to find.

  Returns:
    The id, or None if its not found.
  """

  # Iterate over the dictionary, and check if the name of the user
  # matches the name that was passed in.
  for value in dictionary.values():
    if value['name'] == name:
      # If the name matches, return the id of the user.
      return value['id']

  return None


def combine_bounding_boxes(
    box1: list[float],
    box2: list[float],
) -> list[float]:
  """Combines two bounding boxes.

  Args:
    box1: A list of four numbers representing the coordinates of the first
      bounding box.
    box2: A list of four numbers representing the coordinates of the second
      bounding box.

  Returns:
    A list of four numbers representing the coordinates of the combined
    bounding box.
  """

  ymin = min(box1[0], box2[0])
  xmin = min(box1[1], box2[1])
  ymax = max(box1[2], box2[2])
  xmax = max(box1[3], box2[3])

  return [ymin, xmin, ymax, xmax]


def calculate_combined_scores_boxes_classes(
    i: int,
    j: int,
    results_1: DetectionResult,
    results_2: DetectionResult,
    category_indices: list[list[Any]],
    category_index_combined: dict[int, ItemDict],
) -> tuple[Any, list[float], Any, Optional[int]]:
  """Calculate combined scores, boxes, and classes for matched masks.

  Args:
      i: Index of the mask from the results_1.
      j: Index of the mask from the results_2.
      results_1: A dictionary which contains the results from the first model.
      results_2: A dictionary which contains the results from the second model.
      category_indices: list of sub lists which contains the labels of 1st and
        2nd ML model.
      category_index_combined: Combined category index.

  Returns:
      tuple: A tuple containing:
          - avg_score: Average score of the matched masks.
          - combined_box: Combined bounding box for the matched masks.
          - combined_label: Combined label of the matched masks.
          - result_id: ID associated with the combined label.
  """
  score_1 = results_1['detection_scores'][0][i]
  score_2 = results_2['detection_scores'][0][j]
  avg_score = (score_1 + score_2) / 2

  box_1 = results_1['detection_boxes'][0][i]
  box_2 = results_2['detection_boxes'][0][j]
  combined_box = combine_bounding_boxes(box_1, box_2)

  class_1 = results_1['detection_classes'][0][i]
  class_2 = results_2['detection_classes'][0][j]
  combined_label = (
      category_indices[0][class_1] + '_' + category_indices[1][class_2]
  )
  result_id = find_id_by_name(category_index_combined, combined_label)

  return avg_score, combined_box, combined_label, result_id


def calculate_single_result(
    index: int,
    result: DetectionResult,
    category_indices: list[list[Any]],
    flag: Any | str,
) -> tuple[float, tuple[float, float, float, float], str]:
  """Calculate scores, boxes, and classes for non-matched masks.

  Args:
      index: Index of the mask in the results.
      result: A dictionary containing detection results (either from results_1
        or results_2).
      category_indices: list of category indices.
      flag: To identify whose model did not detected an object.

  Returns:
    score: Score of the mask.
    box: Bounding box of the mask.
    combined_label: Label of the mask with the added suffix.
  """
  combined_label = 'Default Value'
  score = result['detection_scores'][0][index]
  box = result['detection_boxes'][0][index]
  class_idx = result['detection_classes'][0][index]
  if flag == 'after':
    combined_label = category_indices[class_idx] + '_Na'
  elif flag == 'before':
    combined_label = 'Na_' + category_indices[class_idx]
  return score, box, combined_label


def calculate_iou(
    mask1: np.ndarray, mask2: np.ndarray
) -> tuple[float, np.ndarray]:
  """Calculates the intersection over union (IoU) score for two masks.

  Args:
    mask1: The first mask.
    mask2: The second mask.

  Returns:
    The IoU scorea and union of two masks.
  """

  # Check if the masks have the same dimensions.
  if mask1.shape != mask2.shape:
    raise ValueError('The masks must have the same dimensions.')

  intersection = np.logical_and(mask1, mask2)
  union = np.logical_or(mask1, mask2)
  iou_score = np.sum(intersection) / np.sum(union)
  return iou_score, union


def find_similar_masks(
    results_1: DetectionResult,
    results_2: DetectionResult,
    num_detections: int,
    min_score_thresh: float,
    category_indices: list[list[Any]],
    category_index_combined: dict[int, ItemDict],
    area_threshold: float,
    iou_threshold: float = 0.8,
) -> dict[str, np.ndarray]:
  """Aligns the masks of the detections in `results_1` and `results_2`.

  Args:
    results_1: A dictionary which contains the results from the first model.
    results_2: A dictionary which contains the results from the second model.
    num_detections: The number of detections to consider.
    min_score_thresh: The minimum score threshold for a detection
    category_indices: list of sub lists which contains the labels of 1st and 2nd
      ML model
    category_index_combined: A dictionary with an object ID and nested
      dictionary with name. e.g. {1: {'id': 1, 'name': 'Fiber_Na_Bag',
      'supercategory': 'objects'}}
    area_threshold: Threshold for mask area consideration.
    iou_threshold: IOU threshold to compare masks.

  Returns:
     A dictionary containing the following keys:
       - num_detections: The number of aligned detections.
       - detection_classes: A NumPy array of shape (num_detections,) containing
       the classes for the aligned detections.
       - detection_scores: A NumPy array of shape (num_detections,) containing
       the scores for the aligned detections.
       - detection_boxes: A NumPy array of shape (num_detections, 4) containing
       the bounding boxes for the aligned detections.
       - detection_classes_names: A list of strings containing the names of the
       classes for the aligned detections.
       - detection_masks_reframed: A NumPy array of shape (num_detections,
       height, width) containing the full masks for the aligned detections.
  """
  detection_masks_reframed = []
  detection_scores = []
  detection_boxes = []
  detection_classes = []
  detection_classes_names = []

  aligned_masks = 0
  masks_list1 = results_1['detection_masks_reframed'][:num_detections]
  masks_list2 = results_2['detection_masks_reframed'][:num_detections]
  scores_list1 = results_1['detection_scores'][0]
  scores_list2 = results_2['detection_scores'][0]
  matched_masks_list2 = [False] * len(masks_list2)
  matched_masks_list1 = [False] * len(masks_list1)

  for i, mask1 in enumerate(masks_list1):
    if (scores_list1[i] > min_score_thresh) and (
        np.sum(mask1) < area_threshold
    ):
      is_similar = False

      for j, mask2 in enumerate(masks_list2):
        if scores_list2[j] > min_score_thresh and (
            np.sum(mask2) < area_threshold
        ):
          iou, union = calculate_iou(mask1, mask2)

          # masks which are present both in the 'detection_masks_reframed'
          # key of 'results_1' & 'results_2' dictionary
          if iou > iou_threshold:
            aligned_masks += 1
            is_similar = True
            matched_masks_list2[j] = True
            matched_masks_list1[i] = True

            detection_masks_reframed.append(union)

            avg_score, combined_box, combined_label, result_id = (
                calculate_combined_scores_boxes_classes(
                    i,
                    j,
                    results_1,
                    results_2,
                    category_indices,
                    category_index_combined,
                )
            )
            detection_scores.append(avg_score)
            detection_boxes.append(combined_box)
            detection_classes_names.append(combined_label)
            detection_classes.append(result_id)
            break

      # masks which are only present in the 'detection_masks_reframed'
      # of 'results_1' dictionary
      if not is_similar:
        aligned_masks += 1
        detection_masks_reframed.append(mask1)
        score, box, combined_label = calculate_single_result(
            i, results_1, category_indices[0], 'after'
        )
        detection_scores.append(score)
        detection_boxes.append(box)
        detection_classes_names.append(combined_label)
        result_id = find_id_by_name(category_index_combined, combined_label)
        detection_classes.append(result_id)

  # masks which are only present in the 'detection_masks_reframed'
  # key of 'results_2' dictionary
  for k, mask2 in enumerate(masks_list2):
    if (
        (not matched_masks_list2[k])
        and (scores_list2[k] > min_score_thresh)
        and (np.sum(mask2) < area_threshold)
    ):
      aligned_masks += 1
      detection_masks_reframed.append(mask2)
      score, box, combined_label = calculate_single_result(
          k, results_2, category_indices[1], 'before'
      )
      detection_scores.append(score)
      detection_boxes.append(box)
      detection_classes_names.append(combined_label)
      result_id = find_id_by_name(category_index_combined, combined_label)
      detection_classes.append(result_id)

  final_result = {
      'num_detections': np.array([aligned_masks]),
      'detection_classes': np.array(detection_classes),
      'detection_scores': np.array([detection_scores]),
      'detection_boxes': np.array([detection_boxes]),
      'detection_classes_names': np.array(detection_classes_names),
      'detection_masks_reframed': np.array(detection_masks_reframed),
  }

  return final_result


def filter_bounding_boxes(
    bounding_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
    area_ratio_threshold: float = 0.8,
) -> tuple[list[tuple[int, int, int, int]], list[int]]:
  """Filters overlapping bounding boxes based on IoU and area ratio criteria.

  This function filters out overlapping bounding boxes from a given list based
  on Intersection over Union (IoU) and area ratio of the intersection to the
  smaller bounding box's area.

  Args:
      bounding_boxes: A list of bounding boxes, where each bounding box is
        represented as a tuple of (xmin, ymin, xmax, ymax).
      iou_threshold: Threshold for Intersection over Union. Bounding boxes with
        IoU above this threshold will be considered overlapping. Defaults to
        0.5.
      area_ratio_threshold: Threshold for the area ratio of the intersection to
        the smaller bounding box's area. Defaults to 0.8.

  Returns:
      tuple: A tuple containing:
          - filtered_boxes: A list of bounding boxes that passed the filtering
          criteria.
          - eliminated_indices: Indices of the bounding boxes that didn't pass
          the filtering criteria.

  Example:
      >>> bounding_boxes = [(10, 10, 50, 50), (20, 20, 60, 60)]
      >>> filter_bounding_boxes(bounding_boxes)
      ([(10, 10, 50, 50)], [1])
  """
  filtered_boxes = []
  eliminated_indices = []

  # Enumerate and sort the boxes based on their area in descending order
  enumerated_boxes = list(enumerate(bounding_boxes))
  sorted_boxes = sorted(
      enumerated_boxes,
      key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
      reverse=True,
  )

  for idx, bbox in sorted_boxes:
    skip_box = False

    # Calculate areas of individual bounding boxes
    area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    for jdx, other_bbox in sorted_boxes:
      if idx == jdx:
        continue

      # Calculate intersection coordinates
      xmin_inter = max(bbox[0], other_bbox[0])
      ymin_inter = max(bbox[1], other_bbox[1])
      xmax_inter = min(bbox[2], other_bbox[2])
      ymax_inter = min(bbox[3], other_bbox[3])

      # Calculate intersection area
      width_inter = max(0, xmax_inter - xmin_inter)
      height_inter = max(0, ymax_inter - ymin_inter)
      area_inter = width_inter * height_inter

      area_other_bbox = (other_bbox[2] - other_bbox[0]) * (
          other_bbox[3] - other_bbox[1]
      )

      # Calculate area ratio
      area_ratio = area_inter / min(area_bbox, area_other_bbox)

      # Check for overlapping and area ratio thresholds
      if area_ratio > area_ratio_threshold:
        if area_bbox > area_other_bbox:
          skip_box = True
          eliminated_indices.append(idx)
          break
      elif (
          area_inter > 0
          and area_inter / (area_bbox + area_other_bbox - area_inter)
          > iou_threshold
      ):
        if area_bbox > area_other_bbox:
          skip_box = True
          eliminated_indices.append(idx)
          break

    if not skip_box:
      filtered_boxes.append(bbox)

  return filtered_boxes, eliminated_indices
