# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Evaluate Object Detection result on a single image.

Annotate each detected result as true positives or false positive according to
a predefined IOU ratio. Non Maximum Supression is used by default. Multi class
detection is supported by default.
"""
import numpy as np

from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops


class PerImageEvaluation(object):
  """Evaluate detection result of a single image."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=0.3,
               nms_max_output_boxes=50):
    """Initialized PerImageEvaluation by evaluation parameters.

    Args:
      num_groundtruth_classes: Number of ground truth object classes
      matching_iou_threshold: A ratio of area intersection to union, which is
          the threshold to consider whether a detection is true positive or not
      nms_iou_threshold: IOU threshold used in Non Maximum Suppression.
      nms_max_output_boxes: Number of maximum output boxes in NMS.
    """
    self.matching_iou_threshold = matching_iou_threshold
    self.nms_iou_threshold = nms_iou_threshold
    self.nms_max_output_boxes = nms_max_output_boxes
    self.num_groundtruth_classes = num_groundtruth_classes

  def compute_object_detection_metrics(self, detected_boxes, detected_scores,
                                       detected_class_labels, groundtruth_boxes,
                                       groundtruth_class_labels,
                                       groundtruth_is_difficult_lists):
    """Compute Object Detection related metrics from a single image.

    Args:
      detected_boxes: A float numpy array of shape [N, 4], representing N
          regions of detected object regions.
          Each row is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N, 1], representing
          the confidence scores of the detected N object instances.
      detected_class_labels: A integer numpy array of shape [N, 1], repreneting
          the class labels of the detected N object instances.
      groundtruth_boxes: A float numpy array of shape [M, 4], representing M
          regions of object instances in ground truth
      groundtruth_class_labels: An integer numpy array of shape [M, 1],
          representing M class labels of object instances in ground truth
      groundtruth_is_difficult_lists: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not

    Returns:
      scores: A list of C float numpy arrays. Each numpy array is of
          shape [K, 1], representing K scores detected with object class
          label c
      tp_fp_labels: A list of C boolean numpy arrays. Each numpy array
          is of shape [K, 1], representing K True/False positive label of
          object instances detected with class label c
      is_class_correctly_detected_in_image: a numpy integer array of
          shape [C, 1], indicating whether the correponding class has a least
          one instance being correctly detected in the image
    """
    detected_boxes, detected_scores, detected_class_labels = (
        self._remove_invalid_boxes(detected_boxes, detected_scores,
                                   detected_class_labels))
    scores, tp_fp_labels = self._compute_tp_fp(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels,
        groundtruth_is_difficult_lists)
    is_class_correctly_detected_in_image = self._compute_cor_loc(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels)
    return scores, tp_fp_labels, is_class_correctly_detected_in_image

  def _compute_cor_loc(self, detected_boxes, detected_scores,
                       detected_class_labels, groundtruth_boxes,
                       groundtruth_class_labels):
    """Compute CorLoc score for object detection result.

    Args:
      detected_boxes: A float numpy array of shape [N, 4], representing N
          regions of detected object regions.
          Each row is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N, 1], representing
          the confidence scores of the detected N object instances.
      detected_class_labels: A integer numpy array of shape [N, 1], repreneting
          the class labels of the detected N object instances.
      groundtruth_boxes: A float numpy array of shape [M, 4], representing M
          regions of object instances in ground truth
      groundtruth_class_labels: An integer numpy array of shape [M, 1],
          representing M class labels of object instances in ground truth
    Returns:
      is_class_correctly_detected_in_image: a numpy integer array of
          shape [C, 1], indicating whether the correponding class has a least
          one instance being correctly detected in the image
    """
    is_class_correctly_detected_in_image = np.zeros(
        self.num_groundtruth_classes, dtype=int)
    for i in range(self.num_groundtruth_classes):
      gt_boxes_at_ith_class = groundtruth_boxes[
          groundtruth_class_labels == i, :]
      detected_boxes_at_ith_class = detected_boxes[
          detected_class_labels == i, :]
      detected_scores_at_ith_class = detected_scores[detected_class_labels == i]
      is_class_correctly_detected_in_image[i] = (
          self._compute_is_aclass_correctly_detected_in_image(
              detected_boxes_at_ith_class, detected_scores_at_ith_class,
              gt_boxes_at_ith_class))

    return is_class_correctly_detected_in_image

  def _compute_is_aclass_correctly_detected_in_image(
      self, detected_boxes, detected_scores, groundtruth_boxes):
    """Compute CorLoc score for a single class.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates

    Returns:
      is_class_correctly_detected_in_image: An integer 1 or 0 denoting whether a
          class is correctly detected in the image or not
    """
    if detected_boxes.size > 0:
      if groundtruth_boxes.size > 0:
        max_score_id = np.argmax(detected_scores)
        detected_boxlist = np_box_list.BoxList(
            np.expand_dims(detected_boxes[max_score_id, :], axis=0))
        gt_boxlist = np_box_list.BoxList(groundtruth_boxes)
        iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
        if np.max(iou) >= self.matching_iou_threshold:
          return 1
    return 0

  def _compute_tp_fp(self, detected_boxes, detected_scores,
                     detected_class_labels, groundtruth_boxes,
                     groundtruth_class_labels, groundtruth_is_difficult_lists):
    """Labels true/false positives of detections of an image across all classes.

    Args:
      detected_boxes: A float numpy array of shape [N, 4], representing N
          regions of detected object regions.
          Each row is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N, 1], representing
          the confidence scores of the detected N object instances.
      detected_class_labels: A integer numpy array of shape [N, 1], repreneting
          the class labels of the detected N object instances.
      groundtruth_boxes: A float numpy array of shape [M, 4], representing M
          regions of object instances in ground truth
      groundtruth_class_labels: An integer numpy array of shape [M, 1],
          representing M class labels of object instances in ground truth
      groundtruth_is_difficult_lists: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not

    Returns:
      result_scores: A list of float numpy arrays. Each numpy array is of
          shape [K, 1], representing K scores detected with object class
          label c
      result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of
          shape [K, 1], representing K True/False positive label of object
          instances detected with class label c
    """
    result_scores = []
    result_tp_fp_labels = []
    for i in range(self.num_groundtruth_classes):
      gt_boxes_at_ith_class = groundtruth_boxes[(groundtruth_class_labels == i
                                                ), :]
      groundtruth_is_difficult_list_at_ith_class = (
          groundtruth_is_difficult_lists[groundtruth_class_labels == i])
      detected_boxes_at_ith_class = detected_boxes[(detected_class_labels == i
                                                   ), :]
      detected_scores_at_ith_class = detected_scores[detected_class_labels == i]
      scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
          detected_boxes_at_ith_class, detected_scores_at_ith_class,
          gt_boxes_at_ith_class, groundtruth_is_difficult_list_at_ith_class)
      result_scores.append(scores)
      result_tp_fp_labels.append(tp_fp_labels)
    return result_scores, result_tp_fp_labels

  def _remove_invalid_boxes(self, detected_boxes, detected_scores,
                            detected_class_labels):
    valid_indices = np.logical_and(detected_boxes[:, 0] < detected_boxes[:, 2],
                                   detected_boxes[:, 1] < detected_boxes[:, 3])
    return (detected_boxes[valid_indices, :], detected_scores[valid_indices],
            detected_class_labels[valid_indices])

  def _compute_tp_fp_for_single_class(self, detected_boxes, detected_scores,
                                      groundtruth_boxes,
                                      groundtruth_is_difficult_list):
    """Labels boxes detected with the same class from the same image as tp/fp.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not

    Returns:
      scores: A numpy array representing the detection scores
      tp_fp_labels: a boolean numpy array indicating whether a detection is a
      true positive.

    """
    if detected_boxes.size == 0:
      return np.array([], dtype=float), np.array([], dtype=bool)
    detected_boxlist = np_box_list.BoxList(detected_boxes)
    detected_boxlist.add_field('scores', detected_scores)
    detected_boxlist = np_box_list_ops.non_max_suppression(
        detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)

    scores = detected_boxlist.get_field('scores')

    if groundtruth_boxes.size == 0:
      return scores, np.zeros(detected_boxlist.num_boxes(), dtype=bool)
    gt_boxlist = np_box_list.BoxList(groundtruth_boxes)

    iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
    max_overlap_gt_ids = np.argmax(iou, axis=1)
    is_gt_box_detected = np.zeros(gt_boxlist.num_boxes(), dtype=bool)
    tp_fp_labels = np.zeros(detected_boxlist.num_boxes(), dtype=bool)
    is_matched_to_difficult_box = np.zeros(
        detected_boxlist.num_boxes(), dtype=bool)
    for i in range(detected_boxlist.num_boxes()):
      gt_id = max_overlap_gt_ids[i]
      if iou[i, gt_id] >= self.matching_iou_threshold:
        if not groundtruth_is_difficult_list[gt_id]:
          if not is_gt_box_detected[gt_id]:
            tp_fp_labels[i] = True
            is_gt_box_detected[gt_id] = True
        else:
          is_matched_to_difficult_box[i] = True
    return scores[~is_matched_to_difficult_box], tp_fp_labels[
        ~is_matched_to_difficult_box]
