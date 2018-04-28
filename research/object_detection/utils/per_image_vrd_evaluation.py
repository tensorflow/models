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
"""Evaluates Visual Relations Detection(VRD) result evaluation on an image.

Annotate each VRD result as true positives or false positive according to
a predefined IOU ratio. Multi-class detection is supported by default.
Based on the settings, per image evaluation is performed either on phrase
detection subtask or on relation detection subtask.
"""
import numpy as np

from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops


class PerImageVRDEvaluation(object):
  """Evaluate vrd result of a single image."""

  def __init__(self, matching_iou_threshold=0.5):
    """Initialized PerImageVRDEvaluation by evaluation parameters.

    Args:
      matching_iou_threshold: A ratio of area intersection to union, which is
          the threshold to consider whether a detection is true positive or not;
          in phrase detection subtask.
    """
    self.matching_iou_threshold = matching_iou_threshold

  def compute_detection_tp_fp(self, detected_box_tuples, detected_scores,
                              detected_class_tuples, groundtruth_box_tuples,
                              groundtruth_class_tuples):
    """Evaluates VRD as being tp, fp from a single image.

    Args:
      detected_box_tuples: A numpy array of structures with shape [N,],
          representing N tuples, each tuple containing the same number of named
          bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max].
      detected_scores: A float numpy array of shape [N,], representing
          the confidence scores of the detected N object instances.
      detected_class_tuples: A numpy array of structures shape [N,],
          representing the class labels of the corresponding bounding boxes and
          possibly additional classes.
      groundtruth_box_tuples: A float numpy array of structures with the shape
          [M,], representing M tuples, each tuple containing the same number
          of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max].
      groundtruth_class_tuples: A numpy array of structures shape [M,],
          representing  the class labels of the corresponding bounding boxes and
          possibly additional classes.

    Returns:
      scores: A single numpy array with shape [N,], representing N scores
          detected with object class, sorted in descentent order.
      tp_fp_labels: A single boolean numpy array of shape [N,], representing N
          True/False positive label, one label per tuple. The labels are sorted
          so that the order of the labels matches the order of the scores.
    """

    scores, tp_fp_labels = self._compute_tp_fp(
        detected_box_tuples=detected_box_tuples,
        detected_scores=detected_scores,
        detected_class_tuples=detected_class_tuples,
        groundtruth_box_tuples=groundtruth_box_tuples,
        groundtruth_class_tuples=groundtruth_class_tuples)

    return scores, tp_fp_labels

  def _compute_tp_fp(self, detected_box_tuples, detected_scores,
                     detected_class_tuples, groundtruth_box_tuples,
                     groundtruth_class_tuples):
    """Labels as true/false positives detection tuples across all classes.

    Args:
      detected_box_tuples: A numpy array of structures with shape [N,],
          representing N tuples, each tuple containing the same number of named
          bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N,], representing
          the confidence scores of the detected N object instances.
      detected_class_tuples: A numpy array of structures shape [N,],
          representing the class labels of the corresponding bounding boxes and
          possibly additional classes.
      groundtruth_box_tuples: A float numpy array of structures with the shape
          [M,], representing M tuples, each tuple containing the same number
          of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max]
      groundtruth_class_tuples: A numpy array of structures shape [M,],
          representing  the class labels of the corresponding bounding boxes and
          possibly additional classes.

    Returns:
      scores: A single numpy array with shape [N,], representing N scores
          detected with object class, sorted in descentent order.
      tp_fp_labels: A single boolean numpy array of shape [N,], representing N
          True/False positive label, one label per tuple. The labels are sorted
          so that the order of the labels matches the order of the scores.

    """
    unique_gt_tuples = np.unique(
        np.concatenate((groundtruth_class_tuples, detected_class_tuples)))
    result_scores = []
    result_tp_fp_labels = []

    for unique_tuple in unique_gt_tuples:
      detections_selector = (detected_class_tuples == unique_tuple)
      gt_selector = (groundtruth_class_tuples == unique_tuple)
      scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
          detected_box_tuples=detected_box_tuples[detections_selector],
          detected_scores=detected_scores[detections_selector],
          groundtruth_box_tuples=groundtruth_box_tuples[gt_selector])
      result_scores.append(scores)
      result_tp_fp_labels.append(tp_fp_labels)

    result_scores = np.concatenate(result_scores)
    result_tp_fp_labels = np.concatenate(result_tp_fp_labels)

    sorted_indices = np.argsort(result_scores)
    sorted_indices = sorted_indices[::-1]

    return result_scores[sorted_indices], result_tp_fp_labels[sorted_indices]

  def _get_overlaps_and_scores_relation_tuples(
      self, detected_box_tuples, detected_scores, groundtruth_box_tuples):
    """Computes overlaps and scores between detected and groundtruth tuples.

    Both detections and groundtruth boxes have the same class tuples.

    Args:
      detected_box_tuples: A numpy array of structures with shape [N,],
          representing N tuples, each tuple containing the same number of named
          bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N,], representing
          the confidence scores of the detected N object instances.
      groundtruth_box_tuples: A float numpy array of structures with the shape
          [M,], representing M tuples, each tuple containing the same number
          of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max]

    Returns:
      result_iou: A float numpy array of size
        [num_detected_tuples, num_gt_box_tuples].
      scores: The score of the detected boxlist.
    """

    result_iou = np.ones(
        (detected_box_tuples.shape[0], groundtruth_box_tuples.shape[0]),
        dtype=float)
    for field in detected_box_tuples.dtype.fields:
      detected_boxlist_field = np_box_list.BoxList(detected_box_tuples[field])
      detected_boxlist_field.add_field('scores', detected_scores)
      detected_boxlist_field = np_box_list_ops.sort_by_field(
          detected_boxlist_field, 'scores')
      gt_boxlist_field = np_box_list.BoxList(groundtruth_box_tuples[field])
      iou_field = np_box_list_ops.iou(detected_boxlist_field, gt_boxlist_field)
      result_iou = np.minimum(iou_field, result_iou)
    scores = detected_boxlist_field.get_field('scores')
    return result_iou, scores

  def _compute_tp_fp_for_single_class(self, detected_box_tuples,
                                      detected_scores, groundtruth_box_tuples):
    """Labels boxes detected with the same class from the same image as tp/fp.

    Args:
      detected_box_tuples: A numpy array of structures with shape [N,],
          representing N tuples, each tuple containing the same number of named
          bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max]
      detected_scores: A float numpy array of shape [N,], representing
          the confidence scores of the detected N object instances.
      groundtruth_box_tuples: A float numpy array of structures with the shape
          [M,], representing M tuples, each tuple containing the same number
          of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max]

    Returns:
      Two arrays of the same size, containing true/false for N boxes that were
      evaluated as being true positives or false positives;

      scores: A numpy array representing the detection scores.
      tp_fp_labels: a boolean numpy array indicating whether a detection is a
          true positive.
    """
    if detected_box_tuples.size == 0:
      return np.array([], dtype=float), np.array([], dtype=bool)

    min_iou, scores = self._get_overlaps_and_scores_relation_tuples(
        detected_box_tuples=detected_box_tuples,
        detected_scores=detected_scores,
        groundtruth_box_tuples=groundtruth_box_tuples)

    num_detected_tuples = detected_box_tuples.shape[0]
    tp_fp_labels = np.zeros(num_detected_tuples, dtype=bool)

    if min_iou.shape[1] > 0:
      max_overlap_gt_ids = np.argmax(min_iou, axis=1)
      is_gt_tuple_detected = np.zeros(min_iou.shape[1], dtype=bool)
      for i in range(num_detected_tuples):
        gt_id = max_overlap_gt_ids[i]
        if min_iou[i, gt_id] >= self.matching_iou_threshold:
          if not is_gt_tuple_detected[gt_id]:
            tp_fp_labels[i] = True
            is_gt_tuple_detected[gt_id] = True

    return scores, tp_fp_labels
