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
Based on the settings, per image evaluation is either performed on boxes or
on object masks.
"""
import numpy as np

from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_box_mask_list
from object_detection.utils import np_box_mask_list_ops


class PerImageEvaluation(object):
  """Evaluate detection result of a single image."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=0.3,
               nms_max_output_boxes=50,
               group_of_weight=0.0):
    """Initialized PerImageEvaluation by evaluation parameters.

    Args:
      num_groundtruth_classes: Number of ground truth object classes
      matching_iou_threshold: A ratio of area intersection to union, which is
          the threshold to consider whether a detection is true positive or not
      nms_iou_threshold: IOU threshold used in Non Maximum Suppression.
      nms_max_output_boxes: Number of maximum output boxes in NMS.
      group_of_weight: Weight of the group-of boxes.
    """
    self.matching_iou_threshold = matching_iou_threshold
    self.nms_iou_threshold = nms_iou_threshold
    self.nms_max_output_boxes = nms_max_output_boxes
    self.num_groundtruth_classes = num_groundtruth_classes
    self.group_of_weight = group_of_weight

  def compute_object_detection_metrics(
      self, detected_boxes, detected_scores, detected_class_labels,
      groundtruth_boxes, groundtruth_class_labels,
      groundtruth_is_difficult_list, groundtruth_is_group_of_list,
      detected_masks=None, groundtruth_masks=None):
    """Evaluates detections as being tp, fp or weighted from a single image.

    The evaluation is done in two stages:
     1. All detections are matched to non group-of boxes; true positives are
        determined and detections matched to difficult boxes are ignored.
     2. Detections that are determined as false positives are matched against
        group-of boxes and weighted if matched.

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
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag
      detected_masks: (optional) A uint8 numpy array of shape
        [N, height, width]. If not None, the metrics will be computed based
        on masks.
      groundtruth_masks: (optional) A uint8 numpy array of shape
        [M, height, width].

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
    detected_boxes, detected_scores, detected_class_labels, detected_masks = (
        self._remove_invalid_boxes(detected_boxes, detected_scores,
                                   detected_class_labels, detected_masks))
    scores, tp_fp_labels = self._compute_tp_fp(
        detected_boxes=detected_boxes,
        detected_scores=detected_scores,
        detected_class_labels=detected_class_labels,
        groundtruth_boxes=groundtruth_boxes,
        groundtruth_class_labels=groundtruth_class_labels,
        groundtruth_is_difficult_list=groundtruth_is_difficult_list,
        groundtruth_is_group_of_list=groundtruth_is_group_of_list,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)

    is_class_correctly_detected_in_image = self._compute_cor_loc(
        detected_boxes=detected_boxes,
        detected_scores=detected_scores,
        detected_class_labels=detected_class_labels,
        groundtruth_boxes=groundtruth_boxes,
        groundtruth_class_labels=groundtruth_class_labels,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)

    return scores, tp_fp_labels, is_class_correctly_detected_in_image

  def _compute_cor_loc(self, detected_boxes, detected_scores,
                       detected_class_labels, groundtruth_boxes,
                       groundtruth_class_labels, detected_masks=None,
                       groundtruth_masks=None):
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
      detected_masks: (optional) A uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A uint8 numpy array of shape
        [M, height, width].

    Returns:
      is_class_correctly_detected_in_image: a numpy integer array of
          shape [C, 1], indicating whether the correponding class has a least
          one instance being correctly detected in the image

    Raises:
      ValueError: If detected masks is not None but groundtruth masks are None,
        or the other way around.
    """
    if (detected_masks is not None and
        groundtruth_masks is None) or (detected_masks is None and
                                       groundtruth_masks is not None):
      raise ValueError(
          'If `detected_masks` is provided, then `groundtruth_masks` should '
          'also be provided.'
      )

    is_class_correctly_detected_in_image = np.zeros(
        self.num_groundtruth_classes, dtype=int)
    for i in range(self.num_groundtruth_classes):
      (gt_boxes_at_ith_class, gt_masks_at_ith_class,
       detected_boxes_at_ith_class, detected_scores_at_ith_class,
       detected_masks_at_ith_class) = self._get_ith_class_arrays(
           detected_boxes, detected_scores, detected_masks,
           detected_class_labels, groundtruth_boxes, groundtruth_masks,
           groundtruth_class_labels, i)
      is_class_correctly_detected_in_image[i] = (
          self._compute_is_class_correctly_detected_in_image(
              detected_boxes=detected_boxes_at_ith_class,
              detected_scores=detected_scores_at_ith_class,
              groundtruth_boxes=gt_boxes_at_ith_class,
              detected_masks=detected_masks_at_ith_class,
              groundtruth_masks=gt_masks_at_ith_class))

    return is_class_correctly_detected_in_image

  def _compute_is_class_correctly_detected_in_image(
      self, detected_boxes, detected_scores, groundtruth_boxes,
      detected_masks=None, groundtruth_masks=None):
    """Compute CorLoc score for a single class.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      detected_masks: (optional) A np.uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A np.uint8 numpy array of shape
        [M, height, width].

    Returns:
      is_class_correctly_detected_in_image: An integer 1 or 0 denoting whether a
          class is correctly detected in the image or not
    """
    if detected_boxes.size > 0:
      if groundtruth_boxes.size > 0:
        max_score_id = np.argmax(detected_scores)
        mask_mode = False
        if detected_masks is not None and groundtruth_masks is not None:
          mask_mode = True
        if mask_mode:
          detected_boxlist = np_box_mask_list.BoxMaskList(
              box_data=np.expand_dims(detected_boxes[max_score_id], axis=0),
              mask_data=np.expand_dims(detected_masks[max_score_id], axis=0))
          gt_boxlist = np_box_mask_list.BoxMaskList(
              box_data=groundtruth_boxes, mask_data=groundtruth_masks)
          iou = np_box_mask_list_ops.iou(detected_boxlist, gt_boxlist)
        else:
          detected_boxlist = np_box_list.BoxList(
              np.expand_dims(detected_boxes[max_score_id, :], axis=0))
          gt_boxlist = np_box_list.BoxList(groundtruth_boxes)
          iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
        if np.max(iou) >= self.matching_iou_threshold:
          return 1
    return 0

  def _compute_tp_fp(self, detected_boxes, detected_scores,
                     detected_class_labels, groundtruth_boxes,
                     groundtruth_class_labels, groundtruth_is_difficult_list,
                     groundtruth_is_group_of_list,
                     detected_masks=None, groundtruth_masks=None):
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
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag
      detected_masks: (optional) A np.uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A np.uint8 numpy array of shape
        [M, height, width].

    Returns:
      result_scores: A list of float numpy arrays. Each numpy array is of
          shape [K, 1], representing K scores detected with object class
          label c
      result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of
          shape [K, 1], representing K True/False positive label of object
          instances detected with class label c

    Raises:
      ValueError: If detected masks is not None but groundtruth masks are None,
        or the other way around.
    """
    if detected_masks is not None and groundtruth_masks is None:
      raise ValueError(
          'Detected masks is available but groundtruth masks is not.')
    if detected_masks is None and groundtruth_masks is not None:
      raise ValueError(
          'Groundtruth masks is available but detected masks is not.')

    result_scores = []
    result_tp_fp_labels = []
    for i in range(self.num_groundtruth_classes):
      groundtruth_is_difficult_list_at_ith_class = (
          groundtruth_is_difficult_list[groundtruth_class_labels == i])
      groundtruth_is_group_of_list_at_ith_class = (
          groundtruth_is_group_of_list[groundtruth_class_labels == i])
      (gt_boxes_at_ith_class, gt_masks_at_ith_class,
       detected_boxes_at_ith_class, detected_scores_at_ith_class,
       detected_masks_at_ith_class) = self._get_ith_class_arrays(
           detected_boxes, detected_scores, detected_masks,
           detected_class_labels, groundtruth_boxes, groundtruth_masks,
           groundtruth_class_labels, i)
      scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
          detected_boxes=detected_boxes_at_ith_class,
          detected_scores=detected_scores_at_ith_class,
          groundtruth_boxes=gt_boxes_at_ith_class,
          groundtruth_is_difficult_list=
          groundtruth_is_difficult_list_at_ith_class,
          groundtruth_is_group_of_list=
          groundtruth_is_group_of_list_at_ith_class,
          detected_masks=detected_masks_at_ith_class,
          groundtruth_masks=gt_masks_at_ith_class)
      result_scores.append(scores)
      result_tp_fp_labels.append(tp_fp_labels)
    return result_scores, result_tp_fp_labels

  def _get_overlaps_and_scores_mask_mode(
      self, detected_boxes, detected_scores, detected_masks, groundtruth_boxes,
      groundtruth_masks, groundtruth_is_group_of_list):
    """Computes overlaps and scores between detected and groudntruth masks.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      detected_masks: A uint8 numpy array of shape [N, height, width]. If not
          None, the scores will be computed based on masks.
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_masks: A uint8 numpy array of shape [M, height, width].
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag. If a groundtruth box
          is group-of box, every detection matching this box is ignored.

    Returns:
      iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
      ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_group_of_boxlist.num_boxes() == 0 it will be None.
      scores: The score of the detected boxlist.
      num_boxes: Number of non-maximum suppressed detected boxes.
    """
    detected_boxlist = np_box_mask_list.BoxMaskList(
        box_data=detected_boxes, mask_data=detected_masks)
    detected_boxlist.add_field('scores', detected_scores)
    detected_boxlist = np_box_mask_list_ops.non_max_suppression(
        detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
    gt_non_group_of_boxlist = np_box_mask_list.BoxMaskList(
        box_data=groundtruth_boxes[~groundtruth_is_group_of_list],
        mask_data=groundtruth_masks[~groundtruth_is_group_of_list])
    gt_group_of_boxlist = np_box_mask_list.BoxMaskList(
        box_data=groundtruth_boxes[groundtruth_is_group_of_list],
        mask_data=groundtruth_masks[groundtruth_is_group_of_list])
    iou = np_box_mask_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
    ioa = np.transpose(
        np_box_mask_list_ops.ioa(gt_group_of_boxlist, detected_boxlist))
    scores = detected_boxlist.get_field('scores')
    num_boxes = detected_boxlist.num_boxes()
    return iou, ioa, scores, num_boxes

  def _get_overlaps_and_scores_box_mode(
      self,
      detected_boxes,
      detected_scores,
      groundtruth_boxes,
      groundtruth_is_group_of_list):
    """Computes overlaps and scores between detected and groudntruth boxes.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag. If a groundtruth box
          is group-of box, every detection matching this box is ignored.

    Returns:
      iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_non_group_of_boxlist.num_boxes() == 0 it will be None.
      ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If
          gt_group_of_boxlist.num_boxes() == 0 it will be None.
      scores: The score of the detected boxlist.
      num_boxes: Number of non-maximum suppressed detected boxes.
    """
    detected_boxlist = np_box_list.BoxList(detected_boxes)
    detected_boxlist.add_field('scores', detected_scores)
    detected_boxlist = np_box_list_ops.non_max_suppression(
        detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
    gt_non_group_of_boxlist = np_box_list.BoxList(
        groundtruth_boxes[~groundtruth_is_group_of_list])
    gt_group_of_boxlist = np_box_list.BoxList(
        groundtruth_boxes[groundtruth_is_group_of_list])
    iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
    ioa = np.transpose(
        np_box_list_ops.ioa(gt_group_of_boxlist, detected_boxlist))
    scores = detected_boxlist.get_field('scores')
    num_boxes = detected_boxlist.num_boxes()
    return iou, ioa, scores, num_boxes

  def _compute_tp_fp_for_single_class(
      self, detected_boxes, detected_scores, groundtruth_boxes,
      groundtruth_is_difficult_list, groundtruth_is_group_of_list,
      detected_masks=None, groundtruth_masks=None):
    """Labels boxes detected with the same class from the same image as tp/fp.

    Args:
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates
      detected_scores: A 1-d numpy array of length N representing classification
          score
      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
          box coordinates
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not. If a
          groundtruth box is difficult, every detection matching this box
          is ignored.
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box has group-of tag. If a groundtruth box
          is group-of box, every detection matching this box is ignored.
      detected_masks: (optional) A uint8 numpy array of shape
        [N, height, width]. If not None, the scores will be computed based
        on masks.
      groundtruth_masks: (optional) A uint8 numpy array of shape
        [M, height, width].

    Returns:
      Two arrays of the same size, containing all boxes that were evaluated as
      being true positives or false positives; if a box matched to a difficult
      box or to a group-of box, it is ignored.

      scores: A numpy array representing the detection scores.
      tp_fp_labels: a boolean numpy array indicating whether a detection is a
          true positive.
    """
    if detected_boxes.size == 0:
      return np.array([], dtype=float), np.array([], dtype=bool)

    mask_mode = False
    if detected_masks is not None and groundtruth_masks is not None:
      mask_mode = True

    if mask_mode:
      (iou, ioa, scores,
       num_detected_boxes) = self._get_overlaps_and_scores_mask_mode(
           detected_boxes=detected_boxes,
           detected_scores=detected_scores,
           detected_masks=detected_masks,
           groundtruth_boxes=groundtruth_boxes,
           groundtruth_masks=groundtruth_masks,
           groundtruth_is_group_of_list=groundtruth_is_group_of_list)
    else:
      (iou, ioa, scores,
       num_detected_boxes) = self._get_overlaps_and_scores_box_mode(
           detected_boxes=detected_boxes,
           detected_scores=detected_scores,
           groundtruth_boxes=groundtruth_boxes,
           groundtruth_is_group_of_list=groundtruth_is_group_of_list)

    if groundtruth_boxes.size == 0:
      return scores, np.zeros(num_detected_boxes, dtype=bool)

    tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
    is_matched_to_difficult_box = np.zeros(num_detected_boxes, dtype=bool)
    is_matched_to_group_of_box = np.zeros(num_detected_boxes, dtype=bool)

    # The evaluation is done in two stages:
    # 1. All detections are matched to non group-of boxes; true positives are
    #    determined and detections matched to difficult boxes are ignored.
    # 2. Detections that are determined as false positives are matched against
    #    group-of boxes and scored with weight w per ground truth box is
    # matched.

    # Tp-fp evaluation for non-group of boxes (if any).
    if iou.shape[1] > 0:
      groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
          ~groundtruth_is_group_of_list]
      max_overlap_gt_ids = np.argmax(iou, axis=1)
      is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
      for i in range(num_detected_boxes):
        gt_id = max_overlap_gt_ids[i]
        if iou[i, gt_id] >= self.matching_iou_threshold:
          if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
            if not is_gt_box_detected[gt_id]:
              tp_fp_labels[i] = True
              is_gt_box_detected[gt_id] = True
          else:
            is_matched_to_difficult_box[i] = True

    scores_group_of = np.zeros(ioa.shape[1], dtype=float)
    tp_fp_labels_group_of = self.group_of_weight * np.ones(
        ioa.shape[1], dtype=float)
    # Tp-fp evaluation for group of boxes.
    if ioa.shape[1] > 0:
      max_overlap_group_of_gt_ids = np.argmax(ioa, axis=1)
      for i in range(num_detected_boxes):
        gt_id = max_overlap_group_of_gt_ids[i]
        if (not tp_fp_labels[i] and not is_matched_to_difficult_box[i] and
            ioa[i, gt_id] >= self.matching_iou_threshold):
          is_matched_to_group_of_box[i] = True
          scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])
      selector = np.where((scores_group_of > 0) & (tp_fp_labels_group_of > 0))
      scores_group_of = scores_group_of[selector]
      tp_fp_labels_group_of = tp_fp_labels_group_of[selector]

    return np.concatenate(
        (scores[~is_matched_to_difficult_box
                & ~is_matched_to_group_of_box],
         scores_group_of)), np.concatenate(
             (tp_fp_labels[~is_matched_to_difficult_box
                           & ~is_matched_to_group_of_box].astype(float),
              tp_fp_labels_group_of))

  def _get_ith_class_arrays(self, detected_boxes, detected_scores,
                            detected_masks, detected_class_labels,
                            groundtruth_boxes, groundtruth_masks,
                            groundtruth_class_labels, class_index):
    """Returns numpy arrays belonging to class with index `class_index`.

    Args:
      detected_boxes: A numpy array containing detected boxes.
      detected_scores: A numpy array containing detected scores.
      detected_masks: A numpy array containing detected masks.
      detected_class_labels: A numpy array containing detected class labels.
      groundtruth_boxes: A numpy array containing groundtruth boxes.
      groundtruth_masks: A numpy array containing groundtruth masks.
      groundtruth_class_labels: A numpy array containing groundtruth class
        labels.
      class_index: An integer index.

    Returns:
      gt_boxes_at_ith_class: A numpy array containing groundtruth boxes labeled
        as ith class.
      gt_masks_at_ith_class: A numpy array containing groundtruth masks labeled
        as ith class.
      detected_boxes_at_ith_class: A numpy array containing detected boxes
        corresponding to the ith class.
      detected_scores_at_ith_class: A numpy array containing detected scores
        corresponding to the ith class.
      detected_masks_at_ith_class: A numpy array containing detected masks
        corresponding to the ith class.
    """
    selected_groundtruth = (groundtruth_class_labels == class_index)
    gt_boxes_at_ith_class = groundtruth_boxes[selected_groundtruth]
    if groundtruth_masks is not None:
      gt_masks_at_ith_class = groundtruth_masks[selected_groundtruth]
    else:
      gt_masks_at_ith_class = None
    selected_detections = (detected_class_labels == class_index)
    detected_boxes_at_ith_class = detected_boxes[selected_detections]
    detected_scores_at_ith_class = detected_scores[selected_detections]
    if detected_masks is not None:
      detected_masks_at_ith_class = detected_masks[selected_detections]
    else:
      detected_masks_at_ith_class = None
    return (gt_boxes_at_ith_class, gt_masks_at_ith_class,
            detected_boxes_at_ith_class, detected_scores_at_ith_class,
            detected_masks_at_ith_class)

  def _remove_invalid_boxes(self, detected_boxes, detected_scores,
                            detected_class_labels, detected_masks=None):
    """Removes entries with invalid boxes.

    A box is invalid if either its xmax is smaller than its xmin, or its ymax
    is smaller than its ymin.

    Args:
      detected_boxes: A float numpy array of size [num_boxes, 4] containing box
        coordinates in [ymin, xmin, ymax, xmax] format.
      detected_scores: A float numpy array of size [num_boxes].
      detected_class_labels: A int32 numpy array of size [num_boxes].
      detected_masks: A uint8 numpy array of size [num_boxes, height, width].

    Returns:
      valid_detected_boxes: A float numpy array of size [num_valid_boxes, 4]
        containing box coordinates in [ymin, xmin, ymax, xmax] format.
      valid_detected_scores: A float numpy array of size [num_valid_boxes].
      valid_detected_class_labels: A int32 numpy array of size
        [num_valid_boxes].
      valid_detected_masks: A uint8 numpy array of size
        [num_valid_boxes, height, width].
    """
    valid_indices = np.logical_and(detected_boxes[:, 0] < detected_boxes[:, 2],
                                   detected_boxes[:, 1] < detected_boxes[:, 3])
    detected_boxes = detected_boxes[valid_indices]
    detected_scores = detected_scores[valid_indices]
    detected_class_labels = detected_class_labels[valid_indices]
    if detected_masks is not None:
      detected_masks = detected_masks[valid_indices]
    return [
        detected_boxes, detected_scores, detected_class_labels, detected_masks
    ]
