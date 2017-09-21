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

"""object_detection_evaluation module.

ObjectDetectionEvaluation is a class which manages ground truth information of a
object detection dataset, and computes frequently used detection metrics such as
Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.

Note: This module operates on numpy boxes and box lists.
"""

import logging
import numpy as np

from object_detection.utils import metrics
from object_detection.utils import per_image_evaluation


class ObjectDetectionEvaluation(object):
  """Evaluate Object Detection Result."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=1.0,
               nms_max_output_boxes=10000):
    self.per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
        nms_max_output_boxes)
    self.num_class = num_groundtruth_classes

    self.groundtruth_boxes = {}
    self.groundtruth_class_labels = {}
    self.groundtruth_is_difficult_list = {}
    self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=int)
    self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

    self.detection_keys = set()
    self.scores_per_class = [[] for _ in range(self.num_class)]
    self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
    self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
    self.average_precision_per_class = np.empty(self.num_class, dtype=float)
    self.average_precision_per_class.fill(np.nan)
    self.precisions_per_class = []
    self.recalls_per_class = []
    self.corloc_per_class = np.ones(self.num_class, dtype=float)

  def clear_detections(self):
    self.detection_keys = {}
    self.scores_per_class = [[] for _ in range(self.num_class)]
    self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
    self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
    self.average_precision_per_class = np.zeros(self.num_class, dtype=float)
    self.precisions_per_class = []
    self.recalls_per_class = []
    self.corloc_per_class = np.ones(self.num_class, dtype=float)

  def add_single_ground_truth_image_info(self,
                                         image_key,
                                         groundtruth_boxes,
                                         groundtruth_class_labels,
                                         groundtruth_is_difficult_list=None):
    """Add ground truth info of a single image into the evaluation database.

    Args:
      image_key: sha256 key of image content
      groundtruth_boxes: A numpy array of shape [M, 4] representing object box
          coordinates[y_min, x_min, y_max, x_max]
      groundtruth_class_labels: A 1-d numpy array of length M representing class
          labels
      groundtruth_is_difficult_list: A length M numpy boolean array denoting
          whether a ground truth box is a difficult instance or not. To support
          the case that no boxes are difficult, it is by default set as None.
    """
    if image_key in self.groundtruth_boxes:
      logging.warn(
          'image %s has already been added to the ground truth database.',
          image_key)
      return

    self.groundtruth_boxes[image_key] = groundtruth_boxes
    self.groundtruth_class_labels[image_key] = groundtruth_class_labels
    if groundtruth_is_difficult_list is None:
      num_boxes = groundtruth_boxes.shape[0]
      groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
    self.groundtruth_is_difficult_list[
        image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
    self._update_ground_truth_statistics(groundtruth_class_labels,
                                         groundtruth_is_difficult_list)

  def add_single_detected_image_info(self, image_key, detected_boxes,
                                     detected_scores, detected_class_labels):
    """Add detected result of a single image into the evaluation database.

    Args:
      image_key: sha256 key of image content
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates[y_min, x_min, y_max, x_max]
      detected_scores: A 1-d numpy array of length N representing classification
          score
      detected_class_labels: A 1-d numpy array of length N representing class
          labels
    Raises:
      ValueError: if detected_boxes, detected_scores and detected_class_labels
                  do not have the same length.
    """
    if (len(detected_boxes) != len(detected_scores) or
        len(detected_boxes) != len(detected_class_labels)):
      raise ValueError('detected_boxes, detected_scores and '
                       'detected_class_labels should all have same lengths. Got'
                       '[%d, %d, %d]' % len(detected_boxes),
                       len(detected_scores), len(detected_class_labels))

    if image_key in self.detection_keys:
      logging.warn(
          'image %s has already been added to the detection result database',
          image_key)
      return

    self.detection_keys.add(image_key)
    if image_key in self.groundtruth_boxes:
      groundtruth_boxes = self.groundtruth_boxes[image_key]
      groundtruth_class_labels = self.groundtruth_class_labels[image_key]
      groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
          image_key]
    else:
      groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
      groundtruth_class_labels = np.array([], dtype=int)
      groundtruth_is_difficult_list = np.array([], dtype=bool)
    scores, tp_fp_labels, is_class_correctly_detected_in_image = (
        self.per_image_eval.compute_object_detection_metrics(
            detected_boxes, detected_scores, detected_class_labels,
            groundtruth_boxes, groundtruth_class_labels,
            groundtruth_is_difficult_list))
    for i in range(self.num_class):
      self.scores_per_class[i].append(scores[i])
      self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
    (self.num_images_correctly_detected_per_class
    ) += is_class_correctly_detected_in_image

  def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                      groundtruth_is_difficult_list):
    """Update grouth truth statitistics.

    1. Difficult boxes are ignored when counting the number of ground truth
    instances as done in Pascal VOC devkit.
    2. Difficult boxes are treated as normal boxes when computing CorLoc related
    statitistics.

    Args:
      groundtruth_class_labels: An integer numpy array of length M,
          representing M class labels of object instances in ground truth
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
          whether a ground truth box is a difficult instance or not
    """
    for class_index in range(self.num_class):
      num_gt_instances = np.sum(groundtruth_class_labels[
          ~groundtruth_is_difficult_list] == class_index)
      self.num_gt_instances_per_class[class_index] += num_gt_instances
      if np.any(groundtruth_class_labels == class_index):
        self.num_gt_imgs_per_class[class_index] += 1

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      average_precision_per_class: float numpy array of average precision for
          each class.
      mean_ap: mean average precision of all classes, float scalar
      precisions_per_class: List of precisions, each precision is a float numpy
          array
      recalls_per_class: List of recalls, each recall is a float numpy array
      corloc_per_class: numpy float array
      mean_corloc: Mean CorLoc score for each class, float scalar
    """
    if (self.num_gt_instances_per_class == 0).any():
      logging.warn(
          'The following classes have no ground truth examples: %s',
          np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)))
    for class_index in range(self.num_class):
      if self.num_gt_instances_per_class[class_index] == 0:
        continue
      scores = np.concatenate(self.scores_per_class[class_index])
      tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
      precision, recall = metrics.compute_precision_recall(
          scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
      self.precisions_per_class.append(precision)
      self.recalls_per_class.append(recall)
      average_precision = metrics.compute_average_precision(precision, recall)
      self.average_precision_per_class[class_index] = average_precision

    self.corloc_per_class = metrics.compute_cor_loc(
        self.num_gt_imgs_per_class,
        self.num_images_correctly_detected_per_class)

    mean_ap = np.nanmean(self.average_precision_per_class)
    mean_corloc = np.nanmean(self.corloc_per_class)
    return (self.average_precision_per_class, mean_ap,
            self.precisions_per_class, self.recalls_per_class,
            self.corloc_per_class, mean_corloc)

  def get_eval_result(self):
    return EvalResult(self.average_precision_per_class,
                      self.precisions_per_class, self.recalls_per_class,
                      self.corloc_per_class)


class EvalResult(object):

  def __init__(self, average_precisions, precisions, recalls, all_corloc):
    self.precisions = precisions
    self.recalls = recalls
    self.all_corloc = all_corloc
    self.average_precisions = average_precisions
