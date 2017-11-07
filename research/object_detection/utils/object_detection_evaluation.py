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

from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import numpy as np

from object_detection.core import standard_fields
from object_detection.utils import label_map_util
from object_detection.utils import metrics
from object_detection.utils import per_image_evaluation


class DetectionEvaluator(object):
  """Interface for object detection evalution classes.

  Example usage of the Evaluator:
  ------------------------------
  evaluator = DetectionEvaluator(categories)

  # Detections and groundtruth for image 1.
  evaluator.add_single_groundtruth_image_info(...)
  evaluator.add_single_detected_image_info(...)

  # Detections and groundtruth for image 2.
  evaluator.add_single_groundtruth_image_info(...)
  evaluator.add_single_detected_image_info(...)

  metrics_dict = evaluator.evaluate()
  """
  __metaclass__ = ABCMeta

  def __init__(self, categories):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
    """
    self._categories = categories

  @abstractmethod
  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary of groundtruth numpy arrays required
        for evaluations.
    """
    pass

  @abstractmethod
  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary of detection numpy arrays required
        for evaluation.
    """
    pass

  @abstractmethod
  def evaluate(self):
    """Evaluates detections and returns a dictionary of metrics."""
    pass

  @abstractmethod
  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    pass


class ObjectDetectionEvaluator(DetectionEvaluator):
  """A class to evaluate detections."""

  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               metric_prefix=None,
               use_weighted_mean_ap=False):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      evaluate_corlocs: (optional) boolean which determines if corloc scores
        are to be returned or not.
      metric_prefix: (optional) string prefix for metric name; if None, no
        prefix is used.
      use_weighted_mean_ap: (optional) boolean which determines if the mean
        average precision is computed directly from the scores and tp_fp_labels
        of all classes.
    """
    super(ObjectDetectionEvaluator, self).__init__(categories)
    self._num_classes = max([cat['id'] for cat in categories])
    self._matching_iou_threshold = matching_iou_threshold
    self._use_weighted_mean_ap = use_weighted_mean_ap
    self._label_id_offset = 1
    self._evaluation = ObjectDetectionEvaluation(
        self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset)
    self._image_ids = set([])
    self._evaluate_corlocs = evaluate_corlocs
    self._metric_prefix = (metric_prefix + '/') if metric_prefix else ''

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary containing -
        standard_fields.InputDataFields.groundtruth_boxes: float32 numpy array
          of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
          the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
        standard_fields.InputDataFields.groundtruth_classes: integer numpy array
          of shape [num_boxes] containing 1-indexed groundtruth classes for the
          boxes.
        standard_fields.InputDataFields.groundtruth_difficult: Optional length
          M numpy boolean array denoting whether a ground truth box is a
          difficult instance or not. This field is optional to support the case
          that no boxes are difficult.

    Raises:
      ValueError: On adding groundtruth for an image more than once.
    """
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_classes = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_classes]
    groundtruth_classes -= self._label_id_offset
    # If the key is not present in the groundtruth_dict or the array is empty
    # (unless there are no annotations for the groundtruth on this image)
    # use values from the dictionary or insert None otherwise.
    if (standard_fields.InputDataFields.groundtruth_difficult in
        groundtruth_dict.keys() and
        (groundtruth_dict[standard_fields.InputDataFields.groundtruth_difficult]
         .size or not groundtruth_classes.size)):
      groundtruth_difficult = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_difficult]
    else:
      groundtruth_difficult = None
      if not len(self._image_ids) % 1000:
        logging.warn(
            'image %s does not have groundtruth difficult flag specified',
            image_id)
    self._evaluation.add_single_ground_truth_image_info(
        image_id,
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_boxes],
        groundtruth_classes,
        groundtruth_is_difficult_list=groundtruth_difficult)
    self._image_ids.update([image_id])

  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary containing -
        standard_fields.DetectionResultFields.detection_boxes: float32 numpy
          array of shape [num_boxes, 4] containing `num_boxes` detection boxes
          of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
        standard_fields.DetectionResultFields.detection_scores: float32 numpy
          array of shape [num_boxes] containing detection scores for the boxes.
        standard_fields.DetectionResultFields.detection_classes: integer numpy
          array of shape [num_boxes] containing 1-indexed detection classes for
          the boxes.
    """
    detection_classes = detections_dict[
        standard_fields.DetectionResultFields.detection_classes]
    detection_classes -= self._label_id_offset
    self._evaluation.add_single_detected_image_info(
        image_id,
        detections_dict[standard_fields.DetectionResultFields.detection_boxes],
        detections_dict[standard_fields.DetectionResultFields.detection_scores],
        detection_classes)

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A dictionary of metrics with the following fields -

      1. summary_metrics:
        'Precision/mAP@<matching_iou_threshold>IOU': mean average precision at
        the specified IOU threshold.

      2. per_category_ap: category specific results with keys of the form
        'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'.
    """
    (per_class_ap, mean_ap, _, _, per_class_corloc, mean_corloc) = (
        self._evaluation.evaluate())
    pascal_metrics = {
        self._metric_prefix +
        'Precision/mAP@{}IOU'.format(self._matching_iou_threshold):
            mean_ap
    }
    if self._evaluate_corlocs:
      pascal_metrics[self._metric_prefix + 'Precision/meanCorLoc@{}IOU'.format(
          self._matching_iou_threshold)] = mean_corloc
    category_index = label_map_util.create_category_index(self._categories)
    for idx in range(per_class_ap.size):
      if idx + self._label_id_offset in category_index:
        display_name = (
            self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                self._matching_iou_threshold,
                category_index[idx + self._label_id_offset]['name']))
        pascal_metrics[display_name] = per_class_ap[idx]

        # Optionally add CorLoc metrics.classes
        if self._evaluate_corlocs:
          display_name = (
              self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'
              .format(self._matching_iou_threshold,
                      category_index[idx + self._label_id_offset]['name']))
          pascal_metrics[display_name] = per_class_corloc[idx]

    return pascal_metrics

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._evaluation = ObjectDetectionEvaluation(
        self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset)
    self._image_ids.clear()


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using PASCAL metrics."""

  def __init__(self, categories, matching_iou_threshold=0.5):
    super(PascalDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='PASCAL',
        use_weighted_mean_ap=False)


class WeightedPascalDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using weighted PASCAL metrics.

  Weighted PASCAL metrics computes the mean average precision as the average
  precision given the scores and tp_fp_labels of all classes. In comparison,
  PASCAL metrics computes the mean average precision as the mean of the
  per-class average precisions.

  This definition is very similar to the mean of the per-class average
  precisions weighted by class frequency. However, they are typically not the
  same as the average precision is not a linear function of the scores and
  tp_fp_labels.
  """

  def __init__(self, categories, matching_iou_threshold=0.5):
    super(WeightedPascalDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='WeightedPASCAL',
        use_weighted_mean_ap=True)


class OpenImagesDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using Open Images V2 metrics.

    Open Images V2 introduce group_of type of bounding boxes and this metric
    handles those boxes appropriately.
  """

  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      evaluate_corlocs: if True, additionally evaluates and returns CorLoc.
    """
    super(OpenImagesDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold,
        evaluate_corlocs,
        metric_prefix='OpenImagesV2')

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary containing -
        standard_fields.InputDataFields.groundtruth_boxes: float32 numpy array
          of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
          the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
        standard_fields.InputDataFields.groundtruth_classes: integer numpy array
          of shape [num_boxes] containing 1-indexed groundtruth classes for the
          boxes.
        standard_fields.InputDataFields.groundtruth_group_of: Optional length
          M numpy boolean array denoting whether a groundtruth box contains a
          group of instances.

    Raises:
      ValueError: On adding groundtruth for an image more than once.
    """
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_classes = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_classes]
    groundtruth_classes -= self._label_id_offset
    # If the key is not present in the groundtruth_dict or the array is empty
    # (unless there are no annotations for the groundtruth on this image)
    # use values from the dictionary or insert None otherwise.
    if (standard_fields.InputDataFields.groundtruth_group_of in
        groundtruth_dict.keys() and
        (groundtruth_dict[standard_fields.InputDataFields.groundtruth_group_of]
         .size or not groundtruth_classes.size)):
      groundtruth_group_of = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_group_of]
    else:
      groundtruth_group_of = None
      if not len(self._image_ids) % 1000:
        logging.warn(
            'image %s does not have groundtruth group_of flag specified',
            image_id)
    self._evaluation.add_single_ground_truth_image_info(
        image_id,
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_boxes],
        groundtruth_classes,
        groundtruth_is_difficult_list=None,
        groundtruth_is_group_of_list=groundtruth_group_of)
    self._image_ids.update([image_id])


ObjectDetectionEvalMetrics = collections.namedtuple(
    'ObjectDetectionEvalMetrics', [
        'average_precisions', 'mean_ap', 'precisions', 'recalls', 'corlocs',
        'mean_corloc'
    ])


class ObjectDetectionEvaluation(object):
  """Internal implementation of Pascal object detection metrics."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=1.0,
               nms_max_output_boxes=10000,
               use_weighted_mean_ap=False,
               label_id_offset=0):
    self.per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
        nms_max_output_boxes)
    self.num_class = num_groundtruth_classes
    self.label_id_offset = label_id_offset

    self.groundtruth_boxes = {}
    self.groundtruth_class_labels = {}
    self.groundtruth_is_difficult_list = {}
    self.groundtruth_is_group_of_list = {}
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

    self.use_weighted_mean_ap = use_weighted_mean_ap

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
                                         groundtruth_is_difficult_list=None,
                                         groundtruth_is_group_of_list=None):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
        containing `num_boxes` groundtruth boxes of the format
        [ymin, xmin, ymax, xmax] in absolute image coordinates.
      groundtruth_class_labels: integer numpy array of shape [num_boxes]
        containing 0-indexed groundtruth classes for the boxes.
      groundtruth_is_difficult_list: A length M numpy boolean array denoting
        whether a ground truth box is a difficult instance or not. To support
        the case that no boxes are difficult, it is by default set as None.
      groundtruth_is_group_of_list: A length M numpy boolean array denoting
          whether a ground truth box is a group-of box or not. To support
          the case that no boxes are groups-of, it is by default set as None.
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
    if groundtruth_is_group_of_list is None:
      num_boxes = groundtruth_boxes.shape[0]
      groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
    self.groundtruth_is_group_of_list[
        image_key] = groundtruth_is_group_of_list.astype(dtype=bool)

    self._update_ground_truth_statistics(
        groundtruth_class_labels,
        groundtruth_is_difficult_list.astype(dtype=bool),
        groundtruth_is_group_of_list.astype(dtype=bool))

  def add_single_detected_image_info(self, image_key, detected_boxes,
                                     detected_scores, detected_class_labels):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      detected_boxes: float32 numpy array of shape [num_boxes, 4]
        containing `num_boxes` detection boxes of the format
        [ymin, xmin, ymax, xmax] in absolute image coordinates.
      detected_scores: float32 numpy array of shape [num_boxes] containing
        detection scores for the boxes.
      detected_class_labels: integer numpy array of shape [num_boxes] containing
        0-indexed detection classes for the boxes.

    Raises:
      ValueError: if the number of boxes, scores and class labels differ in
        length.
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
      groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
          image_key]
    else:
      groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
      groundtruth_class_labels = np.array([], dtype=int)
      groundtruth_is_difficult_list = np.array([], dtype=bool)
      groundtruth_is_group_of_list = np.array([], dtype=bool)
    scores, tp_fp_labels, is_class_correctly_detected_in_image = (
        self.per_image_eval.compute_object_detection_metrics(
            detected_boxes, detected_scores, detected_class_labels,
            groundtruth_boxes, groundtruth_class_labels,
            groundtruth_is_difficult_list, groundtruth_is_group_of_list))

    for i in range(self.num_class):
      if scores[i].shape[0] > 0:
        self.scores_per_class[i].append(scores[i])
        self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
    (self.num_images_correctly_detected_per_class
    ) += is_class_correctly_detected_in_image

  def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                      groundtruth_is_difficult_list,
                                      groundtruth_is_group_of_list):
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
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
          whether a ground truth box is a group-of box or not
    """
    for class_index in range(self.num_class):
      num_gt_instances = np.sum(groundtruth_class_labels[
          ~groundtruth_is_difficult_list
          & ~groundtruth_is_group_of_list] == class_index)
      self.num_gt_instances_per_class[class_index] += num_gt_instances
      if np.any(groundtruth_class_labels == class_index):
        self.num_gt_imgs_per_class[class_index] += 1

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A named tuple with the following fields -
        average_precision: float numpy array of average precision for
            each class.
        mean_ap: mean average precision of all classes, float scalar
        precisions: List of precisions, each precision is a float numpy
            array
        recalls: List of recalls, each recall is a float numpy array
        corloc: numpy float array
        mean_corloc: Mean CorLoc score for each class, float scalar
    """
    if (self.num_gt_instances_per_class == 0).any():
      logging.warn(
          'The following classes have no ground truth examples: %s',
          np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) +
          self.label_id_offset)

    if self.use_weighted_mean_ap:
      all_scores = np.array([], dtype=float)
      all_tp_fp_labels = np.array([], dtype=bool)

    for class_index in range(self.num_class):
      if self.num_gt_instances_per_class[class_index] == 0:
        continue
      if not self.scores_per_class[class_index]:
        scores = np.array([], dtype=float)
        tp_fp_labels = np.array([], dtype=bool)
      else:
        scores = np.concatenate(self.scores_per_class[class_index])
        tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
      if self.use_weighted_mean_ap:
        all_scores = np.append(all_scores, scores)
        all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
      precision, recall = metrics.compute_precision_recall(
          scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
      self.precisions_per_class.append(precision)
      self.recalls_per_class.append(recall)
      average_precision = metrics.compute_average_precision(precision, recall)
      self.average_precision_per_class[class_index] = average_precision

    self.corloc_per_class = metrics.compute_cor_loc(
        self.num_gt_imgs_per_class,
        self.num_images_correctly_detected_per_class)

    if self.use_weighted_mean_ap:
      num_gt_instances = np.sum(self.num_gt_instances_per_class)
      precision, recall = metrics.compute_precision_recall(
          all_scores, all_tp_fp_labels, num_gt_instances)
      mean_ap = metrics.compute_average_precision(precision, recall)
    else:
      mean_ap = np.nanmean(self.average_precision_per_class)
    mean_corloc = np.nanmean(self.corloc_per_class)
    return ObjectDetectionEvalMetrics(
        self.average_precision_per_class, mean_ap, self.precisions_per_class,
        self.recalls_per_class, self.corloc_per_class, mean_corloc)
