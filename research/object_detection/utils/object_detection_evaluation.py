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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import unicodedata
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.utils import label_map_util
from object_detection.utils import metrics
from object_detection.utils import per_image_evaluation


class DetectionEvaluator(six.with_metaclass(ABCMeta, object)):
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

  def __init__(self, categories):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
    """
    self._categories = categories

  def observe_result_dict_for_single_example(self, eval_dict):
    """Observes an evaluation result dict for a single example.

    When executing eagerly, once all observations have been observed by this
    method you can use `.evaluate()` to get the final metrics.

    When using `tf.estimator.Estimator` for evaluation this function is used by
    `get_estimator_eval_metric_ops()` to construct the metric update op.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating an object
        detection model, returned from
        eval_util.result_dict_for_single_example().

    Returns:
      None when executing eagerly, or an update_op that can be used to update
      the eval metrics in `tf.estimator.EstimatorSpec`.
    """
    raise NotImplementedError('Not implemented for this evaluator!')

  @abstractmethod
  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary of groundtruth numpy arrays required for
        evaluations.
    """
    pass

  @abstractmethod
  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary of detection numpy arrays required for
        evaluation.
    """
    pass

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns dict of metrics to use with `tf.estimator.EstimatorSpec`.

    Note that this must only be implemented if performing evaluation with a
    `tf.estimator.Estimator`.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating an object
        detection model, returned from
        eval_util.result_dict_for_single_example().

    Returns:
      A dictionary of metric names to tuple of value_op and update_op that can
      be used as eval metric ops in `tf.estimator.EstimatorSpec`.
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
               recall_lower_bound=0.0,
               recall_upper_bound=1.0,
               evaluate_corlocs=False,
               evaluate_precision_recall=False,
               metric_prefix=None,
               use_weighted_mean_ap=False,
               evaluate_masks=False,
               group_of_weight=0.0):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      recall_lower_bound: lower bound of recall operating area.
      recall_upper_bound: upper bound of recall operating area.
      evaluate_corlocs: (optional) boolean which determines if corloc scores are
        to be returned or not.
      evaluate_precision_recall: (optional) boolean which determines if
        precision and recall values are to be returned or not.
      metric_prefix: (optional) string prefix for metric name; if None, no
        prefix is used.
      use_weighted_mean_ap: (optional) boolean which determines if the mean
        average precision is computed directly from the scores and tp_fp_labels
        of all classes.
      evaluate_masks: If False, evaluation will be performed based on boxes. If
        True, mask evaluation will be performed instead.
      group_of_weight: Weight of group-of boxes.If set to 0, detections of the
        correct class within a group-of box are ignored. If weight is > 0, then
        if at least one detection falls within a group-of box with
        matching_iou_threshold, weight group_of_weight is added to true
        positives. Consequently, if no detection falls within a group-of box,
        weight group_of_weight is added to false negatives.

    Raises:
      ValueError: If the category ids are not 1-indexed.
    """
    super(ObjectDetectionEvaluator, self).__init__(categories)
    self._num_classes = max([cat['id'] for cat in categories])
    if min(cat['id'] for cat in categories) < 1:
      raise ValueError('Classes should be 1-indexed.')
    self._matching_iou_threshold = matching_iou_threshold
    self._recall_lower_bound = recall_lower_bound
    self._recall_upper_bound = recall_upper_bound
    self._use_weighted_mean_ap = use_weighted_mean_ap
    self._label_id_offset = 1
    self._evaluate_masks = evaluate_masks
    self._group_of_weight = group_of_weight
    self._evaluation = ObjectDetectionEvaluation(
        num_groundtruth_classes=self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        recall_lower_bound=self._recall_lower_bound,
        recall_upper_bound=self._recall_upper_bound,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset,
        group_of_weight=self._group_of_weight)
    self._image_ids = set([])
    self._evaluate_corlocs = evaluate_corlocs
    self._evaluate_precision_recall = evaluate_precision_recall
    self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''
    self._expected_keys = set([
        standard_fields.InputDataFields.key,
        standard_fields.InputDataFields.groundtruth_boxes,
        standard_fields.InputDataFields.groundtruth_classes,
        standard_fields.InputDataFields.groundtruth_difficult,
        standard_fields.InputDataFields.groundtruth_instance_masks,
        standard_fields.DetectionResultFields.detection_boxes,
        standard_fields.DetectionResultFields.detection_scores,
        standard_fields.DetectionResultFields.detection_classes,
        standard_fields.DetectionResultFields.detection_masks
    ])
    self._build_metric_names()

  def _build_metric_names(self):
    """Builds a list with metric names."""
    if self._recall_lower_bound > 0.0 or self._recall_upper_bound < 1.0:
      self._metric_names = [
          self._metric_prefix +
          'Precision/mAP@{}IOU@[{:.1f},{:.1f}]Recall'.format(
              self._matching_iou_threshold, self._recall_lower_bound,
              self._recall_upper_bound)
      ]
    else:
      self._metric_names = [
          self._metric_prefix +
          'Precision/mAP@{}IOU'.format(self._matching_iou_threshold)
      ]
    if self._evaluate_corlocs:
      self._metric_names.append(
          self._metric_prefix +
          'Precision/meanCorLoc@{}IOU'.format(self._matching_iou_threshold))

    category_index = label_map_util.create_category_index(self._categories)
    for idx in range(self._num_classes):
      if idx + self._label_id_offset in category_index:
        category_name = category_index[idx + self._label_id_offset]['name']
        try:
          category_name = six.text_type(category_name, 'utf-8')
        except TypeError:
          pass
        category_name = unicodedata.normalize('NFKD', category_name)
        if six.PY2:
          category_name = category_name.encode('ascii', 'ignore')
        self._metric_names.append(
            self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                self._matching_iou_threshold, category_name))
        if self._evaluate_corlocs:
          self._metric_names.append(
              self._metric_prefix +
              'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                  self._matching_iou_threshold, category_name))

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
        standard_fields.InputDataFields.groundtruth_difficult: Optional length M
          numpy boolean array denoting whether a ground truth box is a difficult
          instance or not. This field is optional to support the case that no
          boxes are difficult.
        standard_fields.InputDataFields.groundtruth_instance_masks: Optional
          numpy array of shape [num_boxes, height, width] with values in {0, 1}.

    Raises:
      ValueError: On adding groundtruth for an image more than once. Will also
        raise error if instance masks are not in groundtruth dictionary.
    """
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_classes = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes] -
        self._label_id_offset)
    # If the key is not present in the groundtruth_dict or the array is empty
    # (unless there are no annotations for the groundtruth on this image)
    # use values from the dictionary or insert None otherwise.
    if (standard_fields.InputDataFields.groundtruth_difficult in six.viewkeys(
        groundtruth_dict) and
        (groundtruth_dict[standard_fields.InputDataFields.groundtruth_difficult]
         .size or not groundtruth_classes.size)):
      groundtruth_difficult = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_difficult]
    else:
      groundtruth_difficult = None
      if not len(self._image_ids) % 1000:
        logging.warning(
            'image %s does not have groundtruth difficult flag specified',
            image_id)
    groundtruth_masks = None
    if self._evaluate_masks:
      if (standard_fields.InputDataFields.groundtruth_instance_masks not in
          groundtruth_dict):
        raise ValueError('Instance masks not in groundtruth dictionary.')
      groundtruth_masks = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_instance_masks]
    self._evaluation.add_single_ground_truth_image_info(
        image_key=image_id,
        groundtruth_boxes=groundtruth_dict[
            standard_fields.InputDataFields.groundtruth_boxes],
        groundtruth_class_labels=groundtruth_classes,
        groundtruth_is_difficult_list=groundtruth_difficult,
        groundtruth_masks=groundtruth_masks)
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
        standard_fields.DetectionResultFields.detection_masks: uint8 numpy array
          of shape [num_boxes, height, width] containing `num_boxes` masks of
          values ranging between 0 and 1.

    Raises:
      ValueError: If detection masks are not in detections dictionary.
    """
    detection_classes = (
        detections_dict[standard_fields.DetectionResultFields.detection_classes]
        - self._label_id_offset)
    detection_masks = None
    if self._evaluate_masks:
      if (standard_fields.DetectionResultFields.detection_masks not in
          detections_dict):
        raise ValueError('Detection masks not in detections dictionary.')
      detection_masks = detections_dict[
          standard_fields.DetectionResultFields.detection_masks]
    self._evaluation.add_single_detected_image_info(
        image_key=image_id,
        detected_boxes=detections_dict[
            standard_fields.DetectionResultFields.detection_boxes],
        detected_scores=detections_dict[
            standard_fields.DetectionResultFields.detection_scores],
        detected_class_labels=detection_classes,
        detected_masks=detection_masks)

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      A dictionary of metrics with the following fields -

      1. summary_metrics:
        '<prefix if not empty>_Precision/mAP@<matching_iou_threshold>IOU': mean
        average precision at the specified IOU threshold.

      2. per_category_ap: category specific results with keys of the form
        '<prefix if not empty>_PerformanceByCategory/
        mAP@<matching_iou_threshold>IOU/category'.
    """
    (per_class_ap, mean_ap, per_class_precision, per_class_recall,
     per_class_corloc, mean_corloc) = (
         self._evaluation.evaluate())
    pascal_metrics = {self._metric_names[0]: mean_ap}
    if self._evaluate_corlocs:
      pascal_metrics[self._metric_names[1]] = mean_corloc
    category_index = label_map_util.create_category_index(self._categories)
    for idx in range(per_class_ap.size):
      if idx + self._label_id_offset in category_index:
        category_name = category_index[idx + self._label_id_offset]['name']
        try:
          category_name = six.text_type(category_name, 'utf-8')
        except TypeError:
          pass
        category_name = unicodedata.normalize('NFKD', category_name)
        if six.PY2:
          category_name = category_name.encode('ascii', 'ignore')
        display_name = (
            self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                self._matching_iou_threshold, category_name))
        pascal_metrics[display_name] = per_class_ap[idx]

        # Optionally add precision and recall values
        if self._evaluate_precision_recall:
          display_name = (
              self._metric_prefix +
              'PerformanceByCategory/Precision@{}IOU/{}'.format(
                  self._matching_iou_threshold, category_name))
          pascal_metrics[display_name] = per_class_precision[idx]
          display_name = (
              self._metric_prefix +
              'PerformanceByCategory/Recall@{}IOU/{}'.format(
                  self._matching_iou_threshold, category_name))
          pascal_metrics[display_name] = per_class_recall[idx]

        # Optionally add CorLoc metrics.classes
        if self._evaluate_corlocs:
          display_name = (
              self._metric_prefix +
              'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                  self._matching_iou_threshold, category_name))
          pascal_metrics[display_name] = per_class_corloc[idx]

    return pascal_metrics

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._evaluation = ObjectDetectionEvaluation(
        num_groundtruth_classes=self._num_classes,
        matching_iou_threshold=self._matching_iou_threshold,
        use_weighted_mean_ap=self._use_weighted_mean_ap,
        label_id_offset=self._label_id_offset)
    self._image_ids.clear()

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns dict of metrics to use with `tf.estimator.EstimatorSpec`.

    Note that this must only be implemented if performing evaluation with a
    `tf.estimator.Estimator`.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating an object
        detection model, returned from
        eval_util.result_dict_for_single_example(). It must contain
        standard_fields.InputDataFields.key.

    Returns:
      A dictionary of metric names to tuple of value_op and update_op that can
      be used as eval metric ops in `tf.estimator.EstimatorSpec`.
    """
    # remove unexpected fields
    eval_dict_filtered = dict()
    for key, value in eval_dict.items():
      if key in self._expected_keys:
        eval_dict_filtered[key] = value

    eval_dict_keys = list(eval_dict_filtered.keys())

    def update_op(image_id, *eval_dict_batched_as_list):
      """Update operation that adds batch of images to ObjectDetectionEvaluator.

      Args:
        image_id: image id (single id or an array)
        *eval_dict_batched_as_list: the values of the dictionary of tensors.
      """
      if np.isscalar(image_id):
        single_example_dict = dict(
            zip(eval_dict_keys, eval_dict_batched_as_list))
        self.add_single_ground_truth_image_info(image_id, single_example_dict)
        self.add_single_detected_image_info(image_id, single_example_dict)
      else:
        for unzipped_tuple in zip(*eval_dict_batched_as_list):
          single_example_dict = dict(zip(eval_dict_keys, unzipped_tuple))
          image_id = single_example_dict[standard_fields.InputDataFields.key]
          self.add_single_ground_truth_image_info(image_id, single_example_dict)
          self.add_single_detected_image_info(image_id, single_example_dict)

    args = [eval_dict_filtered[standard_fields.InputDataFields.key]]
    args.extend(six.itervalues(eval_dict_filtered))
    update_op = tf.py_func(update_op, args, [])

    def first_value_func():
      self._metrics = self.evaluate()
      self.clear()
      return np.float32(self._metrics[self._metric_names[0]])

    def value_func_factory(metric_name):

      def value_func():
        return np.float32(self._metrics[metric_name])

      return value_func

    # Ensure that the metrics are only evaluated once.
    first_value_op = tf.py_func(first_value_func, [], tf.float32)
    eval_metric_ops = {self._metric_names[0]: (first_value_op, update_op)}
    with tf.control_dependencies([first_value_op]):
      for metric_name in self._metric_names[1:]:
        eval_metric_ops[metric_name] = (tf.py_func(
            value_func_factory(metric_name), [], np.float32), update_op)
    return eval_metric_ops


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using PASCAL metrics."""

  def __init__(self, categories, matching_iou_threshold=0.5):
    super(PascalDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='PascalBoxes',
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
        metric_prefix='WeightedPascalBoxes',
        use_weighted_mean_ap=True)


class PrecisionAtRecallDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using precision@recall metrics."""

  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               recall_lower_bound=0.0,
               recall_upper_bound=1.0):
    super(PrecisionAtRecallDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        recall_lower_bound=recall_lower_bound,
        recall_upper_bound=recall_upper_bound,
        evaluate_corlocs=False,
        metric_prefix='PrecisionAtRecallBoxes',
        use_weighted_mean_ap=False)


class PascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate instance masks using PASCAL metrics."""

  def __init__(self, categories, matching_iou_threshold=0.5):
    super(PascalInstanceSegmentationEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='PascalMasks',
        use_weighted_mean_ap=False,
        evaluate_masks=True)


class WeightedPascalInstanceSegmentationEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate instance masks using weighted PASCAL metrics.

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
    super(WeightedPascalInstanceSegmentationEvaluator, self).__init__(
        categories,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        metric_prefix='WeightedPascalMasks',
        use_weighted_mean_ap=True,
        evaluate_masks=True)


class OpenImagesDetectionEvaluator(ObjectDetectionEvaluator):
  """A class to evaluate detections using Open Images V2 metrics.

    Open Images V2 introduce group_of type of bounding boxes and this metric
    handles those boxes appropriately.
  """

  def __init__(self,
               categories,
               matching_iou_threshold=0.5,
               evaluate_masks=False,
               evaluate_corlocs=False,
               metric_prefix='OpenImagesV2',
               group_of_weight=0.0):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      evaluate_masks: if True, evaluator evaluates masks.
      evaluate_corlocs: if True, additionally evaluates and returns CorLoc.
      metric_prefix: Prefix name of the metric.
      group_of_weight: Weight of the group-of bounding box. If set to 0 (default
        for Open Images V2 detection protocol), detections of the correct class
        within a group-of box are ignored. If weight is > 0, then if at least
        one detection falls within a group-of box with matching_iou_threshold,
        weight group_of_weight is added to true positives. Consequently, if no
        detection falls within a group-of box, weight group_of_weight is added
        to false negatives.
    """

    super(OpenImagesDetectionEvaluator, self).__init__(
        categories,
        matching_iou_threshold,
        evaluate_corlocs,
        metric_prefix=metric_prefix,
        group_of_weight=group_of_weight,
        evaluate_masks=evaluate_masks)
    self._expected_keys = set([
        standard_fields.InputDataFields.key,
        standard_fields.InputDataFields.groundtruth_boxes,
        standard_fields.InputDataFields.groundtruth_classes,
        standard_fields.InputDataFields.groundtruth_group_of,
        standard_fields.DetectionResultFields.detection_boxes,
        standard_fields.DetectionResultFields.detection_scores,
        standard_fields.DetectionResultFields.detection_classes,
    ])
    if evaluate_masks:
      self._expected_keys.add(
          standard_fields.InputDataFields.groundtruth_instance_masks)
      self._expected_keys.add(
          standard_fields.DetectionResultFields.detection_masks)

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
        standard_fields.InputDataFields.groundtruth_group_of: Optional length M
          numpy boolean array denoting whether a groundtruth box contains a
          group of instances.

    Raises:
      ValueError: On adding groundtruth for an image more than once.
    """
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_classes = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes] -
        self._label_id_offset)
    # If the key is not present in the groundtruth_dict or the array is empty
    # (unless there are no annotations for the groundtruth on this image)
    # use values from the dictionary or insert None otherwise.
    if (standard_fields.InputDataFields.groundtruth_group_of in six.viewkeys(
        groundtruth_dict) and
        (groundtruth_dict[standard_fields.InputDataFields.groundtruth_group_of]
         .size or not groundtruth_classes.size)):
      groundtruth_group_of = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_group_of]
    else:
      groundtruth_group_of = None
      if not len(self._image_ids) % 1000:
        logging.warning(
            'image %s does not have groundtruth group_of flag specified',
            image_id)
    if self._evaluate_masks:
      groundtruth_masks = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_instance_masks]
    else:
      groundtruth_masks = None

    self._evaluation.add_single_ground_truth_image_info(
        image_id,
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_boxes],
        groundtruth_classes,
        groundtruth_is_difficult_list=None,
        groundtruth_is_group_of_list=groundtruth_group_of,
        groundtruth_masks=groundtruth_masks)
    self._image_ids.update([image_id])


class OpenImagesChallengeEvaluator(OpenImagesDetectionEvaluator):
  """A class implements Open Images Challenge metrics.

    Both Detection and Instance Segmentation evaluation metrics are implemented.

    Open Images Challenge Detection metric has two major changes in comparison
    with Open Images V2 detection metric:
    - a custom weight might be specified for detecting an object contained in
    a group-of box.
    - verified image-level labels should be explicitelly provided for
    evaluation: in case in image has neither positive nor negative image level
    label of class c, all detections of this class on this image will be
    ignored.

    Open Images Challenge Instance Segmentation metric allows to measure per
    formance of models in case of incomplete annotations: some instances are
    annotations only on box level and some - on image-level. In addition,
    image-level labels are taken into account as in detection metric.

    Open Images Challenge Detection metric default parameters:
    evaluate_masks = False
    group_of_weight = 1.0


    Open Images Challenge Instance Segmentation metric default parameters:
    evaluate_masks = True
    (group_of_weight will not matter)
  """

  def __init__(self,
               categories,
               evaluate_masks=False,
               matching_iou_threshold=0.5,
               evaluate_corlocs=False,
               group_of_weight=1.0):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      evaluate_masks: set to true for instance segmentation metric and to false
        for detection metric.
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      evaluate_corlocs: if True, additionally evaluates and returns CorLoc.
      group_of_weight: Weight of group-of boxes. If set to 0, detections of the
        correct class within a group-of box are ignored. If weight is > 0, then
        if at least one detection falls within a group-of box with
        matching_iou_threshold, weight group_of_weight is added to true
        positives. Consequently, if no detection falls within a group-of box,
        weight group_of_weight is added to false negatives.
    """
    if not evaluate_masks:
      metrics_prefix = 'OpenImagesDetectionChallenge'
    else:
      metrics_prefix = 'OpenImagesInstanceSegmentationChallenge'

    super(OpenImagesChallengeEvaluator, self).__init__(
        categories,
        matching_iou_threshold,
        evaluate_masks=evaluate_masks,
        evaluate_corlocs=evaluate_corlocs,
        group_of_weight=group_of_weight,
        metric_prefix=metrics_prefix)

    self._evaluatable_labels = {}
    self._expected_keys.add(
        standard_fields.InputDataFields.groundtruth_image_classes)

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
        standard_fields.InputDataFields.groundtruth_image_classes: integer 1D
          numpy array containing all classes for which labels are verified.
        standard_fields.InputDataFields.groundtruth_group_of: Optional length M
          numpy boolean array denoting whether a groundtruth box contains a
          group of instances.

    Raises:
      ValueError: On adding groundtruth for an image more than once.
    """
    super(OpenImagesChallengeEvaluator,
          self).add_single_ground_truth_image_info(image_id, groundtruth_dict)
    groundtruth_classes = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes] -
        self._label_id_offset)
    self._evaluatable_labels[image_id] = np.unique(
        np.concatenate(((groundtruth_dict.get(
            standard_fields.InputDataFields.groundtruth_image_classes,
            np.array([], dtype=int)) - self._label_id_offset),
                        groundtruth_classes)))

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

    Raises:
      ValueError: If detection masks are not in detections dictionary.
    """
    if image_id not in self._image_ids:
      # Since for the correct work of evaluator it is assumed that groundtruth
      # is inserted first we make sure to break the code if is it not the case.
      self._image_ids.update([image_id])
      self._evaluatable_labels[image_id] = np.array([])

    detection_classes = (
        detections_dict[standard_fields.DetectionResultFields.detection_classes]
        - self._label_id_offset)
    allowed_classes = np.where(
        np.isin(detection_classes, self._evaluatable_labels[image_id]))
    detection_classes = detection_classes[allowed_classes]
    detected_boxes = detections_dict[
        standard_fields.DetectionResultFields.detection_boxes][allowed_classes]
    detected_scores = detections_dict[
        standard_fields.DetectionResultFields.detection_scores][allowed_classes]

    if self._evaluate_masks:
      detection_masks = detections_dict[standard_fields.DetectionResultFields
                                        .detection_masks][allowed_classes]
    else:
      detection_masks = None
    self._evaluation.add_single_detected_image_info(
        image_key=image_id,
        detected_boxes=detected_boxes,
        detected_scores=detected_scores,
        detected_class_labels=detection_classes,
        detected_masks=detection_masks)

  def clear(self):
    """Clears stored data."""

    super(OpenImagesChallengeEvaluator, self).clear()
    self._evaluatable_labels.clear()


ObjectDetectionEvalMetrics = collections.namedtuple(
    'ObjectDetectionEvalMetrics', [
        'average_precisions', 'mean_ap', 'precisions', 'recalls', 'corlocs',
        'mean_corloc'
    ])


class OpenImagesDetectionChallengeEvaluator(OpenImagesChallengeEvaluator):
  """A class implements Open Images Detection Challenge metric."""

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
    super(OpenImagesDetectionChallengeEvaluator, self).__init__(
        categories=categories,
        evaluate_masks=False,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        group_of_weight=1.0)


class OpenImagesInstanceSegmentationChallengeEvaluator(
    OpenImagesChallengeEvaluator):
  """A class implements Open Images Instance Segmentation Challenge metric."""

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
    super(OpenImagesInstanceSegmentationChallengeEvaluator, self).__init__(
        categories=categories,
        evaluate_masks=True,
        matching_iou_threshold=matching_iou_threshold,
        evaluate_corlocs=False,
        group_of_weight=0.0)


class ObjectDetectionEvaluation(object):
  """Internal implementation of Pascal object detection metrics."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_iou_threshold=1.0,
               nms_max_output_boxes=10000,
               recall_lower_bound=0.0,
               recall_upper_bound=1.0,
               use_weighted_mean_ap=False,
               label_id_offset=0,
               group_of_weight=0.0,
               per_image_eval_class=per_image_evaluation.PerImageEvaluation):
    """Constructor.

    Args:
      num_groundtruth_classes: Number of ground-truth classes.
      matching_iou_threshold: IOU threshold used for matching detected boxes to
        ground-truth boxes.
      nms_iou_threshold: IOU threshold used for non-maximum suppression.
      nms_max_output_boxes: Maximum number of boxes returned by non-maximum
        suppression.
      recall_lower_bound: lower bound of recall operating area
      recall_upper_bound: upper bound of recall operating area
      use_weighted_mean_ap: (optional) boolean which determines if the mean
        average precision is computed directly from the scores and tp_fp_labels
        of all classes.
      label_id_offset: The label id offset.
      group_of_weight: Weight of group-of boxes.If set to 0, detections of the
        correct class within a group-of box are ignored. If weight is > 0, then
        if at least one detection falls within a group-of box with
        matching_iou_threshold, weight group_of_weight is added to true
        positives. Consequently, if no detection falls within a group-of box,
        weight group_of_weight is added to false negatives.
      per_image_eval_class: The class that contains functions for computing per
        image metrics.

    Raises:
      ValueError: if num_groundtruth_classes is smaller than 1.
    """
    if num_groundtruth_classes < 1:
      raise ValueError('Need at least 1 groundtruth class for evaluation.')

    self.per_image_eval = per_image_eval_class(
        num_groundtruth_classes=num_groundtruth_classes,
        matching_iou_threshold=matching_iou_threshold,
        nms_iou_threshold=nms_iou_threshold,
        nms_max_output_boxes=nms_max_output_boxes,
        group_of_weight=group_of_weight)
    self.recall_lower_bound = recall_lower_bound
    self.recall_upper_bound = recall_upper_bound
    self.group_of_weight = group_of_weight
    self.num_class = num_groundtruth_classes
    self.use_weighted_mean_ap = use_weighted_mean_ap
    self.label_id_offset = label_id_offset

    self.groundtruth_boxes = {}
    self.groundtruth_class_labels = {}
    self.groundtruth_masks = {}
    self.groundtruth_is_difficult_list = {}
    self.groundtruth_is_group_of_list = {}
    self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
    self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

    self._initialize_detections()

  def _initialize_detections(self):
    """Initializes internal data structures."""
    self.detection_keys = set()
    self.scores_per_class = [[] for _ in range(self.num_class)]
    self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
    self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
    self.average_precision_per_class = np.empty(self.num_class, dtype=float)
    self.average_precision_per_class.fill(np.nan)
    self.precisions_per_class = [np.nan] * self.num_class
    self.recalls_per_class = [np.nan] * self.num_class

    self.corloc_per_class = np.ones(self.num_class, dtype=float)

  def clear_detections(self):
    self._initialize_detections()

  def add_single_ground_truth_image_info(self,
                                         image_key,
                                         groundtruth_boxes,
                                         groundtruth_class_labels,
                                         groundtruth_is_difficult_list=None,
                                         groundtruth_is_group_of_list=None,
                                         groundtruth_masks=None):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      groundtruth_boxes: float32 numpy array of shape [num_boxes, 4] containing
        `num_boxes` groundtruth boxes of the format [ymin, xmin, ymax, xmax] in
        absolute image coordinates.
      groundtruth_class_labels: integer numpy array of shape [num_boxes]
        containing 0-indexed groundtruth classes for the boxes.
      groundtruth_is_difficult_list: A length M numpy boolean array denoting
        whether a ground truth box is a difficult instance or not. To support
        the case that no boxes are difficult, it is by default set as None.
      groundtruth_is_group_of_list: A length M numpy boolean array denoting
        whether a ground truth box is a group-of box or not. To support the case
        that no boxes are groups-of, it is by default set as None.
      groundtruth_masks: uint8 numpy array of shape [num_boxes, height, width]
        containing `num_boxes` groundtruth masks. The mask values range from 0
        to 1.
    """
    if image_key in self.groundtruth_boxes:
      logging.warning(
          'image %s has already been added to the ground truth database.',
          image_key)
      return

    self.groundtruth_boxes[image_key] = groundtruth_boxes
    self.groundtruth_class_labels[image_key] = groundtruth_class_labels
    self.groundtruth_masks[image_key] = groundtruth_masks
    if groundtruth_is_difficult_list is None:
      num_boxes = groundtruth_boxes.shape[0]
      groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
    self.groundtruth_is_difficult_list[
        image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
    if groundtruth_is_group_of_list is None:
      num_boxes = groundtruth_boxes.shape[0]
      groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
    if groundtruth_masks is None:
      num_boxes = groundtruth_boxes.shape[0]
      mask_presence_indicator = np.zeros(num_boxes, dtype=bool)
    else:
      mask_presence_indicator = (np.sum(groundtruth_masks,
                                        axis=(1, 2)) == 0).astype(dtype=bool)

    self.groundtruth_is_group_of_list[
        image_key] = groundtruth_is_group_of_list.astype(dtype=bool)

    self._update_ground_truth_statistics(
        groundtruth_class_labels,
        groundtruth_is_difficult_list.astype(dtype=bool)
        | mask_presence_indicator,  # ignore boxes without masks
        groundtruth_is_group_of_list.astype(dtype=bool))

  def add_single_detected_image_info(self,
                                     image_key,
                                     detected_boxes,
                                     detected_scores,
                                     detected_class_labels,
                                     detected_masks=None):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      detected_boxes: float32 numpy array of shape [num_boxes, 4] containing
        `num_boxes` detection boxes of the format [ymin, xmin, ymax, xmax] in
        absolute image coordinates.
      detected_scores: float32 numpy array of shape [num_boxes] containing
        detection scores for the boxes.
      detected_class_labels: integer numpy array of shape [num_boxes] containing
        0-indexed detection classes for the boxes.
      detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
        containing `num_boxes` detection masks with values ranging between 0 and
        1.

    Raises:
      ValueError: if the number of boxes, scores and class labels differ in
        length.
    """
    if (len(detected_boxes) != len(detected_scores) or
        len(detected_boxes) != len(detected_class_labels)):
      raise ValueError(
          'detected_boxes, detected_scores and '
          'detected_class_labels should all have same lengths. Got'
          '[%d, %d, %d]' % len(detected_boxes), len(detected_scores),
          len(detected_class_labels))

    if image_key in self.detection_keys:
      logging.warning(
          'image %s has already been added to the detection result database',
          image_key)
      return

    self.detection_keys.add(image_key)
    if image_key in self.groundtruth_boxes:
      groundtruth_boxes = self.groundtruth_boxes[image_key]
      groundtruth_class_labels = self.groundtruth_class_labels[image_key]
      # Masks are popped instead of look up. The reason is that we do not want
      # to keep all masks in memory which can cause memory overflow.
      groundtruth_masks = self.groundtruth_masks.pop(image_key)
      groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
          image_key]
      groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
          image_key]
    else:
      groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
      groundtruth_class_labels = np.array([], dtype=int)
      if detected_masks is None:
        groundtruth_masks = None
      else:
        groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
      groundtruth_is_difficult_list = np.array([], dtype=bool)
      groundtruth_is_group_of_list = np.array([], dtype=bool)
    scores, tp_fp_labels, is_class_correctly_detected_in_image = (
        self.per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
            groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks))

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
      groundtruth_class_labels: An integer numpy array of length M, representing
        M class labels of object instances in ground truth
      groundtruth_is_difficult_list: A boolean numpy array of length M denoting
        whether a ground truth box is a difficult instance or not
      groundtruth_is_group_of_list: A boolean numpy array of length M denoting
        whether a ground truth box is a group-of box or not
    """
    for class_index in range(self.num_class):
      num_gt_instances = np.sum(groundtruth_class_labels[
          ~groundtruth_is_difficult_list
          & ~groundtruth_is_group_of_list] == class_index)
      num_groupof_gt_instances = self.group_of_weight * np.sum(
          groundtruth_class_labels[groundtruth_is_group_of_list
                                   & ~groundtruth_is_difficult_list] ==
          class_index)
      self.num_gt_instances_per_class[
          class_index] += num_gt_instances + num_groupof_gt_instances
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
      logging.warning(
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
        tp_fp_labels = np.array([], dtype=float)
      else:
        scores = np.concatenate(self.scores_per_class[class_index])
        tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
      if self.use_weighted_mean_ap:
        all_scores = np.append(all_scores, scores)
        all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
      precision, recall = metrics.compute_precision_recall(
          scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
      recall_within_bound_indices = [
          index for index, value in enumerate(recall) if
          value >= self.recall_lower_bound and value <= self.recall_upper_bound
      ]
      recall_within_bound = recall[recall_within_bound_indices]
      precision_within_bound = precision[recall_within_bound_indices]

      self.precisions_per_class[class_index] = precision_within_bound
      self.recalls_per_class[class_index] = recall_within_bound
      average_precision = metrics.compute_average_precision(
          precision_within_bound, recall_within_bound)
      self.average_precision_per_class[class_index] = average_precision
      logging.info('average_precision: %f', average_precision)

    self.corloc_per_class = metrics.compute_cor_loc(
        self.num_gt_imgs_per_class,
        self.num_images_correctly_detected_per_class)

    if self.use_weighted_mean_ap:
      num_gt_instances = np.sum(self.num_gt_instances_per_class)
      precision, recall = metrics.compute_precision_recall(
          all_scores, all_tp_fp_labels, num_gt_instances)
      recall_within_bound_indices = [
          index for index, value in enumerate(recall) if
          value >= self.recall_lower_bound and value <= self.recall_upper_bound
      ]
      recall_within_bound = recall[recall_within_bound_indices]
      precision_within_bound = precision[recall_within_bound_indices]
      mean_ap = metrics.compute_average_precision(precision_within_bound,
                                                  recall_within_bound)
    else:
      mean_ap = np.nanmean(self.average_precision_per_class)
    mean_corloc = np.nanmean(self.corloc_per_class)
    return ObjectDetectionEvalMetrics(self.average_precision_per_class, mean_ap,
                                      self.precisions_per_class,
                                      self.recalls_per_class,
                                      self.corloc_per_class, mean_corloc)
