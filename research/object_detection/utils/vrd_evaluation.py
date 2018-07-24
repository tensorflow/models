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
"""Evaluator class for Visual Relations Detection.

VRDDetectionEvaluator is a class which manages ground truth information of a
visual relations detection (vrd) dataset, and computes frequently used detection
metrics such as Precision, Recall, Recall@k, of the provided vrd detection
results.
It supports the following operations:
1) Adding ground truth information of images sequentially.
2) Adding detection results of images sequentially.
3) Evaluating detection metrics on already inserted detection results.

Note1: groundtruth should be inserted before evaluation.
Note2: This module operates on numpy boxes and box lists.
"""

from abc import abstractmethod
import collections
import logging
import numpy as np

from object_detection.core import standard_fields
from object_detection.utils import metrics
from object_detection.utils import object_detection_evaluation
from object_detection.utils import per_image_vrd_evaluation

# Below standard input numpy datatypes are defined:
# box_data_type - datatype of the groundtruth visual relations box annotations;
# this datatype consists of two named boxes: subject bounding box and object
# bounding box. Each box is of the format [y_min, x_min, y_max, x_max], each
# coordinate being of type float32.
# label_data_type - corresponding datatype of the visual relations label
# annotaions; it consists of three numerical class labels: subject class label,
# object class label and relation class label, each class label being of type
# int32.
vrd_box_data_type = np.dtype([('subject', 'f4', (4,)), ('object', 'f4', (4,))])
single_box_data_type = np.dtype([('box', 'f4', (4,))])
label_data_type = np.dtype([('subject', 'i4'), ('object', 'i4'), ('relation',
                                                                  'i4')])


class VRDDetectionEvaluator(object_detection_evaluation.DetectionEvaluator):
  """A class to evaluate VRD detections.

  This class serves as a base class for VRD evaluation in two settings:
  - phrase detection
  - relation detection.
  """

  def __init__(self, matching_iou_threshold=0.5, metric_prefix=None):
    """Constructor.

    Args:
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
      metric_prefix: (optional) string prefix for metric name; if None, no
        prefix is used.

    """
    super(VRDDetectionEvaluator, self).__init__([])
    self._matching_iou_threshold = matching_iou_threshold
    self._evaluation = _VRDDetectionEvaluation(
        matching_iou_threshold=self._matching_iou_threshold)
    self._image_ids = set([])
    self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''
    self._evaluatable_labels = {}
    self._negative_labels = {}

  @abstractmethod
  def _process_groundtruth_boxes(self, groundtruth_box_tuples):
    """Pre-processes boxes before adding them to the VRDDetectionEvaluation.

    Phrase detection and Relation detection subclasses re-implement this method
    depending on the task.

    Args:
      groundtruth_box_tuples:  A numpy array of structures with the shape
        [M, 1], each structure containing the same number of named bounding
        boxes. Each box is of the format [y_min, x_min, y_max, x_max] (see
        datatype vrd_box_data_type, single_box_data_type above).
    """
    raise NotImplementedError(
        '_process_groundtruth_boxes method should be implemented in subclasses'
        'of VRDDetectionEvaluator.')

  @abstractmethod
  def _process_detection_boxes(self, detections_box_tuples):
    """Pre-processes boxes before adding them to the VRDDetectionEvaluation.

    Phrase detection and Relation detection subclasses re-implement this method
    depending on the task.

    Args:
      detections_box_tuples:  A numpy array of structures with the shape
        [M, 1], each structure containing the same number of named bounding
        boxes. Each box is of the format [y_min, x_min, y_max, x_max] (see
        datatype vrd_box_data_type, single_box_data_type above).
    """
    raise NotImplementedError(
        '_process_detection_boxes method should be implemented in subclasses'
        'of VRDDetectionEvaluator.')

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary containing -
        standard_fields.InputDataFields.groundtruth_boxes: A numpy array
          of structures with the shape [M, 1], representing M tuples, each tuple
          containing the same number of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max] (see
          datatype vrd_box_data_type, single_box_data_type above).
        standard_fields.InputDataFields.groundtruth_classes: A numpy array of
          structures shape [M, 1], representing  the class labels of the
          corresponding bounding boxes and possibly additional classes (see
          datatype label_data_type above).
        standard_fields.InputDataFields.groundtruth_image_classes: numpy array
          of shape [K] containing verified labels.
    Raises:
      ValueError: On adding groundtruth for an image more than once.
    """
    if image_id in self._image_ids:
      raise ValueError('Image with id {} already added.'.format(image_id))

    groundtruth_class_tuples = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_classes])
    groundtruth_box_tuples = (
        groundtruth_dict[standard_fields.InputDataFields.groundtruth_boxes])

    self._evaluation.add_single_ground_truth_image_info(
        image_key=image_id,
        groundtruth_box_tuples=self._process_groundtruth_boxes(
            groundtruth_box_tuples),
        groundtruth_class_tuples=groundtruth_class_tuples)
    self._image_ids.update([image_id])
    all_classes = []
    for field in groundtruth_box_tuples.dtype.fields:
      all_classes.append(groundtruth_class_tuples[field])
    groudtruth_positive_classes = np.unique(np.concatenate(all_classes))
    verified_labels = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_image_classes,
        np.array([], dtype=int))
    self._evaluatable_labels[image_id] = np.unique(
        np.concatenate((verified_labels, groudtruth_positive_classes)))

    self._negative_labels[image_id] = np.setdiff1d(verified_labels,
                                                   groudtruth_positive_classes)

  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary containing -
        standard_fields.DetectionResultFields.detection_boxes: A numpy array of
          structures with shape [N, 1], representing N tuples, each tuple
          containing the same number of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max] (as an example
          see datatype vrd_box_data_type, single_box_data_type above).
        standard_fields.DetectionResultFields.detection_scores: float32 numpy
          array of shape [N] containing detection scores for the boxes.
        standard_fields.DetectionResultFields.detection_classes: A numpy array
          of structures shape [N, 1], representing the class labels of the
          corresponding bounding boxes and possibly additional classes (see
          datatype label_data_type above).
    """
    if image_id not in self._image_ids:
      logging.warn('No groundtruth for the image with id %s.', image_id)
      # Since for the correct work of evaluator it is assumed that groundtruth
      # is inserted first we make sure to break the code if is it not the case.
      self._image_ids.update([image_id])
      self._negative_labels[image_id] = np.array([])
      self._evaluatable_labels[image_id] = np.array([])

    num_detections = detections_dict[
        standard_fields.DetectionResultFields.detection_boxes].shape[0]
    detection_class_tuples = detections_dict[
        standard_fields.DetectionResultFields.detection_classes]
    detection_box_tuples = detections_dict[
        standard_fields.DetectionResultFields.detection_boxes]
    negative_selector = np.zeros(num_detections, dtype=bool)
    selector = np.ones(num_detections, dtype=bool)
    # Only check boxable labels
    for field in detection_box_tuples.dtype.fields:
      # Verify if one of the labels is negative (this is sure FP)
      negative_selector |= np.isin(detection_class_tuples[field],
                                   self._negative_labels[image_id])
      # Verify if all labels are verified
      selector &= np.isin(detection_class_tuples[field],
                          self._evaluatable_labels[image_id])
    selector |= negative_selector
    self._evaluation.add_single_detected_image_info(
        image_key=image_id,
        detected_box_tuples=self._process_detection_boxes(
            detection_box_tuples[selector]),
        detected_scores=detections_dict[
            standard_fields.DetectionResultFields.detection_scores][selector],
        detected_class_tuples=detection_class_tuples[selector])

  def evaluate(self, relationships=None):
    """Compute evaluation result.

    Args:
      relationships: A dictionary of numerical label-text label mapping; if
        specified, returns per-relationship AP.

    Returns:
      A dictionary of metrics with the following fields -

      summary_metrics:
        'weightedAP@<matching_iou_threshold>IOU' : weighted average precision
        at the specified IOU threshold.
        'AP@<matching_iou_threshold>IOU/<relationship>' : AP per relationship.
        'mAP@<matching_iou_threshold>IOU': mean average precision at the
        specified IOU threshold.
        'Recall@50@<matching_iou_threshold>IOU': recall@50 at the specified IOU
        threshold.
        'Recall@100@<matching_iou_threshold>IOU': recall@100 at the specified
        IOU threshold.
      if relationships is specified, returns <relationship> in AP metrics as
      readable names, otherwise the names correspond to class numbers.
    """
    (weighted_average_precision, mean_average_precision, average_precisions, _,
     _, recall_50, recall_100, _, _) = (
         self._evaluation.evaluate())

    vrd_metrics = {
        (self._metric_prefix + 'weightedAP@{}IOU'.format(
            self._matching_iou_threshold)):
            weighted_average_precision,
        self._metric_prefix + 'mAP@{}IOU'.format(self._matching_iou_threshold):
            mean_average_precision,
        self._metric_prefix + 'Recall@50@{}IOU'.format(
            self._matching_iou_threshold):
            recall_50,
        self._metric_prefix + 'Recall@100@{}IOU'.format(
            self._matching_iou_threshold):
            recall_100,
    }
    if relationships:
      for key, average_precision in average_precisions.iteritems():
        vrd_metrics[self._metric_prefix + 'AP@{}IOU/{}'.format(
            self._matching_iou_threshold,
            relationships[key])] = average_precision
    else:
      for key, average_precision in average_precisions.iteritems():
        vrd_metrics[self._metric_prefix + 'AP@{}IOU/{}'.format(
            self._matching_iou_threshold, key)] = average_precision

    return vrd_metrics

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._evaluation = _VRDDetectionEvaluation(
        matching_iou_threshold=self._matching_iou_threshold)
    self._image_ids.clear()
    self._negative_labels.clear()
    self._evaluatable_labels.clear()


class VRDRelationDetectionEvaluator(VRDDetectionEvaluator):
  """A class to evaluate VRD detections in relations setting.

  Expected groundtruth box datatype is vrd_box_data_type, expected groudtruth
  labels datatype is label_data_type.
  Expected detection box datatype is vrd_box_data_type, expected detection
  labels
  datatype is label_data_type.
  """

  def __init__(self, matching_iou_threshold=0.5):
    super(VRDRelationDetectionEvaluator, self).__init__(
        matching_iou_threshold=matching_iou_threshold,
        metric_prefix='VRDMetric_Relationships')

  def _process_groundtruth_boxes(self, groundtruth_box_tuples):
    """Pre-processes boxes before adding them to the VRDDetectionEvaluation.

    Args:
      groundtruth_box_tuples: A numpy array of structures with the shape
        [M, 1], each structure containing the same number of named bounding
        boxes. Each box is of the format [y_min, x_min, y_max, x_max].

    Returns:
      Unchanged input.
    """

    return groundtruth_box_tuples

  def _process_detection_boxes(self, detections_box_tuples):
    """Pre-processes boxes before adding them to the VRDDetectionEvaluation.

    Phrase detection and Relation detection subclasses re-implement this method
    depending on the task.

    Args:
      detections_box_tuples:  A numpy array of structures with the shape
        [M, 1], each structure containing the same number of named bounding
        boxes. Each box is of the format [y_min, x_min, y_max, x_max] (see
        datatype vrd_box_data_type, single_box_data_type above).
    Returns:
      Unchanged input.
    """
    return detections_box_tuples


class VRDPhraseDetectionEvaluator(VRDDetectionEvaluator):
  """A class to evaluate VRD detections in phrase setting.

  Expected groundtruth box datatype is vrd_box_data_type, expected groudtruth
  labels datatype is label_data_type.
  Expected detection box datatype is single_box_data_type, expected detection
  labels datatype is label_data_type.
  """

  def __init__(self, matching_iou_threshold=0.5):
    super(VRDPhraseDetectionEvaluator, self).__init__(
        matching_iou_threshold=matching_iou_threshold,
        metric_prefix='VRDMetric_Phrases')

  def _process_groundtruth_boxes(self, groundtruth_box_tuples):
    """Pre-processes boxes before adding them to the VRDDetectionEvaluation.

    In case of phrase evaluation task, evaluation expects exactly one bounding
    box containing all objects in the phrase. This bounding box is computed
    as an enclosing box of all groundtruth boxes of a phrase.

    Args:
      groundtruth_box_tuples: A numpy array of structures with the shape
        [M, 1], each structure containing the same number of named bounding
        boxes. Each box is of the format [y_min, x_min, y_max, x_max]. See
        vrd_box_data_type for an example of structure.

    Returns:
      result: A numpy array of structures with the shape [M, 1], each
        structure containing exactly one named bounding box. i-th output
        structure corresponds to the result of processing i-th input structure,
        where the named bounding box is computed as an enclosing bounding box
        of all bounding boxes of the i-th input structure.
    """
    first_box_key = groundtruth_box_tuples.dtype.fields.keys()[0]
    miny = groundtruth_box_tuples[first_box_key][:, 0]
    minx = groundtruth_box_tuples[first_box_key][:, 1]
    maxy = groundtruth_box_tuples[first_box_key][:, 2]
    maxx = groundtruth_box_tuples[first_box_key][:, 3]
    for fields in groundtruth_box_tuples.dtype.fields:
      miny = np.minimum(groundtruth_box_tuples[fields][:, 0], miny)
      minx = np.minimum(groundtruth_box_tuples[fields][:, 1], minx)
      maxy = np.maximum(groundtruth_box_tuples[fields][:, 2], maxy)
      maxx = np.maximum(groundtruth_box_tuples[fields][:, 3], maxx)
    data_result = []
    for i in range(groundtruth_box_tuples.shape[0]):
      data_result.append(([miny[i], minx[i], maxy[i], maxx[i]],))
    result = np.array(data_result, dtype=[('box', 'f4', (4,))])
    return result

  def _process_detection_boxes(self, detections_box_tuples):
    """Pre-processes boxes before adding them to the VRDDetectionEvaluation.

    In case of phrase evaluation task, evaluation expects exactly one bounding
    box containing all objects in the phrase. This bounding box is computed
    as an enclosing box of all groundtruth boxes of a phrase.

    Args:
      detections_box_tuples: A numpy array of structures with the shape
        [M, 1], each structure containing the same number of named bounding
        boxes. Each box is of the format [y_min, x_min, y_max, x_max]. See
        vrd_box_data_type for an example of this structure.

    Returns:
      result: A numpy array of structures with the shape [M, 1], each
        structure containing exactly one named bounding box. i-th output
        structure corresponds to the result of processing i-th input structure,
        where the named bounding box is computed as an enclosing bounding box
        of all bounding boxes of the i-th input structure.
    """
    first_box_key = detections_box_tuples.dtype.fields.keys()[0]
    miny = detections_box_tuples[first_box_key][:, 0]
    minx = detections_box_tuples[first_box_key][:, 1]
    maxy = detections_box_tuples[first_box_key][:, 2]
    maxx = detections_box_tuples[first_box_key][:, 3]
    for fields in detections_box_tuples.dtype.fields:
      miny = np.minimum(detections_box_tuples[fields][:, 0], miny)
      minx = np.minimum(detections_box_tuples[fields][:, 1], minx)
      maxy = np.maximum(detections_box_tuples[fields][:, 2], maxy)
      maxx = np.maximum(detections_box_tuples[fields][:, 3], maxx)
    data_result = []
    for i in range(detections_box_tuples.shape[0]):
      data_result.append(([miny[i], minx[i], maxy[i], maxx[i]],))
    result = np.array(data_result, dtype=[('box', 'f4', (4,))])
    return result


VRDDetectionEvalMetrics = collections.namedtuple('VRDDetectionEvalMetrics', [
    'weighted_average_precision', 'mean_average_precision',
    'average_precisions', 'precisions', 'recalls', 'recall_50', 'recall_100',
    'median_rank_50', 'median_rank_100'
])


class _VRDDetectionEvaluation(object):
  """Performs metric computation for the VRD task. This class is internal.
  """

  def __init__(self, matching_iou_threshold=0.5):
    """Constructor.

    Args:
      matching_iou_threshold: IOU threshold to use for matching groundtruth
        boxes to detection boxes.
    """
    self._per_image_eval = per_image_vrd_evaluation.PerImageVRDEvaluation(
        matching_iou_threshold=matching_iou_threshold)

    self._groundtruth_box_tuples = {}
    self._groundtruth_class_tuples = {}
    self._num_gt_instances = 0
    self._num_gt_imgs = 0
    self._num_gt_instances_per_relationship = {}

    self.clear_detections()

  def clear_detections(self):
    """Clears detections."""
    self._detection_keys = set()
    self._scores = []
    self._relation_field_values = []
    self._tp_fp_labels = []
    self._average_precisions = {}
    self._precisions = []
    self._recalls = []

  def add_single_ground_truth_image_info(
      self, image_key, groundtruth_box_tuples, groundtruth_class_tuples):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      groundtruth_box_tuples: A numpy array of structures with the shape
          [M, 1], representing M tuples, each tuple containing the same number
          of named bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max].
      groundtruth_class_tuples: A numpy array of structures shape [M, 1],
          representing  the class labels of the corresponding bounding boxes and
          possibly additional classes.
    """
    if image_key in self._groundtruth_box_tuples:
      logging.warn(
          'image %s has already been added to the ground truth database.',
          image_key)
      return

    self._groundtruth_box_tuples[image_key] = groundtruth_box_tuples
    self._groundtruth_class_tuples[image_key] = groundtruth_class_tuples

    self._update_groundtruth_statistics(groundtruth_class_tuples)

  def add_single_detected_image_info(self, image_key, detected_box_tuples,
                                     detected_scores, detected_class_tuples):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_key: A unique string/integer identifier for the image.
      detected_box_tuples: A numpy array of structures with shape [N, 1],
          representing N tuples, each tuple containing the same number of named
          bounding boxes.
          Each box is of the format [y_min, x_min, y_max, x_max].
      detected_scores: A float numpy array of shape [N, 1], representing
          the confidence scores of the detected N object instances.
      detected_class_tuples: A numpy array of structures shape [N, 1],
          representing the class labels of the corresponding bounding boxes and
          possibly additional classes.
    """
    self._detection_keys.add(image_key)
    if image_key in self._groundtruth_box_tuples:
      groundtruth_box_tuples = self._groundtruth_box_tuples[image_key]
      groundtruth_class_tuples = self._groundtruth_class_tuples[image_key]
    else:
      groundtruth_box_tuples = np.empty(
          shape=[0, 4], dtype=detected_box_tuples.dtype)
      groundtruth_class_tuples = np.array([], dtype=detected_class_tuples.dtype)

    scores, tp_fp_labels, mapping = (
        self._per_image_eval.compute_detection_tp_fp(
            detected_box_tuples=detected_box_tuples,
            detected_scores=detected_scores,
            detected_class_tuples=detected_class_tuples,
            groundtruth_box_tuples=groundtruth_box_tuples,
            groundtruth_class_tuples=groundtruth_class_tuples))

    self._scores += [scores]
    self._tp_fp_labels += [tp_fp_labels]
    self._relation_field_values += [detected_class_tuples[mapping]['relation']]

  def _update_groundtruth_statistics(self, groundtruth_class_tuples):
    """Updates grouth truth statistics.

    Args:
      groundtruth_class_tuples: A numpy array of structures shape [M, 1],
          representing  the class labels of the corresponding bounding boxes and
          possibly additional classes.
    """
    self._num_gt_instances += groundtruth_class_tuples.shape[0]
    self._num_gt_imgs += 1
    for relation_field_value in np.unique(groundtruth_class_tuples['relation']):
      if relation_field_value not in self._num_gt_instances_per_relationship:
        self._num_gt_instances_per_relationship[relation_field_value] = 0
      self._num_gt_instances_per_relationship[relation_field_value] += np.sum(
          groundtruth_class_tuples['relation'] == relation_field_value)

  def evaluate(self):
    """Computes evaluation result.

    Returns:
      A named tuple with the following fields -
        average_precision: a float number corresponding to average precision.
        precisions: an array of precisions.
        recalls: an array of recalls.
        recall@50: recall computed on 50 top-scoring samples.
        recall@100: recall computed on 100 top-scoring samples.
        median_rank@50: median rank computed on 50 top-scoring samples.
        median_rank@100: median rank computed on 100 top-scoring samples.
    """
    if self._num_gt_instances == 0:
      logging.warn('No ground truth instances')

    if not self._scores:
      scores = np.array([], dtype=float)
      tp_fp_labels = np.array([], dtype=bool)
    else:
      scores = np.concatenate(self._scores)
      tp_fp_labels = np.concatenate(self._tp_fp_labels)
      relation_field_values = np.concatenate(self._relation_field_values)

    for relation_field_value, _ in (
        self._num_gt_instances_per_relationship.iteritems()):
      precisions, recalls = metrics.compute_precision_recall(
          scores[relation_field_values == relation_field_value],
          tp_fp_labels[relation_field_values == relation_field_value],
          self._num_gt_instances_per_relationship[relation_field_value])
      self._average_precisions[
          relation_field_value] = metrics.compute_average_precision(
              precisions, recalls)

    self._mean_average_precision = np.mean(self._average_precisions.values())

    self._precisions, self._recalls = metrics.compute_precision_recall(
        scores, tp_fp_labels, self._num_gt_instances)
    self._weighted_average_precision = metrics.compute_average_precision(
        self._precisions, self._recalls)

    self._recall_50 = (
        metrics.compute_recall_at_k(self._tp_fp_labels, self._num_gt_instances,
                                    50))
    self._median_rank_50 = (
        metrics.compute_median_rank_at_k(self._tp_fp_labels, 50))
    self._recall_100 = (
        metrics.compute_recall_at_k(self._tp_fp_labels, self._num_gt_instances,
                                    100))
    self._median_rank_100 = (
        metrics.compute_median_rank_at_k(self._tp_fp_labels, 100))

    return VRDDetectionEvalMetrics(
        self._weighted_average_precision, self._mean_average_precision,
        self._average_precisions, self._precisions, self._recalls,
        self._recall_50, self._recall_100, self._median_rank_50,
        self._median_rank_100)
