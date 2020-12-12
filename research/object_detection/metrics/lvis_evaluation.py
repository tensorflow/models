# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Class for evaluating object detections with LVIS metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re

from lvis import results as lvis_results

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import standard_fields as fields
from object_detection.metrics import lvis_tools
from object_detection.utils import object_detection_evaluation


def convert_masks_to_binary(masks):
  """Converts masks to 0 or 1 and uint8 type."""
  return (masks > 0).astype(np.uint8)


class LVISMaskEvaluator(object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate LVIS mask metrics."""

  def __init__(self,
               categories,
               include_metrics_per_category=False,
               export_path=None):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      include_metrics_per_category: Additionally include per-category metrics
        (this option is currently unsupported).
      export_path: Path to export detections to LVIS compatible JSON format.
    """
    super(LVISMaskEvaluator, self).__init__(categories)
    self._image_ids_with_detections = set([])
    self._groundtruth_list = []
    self._detection_masks_list = []
    self._category_id_set = set([cat['id'] for cat in self._categories])
    self._annotation_id = 1
    self._image_id_to_mask_shape_map = {}
    self._image_id_to_verified_neg_classes = {}
    self._image_id_to_not_exhaustive_classes = {}
    if include_metrics_per_category:
      raise ValueError('include_metrics_per_category not yet supported '
                       'for LVISMaskEvaluator.')
    self._export_path = export_path

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._image_id_to_mask_shape_map.clear()
    self._image_ids_with_detections.clear()
    self._image_id_to_verified_neg_classes.clear()
    self._image_id_to_not_exhaustive_classes.clear()
    self._groundtruth_list = []
    self._detection_masks_list = []

  def add_single_ground_truth_image_info(self,
                                         image_id,
                                         groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    If the image has already been added, a warning is logged, and groundtruth is
    ignored.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary containing -
        InputDataFields.groundtruth_boxes: float32 numpy array of shape
          [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
          [ymin, xmin, ymax, xmax] in absolute image coordinates.
        InputDataFields.groundtruth_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed groundtruth classes for the boxes.
        InputDataFields.groundtruth_instance_masks: uint8 numpy array of shape
          [num_masks, image_height, image_width] containing groundtruth masks.
          The elements of the array must be in {0, 1}.
        InputDataFields.groundtruth_verified_neg_classes: [num_classes + 1]
          float indicator vector with values in {0, 1}. The length is
          num_classes + 1 so as to be compatible with the 1-indexed groundtruth
          classes.
        InputDataFields.groundtruth_not_exhaustive_classes: [num_classes + 1]
          float indicator vector with values in {0, 1}. The length is
          num_classes + 1 so as to be compatible with the 1-indexed groundtruth
          classes.
        InputDataFields.groundtruth_area (optional): float numpy array of
          shape [num_boxes] containing the area (in the original absolute
          coordinates) of the annotated object.
    Raises:
      ValueError: if groundtruth_dict is missing a required field
    """
    if image_id in self._image_id_to_mask_shape_map:
      tf.logging.warning('Ignoring ground truth with image id %s since it was '
                         'previously added', image_id)
      return
    for key in [fields.InputDataFields.groundtruth_boxes,
                fields.InputDataFields.groundtruth_classes,
                fields.InputDataFields.groundtruth_instance_masks,
                fields.InputDataFields.groundtruth_verified_neg_classes,
                fields.InputDataFields.groundtruth_not_exhaustive_classes]:
      if key not in groundtruth_dict.keys():
        raise ValueError('groundtruth_dict missing entry: {}'.format(key))

    groundtruth_instance_masks = groundtruth_dict[
        fields.InputDataFields.groundtruth_instance_masks]
    groundtruth_instance_masks = convert_masks_to_binary(
        groundtruth_instance_masks)
    verified_neg_classes_shape = groundtruth_dict[
        fields.InputDataFields.groundtruth_verified_neg_classes].shape
    not_exhaustive_classes_shape = groundtruth_dict[
        fields.InputDataFields.groundtruth_not_exhaustive_classes].shape
    if verified_neg_classes_shape != (len(self._category_id_set) + 1,):
      raise ValueError('Invalid shape for verified_neg_classes_shape.')
    if not_exhaustive_classes_shape != (len(self._category_id_set) + 1,):
      raise ValueError('Invalid shape for not_exhaustive_classes_shape.')
    self._image_id_to_verified_neg_classes[image_id] = np.flatnonzero(
        groundtruth_dict[
            fields.InputDataFields.groundtruth_verified_neg_classes]
        == 1).tolist()
    self._image_id_to_not_exhaustive_classes[image_id] = np.flatnonzero(
        groundtruth_dict[
            fields.InputDataFields.groundtruth_not_exhaustive_classes]
        == 1).tolist()

    # Drop optional fields if empty tensor.
    groundtruth_area = groundtruth_dict.get(
        fields.InputDataFields.groundtruth_area)
    if groundtruth_area is not None and not groundtruth_area.shape[0]:
      groundtruth_area = None

    self._groundtruth_list.extend(
        lvis_tools.ExportSingleImageGroundtruthToLVIS(
            image_id=image_id,
            next_annotation_id=self._annotation_id,
            category_id_set=self._category_id_set,
            groundtruth_boxes=groundtruth_dict[
                fields.InputDataFields.groundtruth_boxes],
            groundtruth_classes=groundtruth_dict[
                fields.InputDataFields.groundtruth_classes],
            groundtruth_masks=groundtruth_instance_masks,
            groundtruth_area=groundtruth_area)
    )

    self._annotation_id += groundtruth_dict[fields.InputDataFields.
                                            groundtruth_boxes].shape[0]
    self._image_id_to_mask_shape_map[image_id] = groundtruth_dict[
        fields.InputDataFields.groundtruth_instance_masks].shape

  def add_single_detected_image_info(self,
                                     image_id,
                                     detections_dict):
    """Adds detections for a single image to be used for evaluation.

    If a detection has already been added for this image id, a warning is
    logged, and the detection is skipped.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary containing -
        DetectionResultFields.detection_scores: float32 numpy array of shape
          [num_boxes] containing detection scores for the boxes.
        DetectionResultFields.detection_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed detection classes for the boxes.
        DetectionResultFields.detection_masks: optional uint8 numpy array of
          shape [num_boxes, image_height, image_width] containing instance
          masks corresponding to the boxes. The elements of the array must be
          in {0, 1}.
    Raises:
      ValueError: If groundtruth for the image_id is not available.
    """
    if image_id not in self._image_id_to_mask_shape_map:
      raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

    if image_id in self._image_ids_with_detections:
      tf.logging.warning('Ignoring detection with image id %s since it was '
                         'previously added', image_id)
      return

    groundtruth_masks_shape = self._image_id_to_mask_shape_map[image_id]
    detection_masks = detections_dict[fields.DetectionResultFields.
                                      detection_masks]
    if groundtruth_masks_shape[1:] != detection_masks.shape[1:]:
      raise ValueError('Spatial shape of groundtruth masks and detection masks '
                       'are incompatible: {} vs {}'.format(
                           groundtruth_masks_shape,
                           detection_masks.shape))
    detection_masks = convert_masks_to_binary(detection_masks)

    self._detection_masks_list.extend(
        lvis_tools.ExportSingleImageDetectionMasksToLVIS(
            image_id=image_id,
            category_id_set=self._category_id_set,
            detection_masks=detection_masks,
            detection_scores=detections_dict[
                fields.DetectionResultFields.detection_scores],
            detection_classes=detections_dict[
                fields.DetectionResultFields.detection_classes]))
    self._image_ids_with_detections.update([image_id])

  def evaluate(self):
    """Evaluates the detection boxes and returns a dictionary of coco metrics.

    Returns:
      A dictionary holding
    """
    if self._export_path:
      tf.logging.info('Dumping detections to json.')
      self.dump_detections_to_json_file(self._export_path)
    tf.logging.info('Performing evaluation on %d images.',
                    len(self._image_id_to_mask_shape_map.keys()))
    # pylint: disable=g-complex-comprehension
    groundtruth_dict = {
        'annotations': self._groundtruth_list,
        'images': [
            {
                'id': int(image_id),
                'height': shape[1],
                'width': shape[2],
                'neg_category_ids':
                    self._image_id_to_verified_neg_classes[image_id],
                'not_exhaustive_category_ids':
                    self._image_id_to_not_exhaustive_classes[image_id]
            } for image_id, shape in self._image_id_to_mask_shape_map.items()],
        'categories': self._categories
    }
    # pylint: enable=g-complex-comprehension
    lvis_wrapped_groundtruth = lvis_tools.LVISWrapper(groundtruth_dict)
    detections = lvis_results.LVISResults(lvis_wrapped_groundtruth,
                                          self._detection_masks_list)
    mask_evaluator = lvis_tools.LVISEvalWrapper(
        lvis_wrapped_groundtruth, detections, iou_type='segm')
    mask_metrics = mask_evaluator.ComputeMetrics()
    mask_metrics = {'DetectionMasks_'+ key: value
                    for key, value in iter(mask_metrics.items())}
    return mask_metrics

  def add_eval_dict(self, eval_dict):
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
    def update_op(image_id_batched, groundtruth_boxes_batched,
                  groundtruth_classes_batched,
                  groundtruth_instance_masks_batched,
                  groundtruth_verified_neg_classes_batched,
                  groundtruth_not_exhaustive_classes_batched,
                  num_gt_boxes_per_image,
                  detection_scores_batched, detection_classes_batched,
                  detection_masks_batched, num_det_boxes_per_image,
                  original_image_spatial_shape):
      """Update op for metrics."""

      for (image_id, groundtruth_boxes, groundtruth_classes,
           groundtruth_instance_masks, groundtruth_verified_neg_classes,
           groundtruth_not_exhaustive_classes, num_gt_box,
           detection_scores, detection_classes,
           detection_masks, num_det_box, original_image_shape) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_instance_masks_batched,
               groundtruth_verified_neg_classes_batched,
               groundtruth_not_exhaustive_classes_batched,
               num_gt_boxes_per_image,
               detection_scores_batched, detection_classes_batched,
               detection_masks_batched, num_det_boxes_per_image,
               original_image_spatial_shape):
        self.add_single_ground_truth_image_info(
            image_id, {
                input_data_fields.groundtruth_boxes:
                    groundtruth_boxes[:num_gt_box],
                input_data_fields.groundtruth_classes:
                    groundtruth_classes[:num_gt_box],
                input_data_fields.groundtruth_instance_masks:
                    groundtruth_instance_masks[
                        :num_gt_box,
                        :original_image_shape[0],
                        :original_image_shape[1]],
                input_data_fields.groundtruth_verified_neg_classes:
                    groundtruth_verified_neg_classes,
                input_data_fields.groundtruth_not_exhaustive_classes:
                    groundtruth_not_exhaustive_classes
            })
        self.add_single_detected_image_info(
            image_id, {
                'detection_scores': detection_scores[:num_det_box],
                'detection_classes': detection_classes[:num_det_box],
                'detection_masks': detection_masks[
                    :num_det_box,
                    :original_image_shape[0],
                    :original_image_shape[1]]
            })

    # Unpack items from the evaluation dictionary.
    input_data_fields = fields.InputDataFields
    detection_fields = fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    original_image_spatial_shape = eval_dict[
        input_data_fields.original_image_spatial_shape]
    groundtruth_boxes = eval_dict[input_data_fields.groundtruth_boxes]
    groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
    groundtruth_instance_masks = eval_dict[
        input_data_fields.groundtruth_instance_masks]
    groundtruth_verified_neg_classes = eval_dict[
        input_data_fields.groundtruth_verified_neg_classes]
    groundtruth_not_exhaustive_classes = eval_dict[
        input_data_fields.groundtruth_not_exhaustive_classes]

    num_gt_boxes_per_image = eval_dict.get(
        input_data_fields.num_groundtruth_boxes, None)
    detection_scores = eval_dict[detection_fields.detection_scores]
    detection_classes = eval_dict[detection_fields.detection_classes]
    detection_masks = eval_dict[detection_fields.detection_masks]
    num_det_boxes_per_image = eval_dict.get(detection_fields.num_detections,
                                            None)

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_instance_masks = tf.expand_dims(groundtruth_instance_masks, 0)
      groundtruth_verified_neg_classes = tf.expand_dims(
          groundtruth_verified_neg_classes, 0)
      groundtruth_not_exhaustive_classes = tf.expand_dims(
          groundtruth_not_exhaustive_classes, 0)
      detection_scores = tf.expand_dims(detection_scores, 0)
      detection_classes = tf.expand_dims(detection_classes, 0)
      detection_masks = tf.expand_dims(detection_masks, 0)

      if num_gt_boxes_per_image is None:
        num_gt_boxes_per_image = tf.shape(groundtruth_boxes)[1:2]
      else:
        num_gt_boxes_per_image = tf.expand_dims(num_gt_boxes_per_image, 0)

      if num_det_boxes_per_image is None:
        num_det_boxes_per_image = tf.shape(detection_scores)[1:2]
      else:
        num_det_boxes_per_image = tf.expand_dims(num_det_boxes_per_image, 0)
    else:
      if num_gt_boxes_per_image is None:
        num_gt_boxes_per_image = tf.tile(
            tf.shape(groundtruth_boxes)[1:2],
            multiples=tf.shape(groundtruth_boxes)[0:1])
      if num_det_boxes_per_image is None:
        num_det_boxes_per_image = tf.tile(
            tf.shape(detection_scores)[1:2],
            multiples=tf.shape(detection_scores)[0:1])

    return tf.py_func(update_op, [
        image_id, groundtruth_boxes, groundtruth_classes,
        groundtruth_instance_masks, groundtruth_verified_neg_classes,
        groundtruth_not_exhaustive_classes,
        num_gt_boxes_per_image, detection_scores, detection_classes,
        detection_masks, num_det_boxes_per_image, original_image_spatial_shape
    ], [])

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns a dictionary of eval metric ops.

    Note that once value_op is called, the detections and groundtruth added via
    update_op are cleared.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating object detection
        performance. For single-image evaluation, this dictionary may be
        produced from eval_util.result_dict_for_single_example(). If multi-image
        evaluation, `eval_dict` should contain the fields
        'num_groundtruth_boxes_per_image' and 'num_det_boxes_per_image' to
        properly unpad the tensors from the batch.

    Returns:
      a dictionary of metric names to tuple of value_op and update_op that can
      be used as eval metric ops in tf.estimator.EstimatorSpec. Note that all
      update ops  must be run together and similarly all value ops must be run
      together to guarantee correct behaviour.
    """
    update_op = self.add_eval_dict(eval_dict)
    metric_names = ['DetectionMasks_Precision/mAP',
                    'DetectionMasks_Precision/mAP@.50IOU',
                    'DetectionMasks_Precision/mAP@.75IOU',
                    'DetectionMasks_Precision/mAP (small)',
                    'DetectionMasks_Precision/mAP (medium)',
                    'DetectionMasks_Precision/mAP (large)',
                    'DetectionMasks_Recall/AR@1',
                    'DetectionMasks_Recall/AR@10',
                    'DetectionMasks_Recall/AR@100',
                    'DetectionMasks_Recall/AR@100 (small)',
                    'DetectionMasks_Recall/AR@100 (medium)',
                    'DetectionMasks_Recall/AR@100 (large)']
    if self._include_metrics_per_category:
      for category_dict in self._categories:
        metric_names.append('DetectionMasks_PerformanceByCategory/mAP/' +
                            category_dict['name'])

    def first_value_func():
      self._metrics = self.evaluate()
      self.clear()
      return np.float32(self._metrics[metric_names[0]])

    def value_func_factory(metric_name):
      def value_func():
        return np.float32(self._metrics[metric_name])
      return value_func

    # Ensure that the metrics are only evaluated once.
    first_value_op = tf.py_func(first_value_func, [], tf.float32)
    eval_metric_ops = {metric_names[0]: (first_value_op, update_op)}
    with tf.control_dependencies([first_value_op]):
      for metric_name in metric_names[1:]:
        eval_metric_ops[metric_name] = (tf.py_func(
            value_func_factory(metric_name), [], np.float32), update_op)
    return eval_metric_ops

  def dump_detections_to_json_file(self, json_output_path):
    """Saves the detections into json_output_path in the format used by MS COCO.

    Args:
      json_output_path: String containing the output file's path. It can be also
        None. In that case nothing will be written to the output file.
    """
    if json_output_path and json_output_path is not None:
      pattern = re.compile(r'\d+\.\d{8,}')
      def mround(match):
        return '{:.2f}'.format(float(match.group()))

      with tf.io.gfile.GFile(json_output_path, 'w') as fid:
        json_string = json.dumps(self._detection_masks_list)
        fid.write(re.sub(pattern, mround, json_string))

      tf.logging.info('Dumping detections to output json file: %s',
                      json_output_path)
