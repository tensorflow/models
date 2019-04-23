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
"""Class for evaluating object detections with COCO metrics."""
import numpy as np
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.metrics import coco_tools
from object_detection.utils import json_utils
from object_detection.utils import object_detection_evaluation


class CocoDetectionEvaluator(object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate COCO detection metrics."""

  def __init__(self,
               categories,
               include_metrics_per_category=False,
               all_metrics_per_category=False):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      include_metrics_per_category: If True, include metrics for each category.
      all_metrics_per_category: Whether to include all the summary metrics for
        each category in per_category_ap. Be careful with setting it to true if
        you have more than handful of categories, because it will pollute
        your mldash.
    """
    super(CocoDetectionEvaluator, self).__init__(categories)
    # _image_ids is a dictionary that maps unique image ids to Booleans which
    # indicate whether a corresponding detection has been added.
    self._image_ids = {}
    self._groundtruth_list = []
    self._detection_boxes_list = []
    self._category_id_set = set([cat['id'] for cat in self._categories])
    self._annotation_id = 1
    self._metrics = None
    self._include_metrics_per_category = include_metrics_per_category
    self._all_metrics_per_category = all_metrics_per_category

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._image_ids.clear()
    self._groundtruth_list = []
    self._detection_boxes_list = []

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
        InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
          shape [num_boxes] containing iscrowd flag for groundtruth boxes.
    """
    if image_id in self._image_ids:
      tf.logging.warning('Ignoring ground truth with image id %s since it was '
                         'previously added', image_id)
      return

    groundtruth_is_crowd = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_is_crowd)
    # Drop groundtruth_is_crowd if empty tensor.
    if groundtruth_is_crowd is not None and not groundtruth_is_crowd.shape[0]:
      groundtruth_is_crowd = None

    self._groundtruth_list.extend(
        coco_tools.ExportSingleImageGroundtruthToCoco(
            image_id=image_id,
            next_annotation_id=self._annotation_id,
            category_id_set=self._category_id_set,
            groundtruth_boxes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_boxes],
            groundtruth_classes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_classes],
            groundtruth_is_crowd=groundtruth_is_crowd))
    self._annotation_id += groundtruth_dict[standard_fields.InputDataFields.
                                            groundtruth_boxes].shape[0]
    # Boolean to indicate whether a detection has been added for this image.
    self._image_ids[image_id] = False

  def add_single_detected_image_info(self,
                                     image_id,
                                     detections_dict):
    """Adds detections for a single image to be used for evaluation.

    If a detection has already been added for this image id, a warning is
    logged, and the detection is skipped.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary containing -
        DetectionResultFields.detection_boxes: float32 numpy array of shape
          [num_boxes, 4] containing `num_boxes` detection boxes of the format
          [ymin, xmin, ymax, xmax] in absolute image coordinates.
        DetectionResultFields.detection_scores: float32 numpy array of shape
          [num_boxes] containing detection scores for the boxes.
        DetectionResultFields.detection_classes: integer numpy array of shape
          [num_boxes] containing 1-indexed detection classes for the boxes.

    Raises:
      ValueError: If groundtruth for the image_id is not available.
    """
    if image_id not in self._image_ids:
      raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

    if self._image_ids[image_id]:
      tf.logging.warning('Ignoring detection with image id %s since it was '
                         'previously added', image_id)
      return

    self._detection_boxes_list.extend(
        coco_tools.ExportSingleImageDetectionBoxesToCoco(
            image_id=image_id,
            category_id_set=self._category_id_set,
            detection_boxes=detections_dict[standard_fields.
                                            DetectionResultFields
                                            .detection_boxes],
            detection_scores=detections_dict[standard_fields.
                                             DetectionResultFields.
                                             detection_scores],
            detection_classes=detections_dict[standard_fields.
                                              DetectionResultFields.
                                              detection_classes]))
    self._image_ids[image_id] = True

  def dump_detections_to_json_file(self, json_output_path):
    """Saves the detections into json_output_path in the format used by MS COCO.

    Args:
      json_output_path: String containing the output file's path. It can be also
        None. In that case nothing will be written to the output file.
    """
    if json_output_path and json_output_path is not None:
      with tf.gfile.GFile(json_output_path, 'w') as fid:
        tf.logging.info('Dumping detections to output json file.')
        json_utils.Dump(
            obj=self._detection_boxes_list, fid=fid, float_digits=4, indent=2)

  def evaluate(self):
    """Evaluates the detection boxes and returns a dictionary of coco metrics.

    Returns:
      A dictionary holding -

      1. summary_metrics:
      'DetectionBoxes_Precision/mAP': mean average precision over classes
        averaged over IOU thresholds ranging from .5 to .95 with .05
        increments.
      'DetectionBoxes_Precision/mAP@.50IOU': mean average precision at 50% IOU
      'DetectionBoxes_Precision/mAP@.75IOU': mean average precision at 75% IOU
      'DetectionBoxes_Precision/mAP (small)': mean average precision for small
        objects (area < 32^2 pixels).
      'DetectionBoxes_Precision/mAP (medium)': mean average precision for
        medium sized objects (32^2 pixels < area < 96^2 pixels).
      'DetectionBoxes_Precision/mAP (large)': mean average precision for large
        objects (96^2 pixels < area < 10000^2 pixels).
      'DetectionBoxes_Recall/AR@1': average recall with 1 detection.
      'DetectionBoxes_Recall/AR@10': average recall with 10 detections.
      'DetectionBoxes_Recall/AR@100': average recall with 100 detections.
      'DetectionBoxes_Recall/AR@100 (small)': average recall for small objects
        with 100.
      'DetectionBoxes_Recall/AR@100 (medium)': average recall for medium objects
        with 100.
      'DetectionBoxes_Recall/AR@100 (large)': average recall for large objects
        with 100 detections.

      2. per_category_ap: if include_metrics_per_category is True, category
      specific results with keys of the form:
      'Precision mAP ByCategory/category' (without the supercategory part if
      no supercategories exist). For backward compatibility
      'PerformanceByCategory' is included in the output regardless of
      all_metrics_per_category.
    """
    groundtruth_dict = {
        'annotations': self._groundtruth_list,
        'images': [{'id': image_id} for image_id in self._image_ids],
        'categories': self._categories
    }
    coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
        self._detection_boxes_list)
    box_evaluator = coco_tools.COCOEvalWrapper(
        coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
    box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
        include_metrics_per_category=self._include_metrics_per_category,
        all_metrics_per_category=self._all_metrics_per_category)
    box_metrics.update(box_per_category_ap)
    box_metrics = {'DetectionBoxes_'+ key: value
                   for key, value in iter(box_metrics.items())}
    return box_metrics

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns a dictionary of eval metric ops.

    Note that once value_op is called, the detections and groundtruth added via
    update_op are cleared.

    This function can take in groundtruth and detections for a batch of images,
    or for a single image. For the latter case, the batch dimension for input
    tensors need not be present.

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
      update ops must be run together and similarly all value ops must be run
      together to guarantee correct behaviour.
    """
    def update_op(
        image_id_batched,
        groundtruth_boxes_batched,
        groundtruth_classes_batched,
        groundtruth_is_crowd_batched,
        num_gt_boxes_per_image,
        detection_boxes_batched,
        detection_scores_batched,
        detection_classes_batched,
        num_det_boxes_per_image,
        is_annotated_batched):
      """Update operation for adding batch of images to Coco evaluator."""

      for (image_id, gt_box, gt_class, gt_is_crowd, num_gt_box, det_box,
           det_score, det_class, num_det_box, is_annotated) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_is_crowd_batched,
               num_gt_boxes_per_image,
               detection_boxes_batched, detection_scores_batched,
               detection_classes_batched, num_det_boxes_per_image,
               is_annotated_batched):
        if is_annotated:
          self.add_single_ground_truth_image_info(
              image_id, {
                  'groundtruth_boxes': gt_box[:num_gt_box],
                  'groundtruth_classes': gt_class[:num_gt_box],
                  'groundtruth_is_crowd': gt_is_crowd[:num_gt_box]
              })
          self.add_single_detected_image_info(
              image_id,
              {'detection_boxes': det_box[:num_det_box],
               'detection_scores': det_score[:num_det_box],
               'detection_classes': det_class[:num_det_box]})

    # Unpack items from the evaluation dictionary.
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    groundtruth_boxes = eval_dict[input_data_fields.groundtruth_boxes]
    groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
    groundtruth_is_crowd = eval_dict.get(
        input_data_fields.groundtruth_is_crowd, None)
    detection_boxes = eval_dict[detection_fields.detection_boxes]
    detection_scores = eval_dict[detection_fields.detection_scores]
    detection_classes = eval_dict[detection_fields.detection_classes]
    num_gt_boxes_per_image = eval_dict.get(
        'num_groundtruth_boxes_per_image', None)
    num_det_boxes_per_image = eval_dict.get('num_det_boxes_per_image', None)
    is_annotated = eval_dict.get('is_annotated', None)

    if groundtruth_is_crowd is None:
      groundtruth_is_crowd = tf.zeros_like(groundtruth_classes, dtype=tf.bool)
    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_is_crowd = tf.expand_dims(groundtruth_is_crowd, 0)
      detection_boxes = tf.expand_dims(detection_boxes, 0)
      detection_scores = tf.expand_dims(detection_scores, 0)
      detection_classes = tf.expand_dims(detection_classes, 0)

      if num_gt_boxes_per_image is None:
        num_gt_boxes_per_image = tf.shape(groundtruth_boxes)[1:2]
      else:
        num_gt_boxes_per_image = tf.expand_dims(num_gt_boxes_per_image, 0)

      if num_det_boxes_per_image is None:
        num_det_boxes_per_image = tf.shape(detection_boxes)[1:2]
      else:
        num_det_boxes_per_image = tf.expand_dims(num_det_boxes_per_image, 0)

      if is_annotated is None:
        is_annotated = tf.constant([True])
      else:
        is_annotated = tf.expand_dims(is_annotated, 0)
    else:
      if num_gt_boxes_per_image is None:
        num_gt_boxes_per_image = tf.tile(
            tf.shape(groundtruth_boxes)[1:2],
            multiples=tf.shape(groundtruth_boxes)[0:1])
      if num_det_boxes_per_image is None:
        num_det_boxes_per_image = tf.tile(
            tf.shape(detection_boxes)[1:2],
            multiples=tf.shape(detection_boxes)[0:1])
      if is_annotated is None:
        is_annotated = tf.ones_like(image_id, dtype=tf.bool)

    update_op = tf.py_func(update_op, [image_id,
                                       groundtruth_boxes,
                                       groundtruth_classes,
                                       groundtruth_is_crowd,
                                       num_gt_boxes_per_image,
                                       detection_boxes,
                                       detection_scores,
                                       detection_classes,
                                       num_det_boxes_per_image,
                                       is_annotated], [])
    metric_names = ['DetectionBoxes_Precision/mAP',
                    'DetectionBoxes_Precision/mAP@.50IOU',
                    'DetectionBoxes_Precision/mAP@.75IOU',
                    'DetectionBoxes_Precision/mAP (large)',
                    'DetectionBoxes_Precision/mAP (medium)',
                    'DetectionBoxes_Precision/mAP (small)',
                    'DetectionBoxes_Recall/AR@1',
                    'DetectionBoxes_Recall/AR@10',
                    'DetectionBoxes_Recall/AR@100',
                    'DetectionBoxes_Recall/AR@100 (large)',
                    'DetectionBoxes_Recall/AR@100 (medium)',
                    'DetectionBoxes_Recall/AR@100 (small)']
    if self._include_metrics_per_category:
      for category_dict in self._categories:
        metric_names.append('DetectionBoxes_PerformanceByCategory/mAP/' +
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


def _check_mask_type_and_value(array_name, masks):
  """Checks whether mask dtype is uint8 and the values are either 0 or 1."""
  if masks.dtype != np.uint8:
    raise ValueError('{} must be of type np.uint8. Found {}.'.format(
        array_name, masks.dtype))
  if np.any(np.logical_and(masks != 0, masks != 1)):
    raise ValueError('{} elements can only be either 0 or 1.'.format(
        array_name))


class CocoMaskEvaluator(object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate COCO detection metrics."""

  def __init__(self, categories, include_metrics_per_category=False):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      include_metrics_per_category: If True, include metrics for each category.
    """
    super(CocoMaskEvaluator, self).__init__(categories)
    self._image_id_to_mask_shape_map = {}
    self._image_ids_with_detections = set([])
    self._groundtruth_list = []
    self._detection_masks_list = []
    self._category_id_set = set([cat['id'] for cat in self._categories])
    self._annotation_id = 1
    self._include_metrics_per_category = include_metrics_per_category

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._image_id_to_mask_shape_map.clear()
    self._image_ids_with_detections.clear()
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
          [num_boxes, image_height, image_width] containing groundtruth masks
          corresponding to the boxes. The elements of the array must be in
          {0, 1}.
    """
    if image_id in self._image_id_to_mask_shape_map:
      tf.logging.warning('Ignoring ground truth with image id %s since it was '
                         'previously added', image_id)
      return

    groundtruth_instance_masks = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_instance_masks]
    _check_mask_type_and_value(standard_fields.InputDataFields.
                               groundtruth_instance_masks,
                               groundtruth_instance_masks)
    self._groundtruth_list.extend(
        coco_tools.
        ExportSingleImageGroundtruthToCoco(
            image_id=image_id,
            next_annotation_id=self._annotation_id,
            category_id_set=self._category_id_set,
            groundtruth_boxes=groundtruth_dict[standard_fields.InputDataFields.
                                               groundtruth_boxes],
            groundtruth_classes=groundtruth_dict[standard_fields.
                                                 InputDataFields.
                                                 groundtruth_classes],
            groundtruth_masks=groundtruth_instance_masks))
    self._annotation_id += groundtruth_dict[standard_fields.InputDataFields.
                                            groundtruth_boxes].shape[0]
    self._image_id_to_mask_shape_map[image_id] = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_instance_masks].shape

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
      ValueError: If groundtruth for the image_id is not available or if
        spatial shapes of groundtruth_instance_masks and detection_masks are
        incompatible.
    """
    if image_id not in self._image_id_to_mask_shape_map:
      raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

    if image_id in self._image_ids_with_detections:
      tf.logging.warning('Ignoring detection with image id %s since it was '
                         'previously added', image_id)
      return

    groundtruth_masks_shape = self._image_id_to_mask_shape_map[image_id]
    detection_masks = detections_dict[standard_fields.DetectionResultFields.
                                      detection_masks]
    if groundtruth_masks_shape[1:] != detection_masks.shape[1:]:
      raise ValueError('Spatial shape of groundtruth masks and detection masks '
                       'are incompatible: {} vs {}'.format(
                           groundtruth_masks_shape,
                           detection_masks.shape))
    _check_mask_type_and_value(standard_fields.DetectionResultFields.
                               detection_masks,
                               detection_masks)
    self._detection_masks_list.extend(
        coco_tools.ExportSingleImageDetectionMasksToCoco(
            image_id=image_id,
            category_id_set=self._category_id_set,
            detection_masks=detection_masks,
            detection_scores=detections_dict[standard_fields.
                                             DetectionResultFields.
                                             detection_scores],
            detection_classes=detections_dict[standard_fields.
                                              DetectionResultFields.
                                              detection_classes]))
    self._image_ids_with_detections.update([image_id])

  def dump_detections_to_json_file(self, json_output_path):
    """Saves the detections into json_output_path in the format used by MS COCO.

    Args:
      json_output_path: String containing the output file's path. It can be also
        None. In that case nothing will be written to the output file.
    """
    if json_output_path and json_output_path is not None:
      tf.logging.info('Dumping detections to output json file.')
      with tf.gfile.GFile(json_output_path, 'w') as fid:
        json_utils.Dump(
            obj=self._detection_masks_list, fid=fid, float_digits=4, indent=2)

  def evaluate(self):
    """Evaluates the detection masks and returns a dictionary of coco metrics.

    Returns:
      A dictionary holding -

      1. summary_metrics:
      'DetectionMasks_Precision/mAP': mean average precision over classes
        averaged over IOU thresholds ranging from .5 to .95 with .05 increments.
      'DetectionMasks_Precision/mAP@.50IOU': mean average precision at 50% IOU.
      'DetectionMasks_Precision/mAP@.75IOU': mean average precision at 75% IOU.
      'DetectionMasks_Precision/mAP (small)': mean average precision for small
        objects (area < 32^2 pixels).
      'DetectionMasks_Precision/mAP (medium)': mean average precision for medium
        sized objects (32^2 pixels < area < 96^2 pixels).
      'DetectionMasks_Precision/mAP (large)': mean average precision for large
        objects (96^2 pixels < area < 10000^2 pixels).
      'DetectionMasks_Recall/AR@1': average recall with 1 detection.
      'DetectionMasks_Recall/AR@10': average recall with 10 detections.
      'DetectionMasks_Recall/AR@100': average recall with 100 detections.
      'DetectionMasks_Recall/AR@100 (small)': average recall for small objects
        with 100 detections.
      'DetectionMasks_Recall/AR@100 (medium)': average recall for medium objects
        with 100 detections.
      'DetectionMasks_Recall/AR@100 (large)': average recall for large objects
        with 100 detections.

      2. per_category_ap: if include_metrics_per_category is True, category
      specific results with keys of the form:
      'Precision mAP ByCategory/category' (without the supercategory part if
      no supercategories exist). For backward compatibility
      'PerformanceByCategory' is included in the output regardless of
      all_metrics_per_category.
    """
    groundtruth_dict = {
        'annotations': self._groundtruth_list,
        'images': [{'id': image_id, 'height': shape[1], 'width': shape[2]}
                   for image_id, shape in self._image_id_to_mask_shape_map.
                   items()],
        'categories': self._categories
    }
    coco_wrapped_groundtruth = coco_tools.COCOWrapper(
        groundtruth_dict, detection_type='segmentation')
    coco_wrapped_detection_masks = coco_wrapped_groundtruth.LoadAnnotations(
        self._detection_masks_list)
    mask_evaluator = coco_tools.COCOEvalWrapper(
        coco_wrapped_groundtruth, coco_wrapped_detection_masks,
        agnostic_mode=False, iou_type='segm')
    mask_metrics, mask_per_category_ap = mask_evaluator.ComputeMetrics(
        include_metrics_per_category=self._include_metrics_per_category)
    mask_metrics.update(mask_per_category_ap)
    mask_metrics = {'DetectionMasks_'+ key: value
                    for key, value in mask_metrics.items()}
    return mask_metrics

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

    def update_op(image_id_batched, groundtruth_boxes_batched,
                  groundtruth_classes_batched,
                  groundtruth_instance_masks_batched,
                  groundtruth_is_crowd_batched, num_gt_boxes_per_image,
                  detection_scores_batched, detection_classes_batched,
                  detection_masks_batched, num_det_boxes_per_image):
      """Update op for metrics."""

      for (image_id, groundtruth_boxes, groundtruth_classes,
           groundtruth_instance_masks, groundtruth_is_crowd, num_gt_box,
           detection_scores, detection_classes,
           detection_masks, num_det_box) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_instance_masks_batched,
               groundtruth_is_crowd_batched, num_gt_boxes_per_image,
               detection_scores_batched, detection_classes_batched,
               detection_masks_batched, num_det_boxes_per_image):
        self.add_single_ground_truth_image_info(
            image_id, {
                'groundtruth_boxes':
                    groundtruth_boxes[:num_gt_box],
                'groundtruth_classes':
                    groundtruth_classes[:num_gt_box],
                'groundtruth_instance_masks':
                    groundtruth_instance_masks[:num_gt_box],
                'groundtruth_is_crowd':
                    groundtruth_is_crowd[:num_gt_box]
            })
        self.add_single_detected_image_info(
            image_id, {
                'detection_scores': detection_scores[:num_det_box],
                'detection_classes': detection_classes[:num_det_box],
                'detection_masks': detection_masks[:num_det_box]
            })

    # Unpack items from the evaluation dictionary.
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    groundtruth_boxes = eval_dict[input_data_fields.groundtruth_boxes]
    groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
    groundtruth_instance_masks = eval_dict[
        input_data_fields.groundtruth_instance_masks]
    groundtruth_is_crowd = eval_dict.get(
        input_data_fields.groundtruth_is_crowd, None)
    num_gt_boxes_per_image = eval_dict.get(
        input_data_fields.num_groundtruth_boxes, None)
    detection_scores = eval_dict[detection_fields.detection_scores]
    detection_classes = eval_dict[detection_fields.detection_classes]
    detection_masks = eval_dict[detection_fields.detection_masks]
    num_det_boxes_per_image = eval_dict.get(detection_fields.num_detections,
                                            None)

    if groundtruth_is_crowd is None:
      groundtruth_is_crowd = tf.zeros_like(groundtruth_classes, dtype=tf.bool)

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_instance_masks = tf.expand_dims(groundtruth_instance_masks, 0)
      groundtruth_is_crowd = tf.expand_dims(groundtruth_is_crowd, 0)
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

    update_op = tf.py_func(update_op, [
        image_id, groundtruth_boxes, groundtruth_classes,
        groundtruth_instance_masks, groundtruth_is_crowd,
        num_gt_boxes_per_image, detection_scores, detection_classes,
        detection_masks, num_det_boxes_per_image
    ], [])

    metric_names = ['DetectionMasks_Precision/mAP',
                    'DetectionMasks_Precision/mAP@.50IOU',
                    'DetectionMasks_Precision/mAP@.75IOU',
                    'DetectionMasks_Precision/mAP (large)',
                    'DetectionMasks_Precision/mAP (medium)',
                    'DetectionMasks_Precision/mAP (small)',
                    'DetectionMasks_Recall/AR@1',
                    'DetectionMasks_Recall/AR@10',
                    'DetectionMasks_Recall/AR@100',
                    'DetectionMasks_Recall/AR@100 (large)',
                    'DetectionMasks_Recall/AR@100 (medium)',
                    'DetectionMasks_Recall/AR@100 (small)']
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
