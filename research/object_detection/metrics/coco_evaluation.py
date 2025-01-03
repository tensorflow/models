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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import standard_fields
from object_detection.metrics import coco_tools
from object_detection.utils import json_utils
from object_detection.utils import np_mask_ops
from object_detection.utils import object_detection_evaluation


class CocoDetectionEvaluator(object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate COCO detection metrics."""

  def __init__(self,
               categories,
               include_metrics_per_category=False,
               all_metrics_per_category=False,
               skip_predictions_for_unlabeled_class=False,
               super_categories=None):
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
      skip_predictions_for_unlabeled_class: Skip predictions that do not match
        with the labeled classes for the image.
      super_categories: None or a python dict mapping super-category names
        (strings) to lists of categories (corresponding to category names
        in the label_map).  Metrics are aggregated along these super-categories
        and added to the `per_category_ap` and are associated with the name
          `PerformanceBySuperCategory/<super-category-name>`.
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
    self._skip_predictions_for_unlabeled_class = skip_predictions_for_unlabeled_class
    self._groundtruth_labeled_classes = {}
    self._super_categories = super_categories

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
        InputDataFields.groundtruth_area (optional): float numpy array of
          shape [num_boxes] containing the area (in the original absolute
          coordinates) of the annotated object.
        InputDataFields.groundtruth_keypoints (optional): float numpy array of
          keypoints with shape [num_boxes, num_keypoints, 2].
        InputDataFields.groundtruth_keypoint_visibilities (optional): integer
          numpy array of keypoint visibilities with shape [num_gt_boxes,
          num_keypoints]. Integer is treated as an enum with 0=not labeled,
          1=labeled but not visible and 2=labeled and visible.
        InputDataFields.groundtruth_labeled_classes (optional): a tensor of
          shape [num_classes + 1] containing the multi-hot tensor indicating the
          classes that each image is labeled for. Note that the classes labels
          are 1-indexed.
    """
    if image_id in self._image_ids:
      tf.logging.warning('Ignoring ground truth with image id %s since it was '
                         'previously added', image_id)
      return

    # Drop optional fields if empty tensor.
    groundtruth_is_crowd = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_is_crowd)
    groundtruth_area = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_area)
    groundtruth_keypoints = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_keypoints)
    groundtruth_keypoint_visibilities = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_keypoint_visibilities)
    if groundtruth_is_crowd is not None and not groundtruth_is_crowd.shape[0]:
      groundtruth_is_crowd = None
    if groundtruth_area is not None and not groundtruth_area.shape[0]:
      groundtruth_area = None
    if groundtruth_keypoints is not None and not groundtruth_keypoints.shape[0]:
      groundtruth_keypoints = None
    if groundtruth_keypoint_visibilities is not None and not groundtruth_keypoint_visibilities.shape[
        0]:
      groundtruth_keypoint_visibilities = None

    self._groundtruth_list.extend(
        coco_tools.ExportSingleImageGroundtruthToCoco(
            image_id=image_id,
            next_annotation_id=self._annotation_id,
            category_id_set=self._category_id_set,
            groundtruth_boxes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_boxes],
            groundtruth_classes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_classes],
            groundtruth_is_crowd=groundtruth_is_crowd,
            groundtruth_area=groundtruth_area,
            groundtruth_keypoints=groundtruth_keypoints,
            groundtruth_keypoint_visibilities=groundtruth_keypoint_visibilities)
    )

    self._annotation_id += groundtruth_dict[standard_fields.InputDataFields.
                                            groundtruth_boxes].shape[0]
    if (standard_fields.InputDataFields.groundtruth_labeled_classes
       ) in groundtruth_dict:
      labeled_classes = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_labeled_classes]
      if labeled_classes.shape != (len(self._category_id_set) + 1,):
        raise ValueError('Invalid shape for groundtruth labeled classes: {}, '
                         'num_categories_including_background: {}'.format(
                             labeled_classes,
                             len(self._category_id_set) + 1))
      self._groundtruth_labeled_classes[image_id] = np.flatnonzero(
          groundtruth_dict[standard_fields.InputDataFields
                           .groundtruth_labeled_classes] == 1).tolist()

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
        DetectionResultFields.detection_keypoints (optional): float numpy array
          of keypoints with shape [num_boxes, num_keypoints, 2].
    Raises:
      ValueError: If groundtruth for the image_id is not available.
    """
    if image_id not in self._image_ids:
      raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

    if self._image_ids[image_id]:
      tf.logging.warning('Ignoring detection with image id %s since it was '
                         'previously added', image_id)
      return

    # Drop optional fields if empty tensor.
    detection_keypoints = detections_dict.get(
        standard_fields.DetectionResultFields.detection_keypoints)
    if detection_keypoints is not None and not detection_keypoints.shape[0]:
      detection_keypoints = None

    if self._skip_predictions_for_unlabeled_class:
      det_classes = detections_dict[
          standard_fields.DetectionResultFields.detection_classes]
      num_det_boxes = det_classes.shape[0]
      keep_box_ids = []
      for box_id in range(num_det_boxes):
        if det_classes[box_id] in self._groundtruth_labeled_classes[image_id]:
          keep_box_ids.append(box_id)
      self._detection_boxes_list.extend(
          coco_tools.ExportSingleImageDetectionBoxesToCoco(
              image_id=image_id,
              category_id_set=self._category_id_set,
              detection_boxes=detections_dict[
                  standard_fields.DetectionResultFields.detection_boxes]
              [keep_box_ids],
              detection_scores=detections_dict[
                  standard_fields.DetectionResultFields.detection_scores]
              [keep_box_ids],
              detection_classes=detections_dict[
                  standard_fields.DetectionResultFields.detection_classes]
              [keep_box_ids],
              detection_keypoints=detection_keypoints))
    else:
      self._detection_boxes_list.extend(
          coco_tools.ExportSingleImageDetectionBoxesToCoco(
              image_id=image_id,
              category_id_set=self._category_id_set,
              detection_boxes=detections_dict[
                  standard_fields.DetectionResultFields.detection_boxes],
              detection_scores=detections_dict[
                  standard_fields.DetectionResultFields.detection_scores],
              detection_classes=detections_dict[
                  standard_fields.DetectionResultFields.detection_classes],
              detection_keypoints=detection_keypoints))
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
        If super_categories are provided, then this will additionally include
      metrics aggregated along the super_categories with keys of the form:
      `PerformanceBySuperCategory/<super-category-name>`
    """
    tf.logging.info('Performing evaluation on %d images.', len(self._image_ids))
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
        all_metrics_per_category=self._all_metrics_per_category,
        super_categories=self._super_categories)
    box_metrics.update(box_per_category_ap)
    box_metrics = {'DetectionBoxes_'+ key: value
                   for key, value in iter(box_metrics.items())}
    return box_metrics

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
                  groundtruth_classes_batched, groundtruth_is_crowd_batched,
                  groundtruth_labeled_classes_batched, num_gt_boxes_per_image,
                  detection_boxes_batched, detection_scores_batched,
                  detection_classes_batched, num_det_boxes_per_image,
                  is_annotated_batched):
      """Update operation for adding batch of images to Coco evaluator."""
      for (image_id, gt_box, gt_class, gt_is_crowd, gt_labeled_classes,
           num_gt_box, det_box, det_score, det_class,
           num_det_box, is_annotated) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_is_crowd_batched,
               groundtruth_labeled_classes_batched, num_gt_boxes_per_image,
               detection_boxes_batched, detection_scores_batched,
               detection_classes_batched, num_det_boxes_per_image,
               is_annotated_batched):
        if is_annotated:
          self.add_single_ground_truth_image_info(
              image_id, {
                  'groundtruth_boxes': gt_box[:num_gt_box],
                  'groundtruth_classes': gt_class[:num_gt_box],
                  'groundtruth_is_crowd': gt_is_crowd[:num_gt_box],
                  'groundtruth_labeled_classes': gt_labeled_classes
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
    groundtruth_labeled_classes = eval_dict.get(
        input_data_fields.groundtruth_labeled_classes, None)
    detection_boxes = eval_dict[detection_fields.detection_boxes]
    detection_scores = eval_dict[detection_fields.detection_scores]
    detection_classes = eval_dict[detection_fields.detection_classes]
    num_gt_boxes_per_image = eval_dict.get(
        input_data_fields.num_groundtruth_boxes, None)
    num_det_boxes_per_image = eval_dict.get(detection_fields.num_detections,
                                            None)
    is_annotated = eval_dict.get('is_annotated', None)

    if groundtruth_is_crowd is None:
      groundtruth_is_crowd = tf.zeros_like(groundtruth_classes, dtype=tf.bool)

    # If groundtruth_labeled_classes is not provided, make it equal to the
    # detection_classes. This assumes that all predictions will be kept to
    # compute eval metrics.
    if groundtruth_labeled_classes is None:
      groundtruth_labeled_classes = tf.reduce_max(
          tf.one_hot(
              tf.cast(detection_classes, tf.int32),
              len(self._category_id_set) + 1),
          axis=-2)

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_is_crowd = tf.expand_dims(groundtruth_is_crowd, 0)
      groundtruth_labeled_classes = tf.expand_dims(groundtruth_labeled_classes,
                                                   0)
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

    return tf.py_func(update_op, [
        image_id, groundtruth_boxes, groundtruth_classes, groundtruth_is_crowd,
        groundtruth_labeled_classes, num_gt_boxes_per_image, detection_boxes,
        detection_scores, detection_classes, num_det_boxes_per_image,
        is_annotated
    ], [])

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
    update_op = self.add_eval_dict(eval_dict)
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


def convert_masks_to_binary(masks):
  """Converts masks to 0 or 1 and uint8 type."""
  return (masks > 0).astype(np.uint8)


class CocoKeypointEvaluator(CocoDetectionEvaluator):
  """Class to evaluate COCO keypoint metrics."""

  def __init__(self,
               category_id,
               category_keypoints,
               class_text,
               oks_sigmas=None):
    """Constructor.

    Args:
      category_id: An integer id uniquely identifying this category.
      category_keypoints: A list specifying keypoint mappings, with items:
          'id': (required) an integer id identifying the keypoint.
          'name': (required) a string representing the keypoint name.
      class_text: A string representing the category name for which keypoint
        metrics are to be computed.
      oks_sigmas: A dict of keypoint name to standard deviation values for OKS
        metrics. If not provided, default value of 0.05 will be used.
    """
    self._category_id = category_id
    self._category_name = class_text
    self._keypoint_ids = sorted(
        [keypoint['id'] for keypoint in category_keypoints])
    kpt_id_to_name = {kpt['id']: kpt['name'] for kpt in category_keypoints}
    if oks_sigmas:
      self._oks_sigmas = np.array([
          oks_sigmas[kpt_id_to_name[idx]] for idx in self._keypoint_ids
      ])
    else:
      # Default all per-keypoint sigmas to 0.
      self._oks_sigmas = np.full((len(self._keypoint_ids)), 0.05)
      tf.logging.warning('No default keypoint OKS sigmas provided. Will use '
                         '0.05')
    tf.logging.info('Using the following keypoint OKS sigmas: {}'.format(
        self._oks_sigmas))
    self._metrics = None
    super(CocoKeypointEvaluator, self).__init__([{
        'id': self._category_id,
        'name': class_text
    }])

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image with keypoints.

    If the image has already been added, a warning is logged, and groundtruth
    is ignored.

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
        InputDataFields.groundtruth_area (optional): float numpy array of
          shape [num_boxes] containing the area (in the original absolute
          coordinates) of the annotated object.
        InputDataFields.groundtruth_keypoints: float numpy array of
          keypoints with shape [num_boxes, num_keypoints, 2].
        InputDataFields.groundtruth_keypoint_visibilities (optional): integer
          numpy array of keypoint visibilities with shape [num_gt_boxes,
          num_keypoints]. Integer is treated as an enum with 0=not labels,
          1=labeled but not visible and 2=labeled and visible.
    """

    # Keep only the groundtruth for our category and its keypoints.
    groundtruth_classes = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_classes]
    groundtruth_boxes = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_boxes]
    groundtruth_keypoints = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_keypoints]
    class_indices = [
        idx for idx, gt_class_id in enumerate(groundtruth_classes)
        if gt_class_id == self._category_id
    ]
    filtered_groundtruth_classes = np.take(
        groundtruth_classes, class_indices, axis=0)
    filtered_groundtruth_boxes = np.take(
        groundtruth_boxes, class_indices, axis=0)
    filtered_groundtruth_keypoints = np.take(
        groundtruth_keypoints, class_indices, axis=0)
    filtered_groundtruth_keypoints = np.take(
        filtered_groundtruth_keypoints, self._keypoint_ids, axis=1)

    filtered_groundtruth_dict = {}
    filtered_groundtruth_dict[
        standard_fields.InputDataFields
        .groundtruth_classes] = filtered_groundtruth_classes
    filtered_groundtruth_dict[standard_fields.InputDataFields
                              .groundtruth_boxes] = filtered_groundtruth_boxes
    filtered_groundtruth_dict[
        standard_fields.InputDataFields
        .groundtruth_keypoints] = filtered_groundtruth_keypoints

    if (standard_fields.InputDataFields.groundtruth_is_crowd in
        groundtruth_dict.keys()):
      groundtruth_is_crowd = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_is_crowd]
      filtered_groundtruth_is_crowd = np.take(groundtruth_is_crowd,
                                              class_indices, 0)
      filtered_groundtruth_dict[
          standard_fields.InputDataFields
          .groundtruth_is_crowd] = filtered_groundtruth_is_crowd
    if (standard_fields.InputDataFields.groundtruth_area in
        groundtruth_dict.keys()):
      groundtruth_area = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_area]
      filtered_groundtruth_area = np.take(groundtruth_area, class_indices, 0)
      filtered_groundtruth_dict[
          standard_fields.InputDataFields
          .groundtruth_area] = filtered_groundtruth_area
    if (standard_fields.InputDataFields.groundtruth_keypoint_visibilities in
        groundtruth_dict.keys()):
      groundtruth_keypoint_visibilities = groundtruth_dict[
          standard_fields.InputDataFields.groundtruth_keypoint_visibilities]
      filtered_groundtruth_keypoint_visibilities = np.take(
          groundtruth_keypoint_visibilities, class_indices, axis=0)
      filtered_groundtruth_keypoint_visibilities = np.take(
          filtered_groundtruth_keypoint_visibilities,
          self._keypoint_ids,
          axis=1)
      filtered_groundtruth_dict[
          standard_fields.InputDataFields.
          groundtruth_keypoint_visibilities] = filtered_groundtruth_keypoint_visibilities

    super(CocoKeypointEvaluator,
          self).add_single_ground_truth_image_info(image_id,
                                                   filtered_groundtruth_dict)

  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image and the specific category for which keypoints are evaluated.

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
        DetectionResultFields.detection_keypoints: float numpy array of
          keypoints with shape [num_boxes, num_keypoints, 2].

    Raises:
      ValueError: If groundtruth for the image_id is not available.
    """

    # Keep only the detections for our category and its keypoints.
    detection_classes = detections_dict[
        standard_fields.DetectionResultFields.detection_classes]
    detection_boxes = detections_dict[
        standard_fields.DetectionResultFields.detection_boxes]
    detection_scores = detections_dict[
        standard_fields.DetectionResultFields.detection_scores]
    detection_keypoints = detections_dict[
        standard_fields.DetectionResultFields.detection_keypoints]
    class_indices = [
        idx for idx, class_id in enumerate(detection_classes)
        if class_id == self._category_id
    ]
    filtered_detection_classes = np.take(
        detection_classes, class_indices, axis=0)
    filtered_detection_boxes = np.take(detection_boxes, class_indices, axis=0)
    filtered_detection_scores = np.take(detection_scores, class_indices, axis=0)
    filtered_detection_keypoints = np.take(
        detection_keypoints, class_indices, axis=0)
    filtered_detection_keypoints = np.take(
        filtered_detection_keypoints, self._keypoint_ids, axis=1)

    filtered_detections_dict = {}
    filtered_detections_dict[standard_fields.DetectionResultFields
                             .detection_classes] = filtered_detection_classes
    filtered_detections_dict[standard_fields.DetectionResultFields
                             .detection_boxes] = filtered_detection_boxes
    filtered_detections_dict[standard_fields.DetectionResultFields
                             .detection_scores] = filtered_detection_scores
    filtered_detections_dict[standard_fields.DetectionResultFields.
                             detection_keypoints] = filtered_detection_keypoints

    super(CocoKeypointEvaluator,
          self).add_single_detected_image_info(image_id,
                                               filtered_detections_dict)

  def evaluate(self):
    """Evaluates the keypoints and returns a dictionary of coco metrics.

    Returns:
      A dictionary holding -

      1. summary_metrics:
      'Keypoints_Precision/mAP': mean average precision over classes
        averaged over OKS thresholds ranging from .5 to .95 with .05
        increments.
      'Keypoints_Precision/mAP@.50IOU': mean average precision at 50% OKS
      'Keypoints_Precision/mAP@.75IOU': mean average precision at 75% OKS
      'Keypoints_Precision/mAP (medium)': mean average precision for medium
        sized objects (32^2 pixels < area < 96^2 pixels).
      'Keypoints_Precision/mAP (large)': mean average precision for large
        objects (96^2 pixels < area < 10000^2 pixels).
      'Keypoints_Recall/AR@1': average recall with 1 detection.
      'Keypoints_Recall/AR@10': average recall with 10 detections.
      'Keypoints_Recall/AR@100': average recall with 100 detections.
      'Keypoints_Recall/AR@100 (medium)': average recall for medium objects with
        100.
      'Keypoints_Recall/AR@100 (large)': average recall for large objects with
        100 detections.
    """
    tf.logging.info('Performing evaluation on %d images.', len(self._image_ids))
    groundtruth_dict = {
        'annotations': self._groundtruth_list,
        'images': [{'id': image_id} for image_id in self._image_ids],
        'categories': self._categories
    }
    coco_wrapped_groundtruth = coco_tools.COCOWrapper(
        groundtruth_dict, detection_type='bbox')
    coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
        self._detection_boxes_list)
    keypoint_evaluator = coco_tools.COCOEvalWrapper(
        coco_wrapped_groundtruth,
        coco_wrapped_detections,
        agnostic_mode=False,
        iou_type='keypoints',
        oks_sigmas=self._oks_sigmas)
    keypoint_metrics, _ = keypoint_evaluator.ComputeMetrics(
        include_metrics_per_category=False, all_metrics_per_category=False)
    keypoint_metrics = {
        'Keypoints_' + key: value
        for key, value in iter(keypoint_metrics.items())
    }
    return keypoint_metrics

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
    def update_op(
        image_id_batched,
        groundtruth_boxes_batched,
        groundtruth_classes_batched,
        groundtruth_is_crowd_batched,
        groundtruth_area_batched,
        groundtruth_keypoints_batched,
        groundtruth_keypoint_visibilities_batched,
        num_gt_boxes_per_image,
        detection_boxes_batched,
        detection_scores_batched,
        detection_classes_batched,
        detection_keypoints_batched,
        num_det_boxes_per_image,
        is_annotated_batched):
      """Update operation for adding batch of images to Coco evaluator."""

      for (image_id, gt_box, gt_class, gt_is_crowd, gt_area, gt_keyp,
           gt_keyp_vis, num_gt_box, det_box, det_score, det_class, det_keyp,
           num_det_box, is_annotated) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_is_crowd_batched,
               groundtruth_area_batched, groundtruth_keypoints_batched,
               groundtruth_keypoint_visibilities_batched,
               num_gt_boxes_per_image, detection_boxes_batched,
               detection_scores_batched, detection_classes_batched,
               detection_keypoints_batched, num_det_boxes_per_image,
               is_annotated_batched):
        if is_annotated:
          self.add_single_ground_truth_image_info(
              image_id, {
                  'groundtruth_boxes': gt_box[:num_gt_box],
                  'groundtruth_classes': gt_class[:num_gt_box],
                  'groundtruth_is_crowd': gt_is_crowd[:num_gt_box],
                  'groundtruth_area': gt_area[:num_gt_box],
                  'groundtruth_keypoints': gt_keyp[:num_gt_box],
                  'groundtruth_keypoint_visibilities': gt_keyp_vis[:num_gt_box]
              })
          self.add_single_detected_image_info(
              image_id, {
                  'detection_boxes': det_box[:num_det_box],
                  'detection_scores': det_score[:num_det_box],
                  'detection_classes': det_class[:num_det_box],
                  'detection_keypoints': det_keyp[:num_det_box],
              })

    # Unpack items from the evaluation dictionary.
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    groundtruth_boxes = eval_dict[input_data_fields.groundtruth_boxes]
    groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
    groundtruth_is_crowd = eval_dict.get(input_data_fields.groundtruth_is_crowd,
                                         None)
    groundtruth_area = eval_dict.get(input_data_fields.groundtruth_area, None)
    groundtruth_keypoints = eval_dict[input_data_fields.groundtruth_keypoints]
    groundtruth_keypoint_visibilities = eval_dict.get(
        input_data_fields.groundtruth_keypoint_visibilities, None)
    detection_boxes = eval_dict[detection_fields.detection_boxes]
    detection_scores = eval_dict[detection_fields.detection_scores]
    detection_classes = eval_dict[detection_fields.detection_classes]
    detection_keypoints = eval_dict[detection_fields.detection_keypoints]
    num_gt_boxes_per_image = eval_dict.get(
        'num_groundtruth_boxes_per_image', None)
    num_det_boxes_per_image = eval_dict.get('num_det_boxes_per_image', None)
    is_annotated = eval_dict.get('is_annotated', None)

    if groundtruth_is_crowd is None:
      groundtruth_is_crowd = tf.zeros_like(groundtruth_classes, dtype=tf.bool)

    if groundtruth_area is None:
      groundtruth_area = tf.zeros_like(groundtruth_classes, dtype=tf.float32)

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_is_crowd = tf.expand_dims(groundtruth_is_crowd, 0)
      groundtruth_area = tf.expand_dims(groundtruth_area, 0)
      groundtruth_keypoints = tf.expand_dims(groundtruth_keypoints, 0)
      detection_boxes = tf.expand_dims(detection_boxes, 0)
      detection_scores = tf.expand_dims(detection_scores, 0)
      detection_classes = tf.expand_dims(detection_classes, 0)
      detection_keypoints = tf.expand_dims(detection_keypoints, 0)

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

      if groundtruth_keypoint_visibilities is None:
        groundtruth_keypoint_visibilities = tf.fill([
            tf.shape(groundtruth_boxes)[1],
            tf.shape(groundtruth_keypoints)[2]
        ], tf.constant(2, dtype=tf.int32))
      groundtruth_keypoint_visibilities = tf.expand_dims(
          groundtruth_keypoint_visibilities, 0)
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
      if groundtruth_keypoint_visibilities is None:
        groundtruth_keypoint_visibilities = tf.fill([
            tf.shape(groundtruth_keypoints)[1],
            tf.shape(groundtruth_keypoints)[2]
        ], tf.constant(2, dtype=tf.int32))
        groundtruth_keypoint_visibilities = tf.tile(
            tf.expand_dims(groundtruth_keypoint_visibilities, 0),
            multiples=[tf.shape(groundtruth_keypoints)[0], 1, 1])

    return tf.py_func(update_op, [
        image_id, groundtruth_boxes, groundtruth_classes, groundtruth_is_crowd,
        groundtruth_area, groundtruth_keypoints,
        groundtruth_keypoint_visibilities, num_gt_boxes_per_image,
        detection_boxes, detection_scores, detection_classes,
        detection_keypoints, num_det_boxes_per_image, is_annotated
    ], [])

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
    update_op = self.add_eval_dict(eval_dict)
    category = self._category_name
    metric_names = [
        'Keypoints_Precision/mAP ByCategory/{}'.format(category),
        'Keypoints_Precision/mAP@.50IOU ByCategory/{}'.format(category),
        'Keypoints_Precision/mAP@.75IOU ByCategory/{}'.format(category),
        'Keypoints_Precision/mAP (large) ByCategory/{}'.format(category),
        'Keypoints_Precision/mAP (medium) ByCategory/{}'.format(category),
        'Keypoints_Recall/AR@1 ByCategory/{}'.format(category),
        'Keypoints_Recall/AR@10 ByCategory/{}'.format(category),
        'Keypoints_Recall/AR@100 ByCategory/{}'.format(category),
        'Keypoints_Recall/AR@100 (large) ByCategory/{}'.format(category),
        'Keypoints_Recall/AR@100 (medium) ByCategory/{}'.format(category)
    ]

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


class CocoMaskEvaluator(object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate COCO detection metrics."""

  def __init__(self, categories,
               include_metrics_per_category=False,
               all_metrics_per_category=False,
               super_categories=None):
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
      super_categories: None or a python dict mapping super-category names
        (strings) to lists of categories (corresponding to category names
        in the label_map).  Metrics are aggregated along these super-categories
        and added to the `per_category_ap` and are associated with the name
          `PerformanceBySuperCategory/<super-category-name>`.
    """
    super(CocoMaskEvaluator, self).__init__(categories)
    self._image_id_to_mask_shape_map = {}
    self._image_ids_with_detections = set([])
    self._groundtruth_list = []
    self._detection_masks_list = []
    self._category_id_set = set([cat['id'] for cat in self._categories])
    self._annotation_id = 1
    self._include_metrics_per_category = include_metrics_per_category
    self._super_categories = super_categories
    self._all_metrics_per_category = all_metrics_per_category

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
        InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
          shape [num_boxes] containing iscrowd flag for groundtruth boxes.
        InputDataFields.groundtruth_area (optional): float numpy array of
          shape [num_boxes] containing the area (in the original absolute
          coordinates) of the annotated object.
    """
    if image_id in self._image_id_to_mask_shape_map:
      tf.logging.warning('Ignoring ground truth with image id %s since it was '
                         'previously added', image_id)
      return

    # Drop optional fields if empty tensor.
    groundtruth_is_crowd = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_is_crowd)
    groundtruth_area = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_area)
    if groundtruth_is_crowd is not None and not groundtruth_is_crowd.shape[0]:
      groundtruth_is_crowd = None
    if groundtruth_area is not None and not groundtruth_area.shape[0]:
      groundtruth_area = None

    groundtruth_instance_masks = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_instance_masks]
    groundtruth_instance_masks = convert_masks_to_binary(
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
            groundtruth_masks=groundtruth_instance_masks,
            groundtruth_is_crowd=groundtruth_is_crowd,
            groundtruth_area=groundtruth_area))
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
    detection_masks = convert_masks_to_binary(detection_masks)
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
        If super_categories are provided, then this will additionally include
      metrics aggregated along the super_categories with keys of the form:
      `PerformanceBySuperCategory/<super-category-name>`
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
        include_metrics_per_category=self._include_metrics_per_category,
        super_categories=self._super_categories,
        all_metrics_per_category=self._all_metrics_per_category)
    mask_metrics.update(mask_per_category_ap)
    mask_metrics = {'DetectionMasks_'+ key: value
                    for key, value in mask_metrics.items()}
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
                  groundtruth_is_crowd_batched, num_gt_boxes_per_image,
                  detection_scores_batched, detection_classes_batched,
                  detection_masks_batched, num_det_boxes_per_image,
                  original_image_spatial_shape):
      """Update op for metrics."""

      for (image_id, groundtruth_boxes, groundtruth_classes,
           groundtruth_instance_masks, groundtruth_is_crowd, num_gt_box,
           detection_scores, detection_classes,
           detection_masks, num_det_box, original_image_shape) in zip(
               image_id_batched, groundtruth_boxes_batched,
               groundtruth_classes_batched, groundtruth_instance_masks_batched,
               groundtruth_is_crowd_batched, num_gt_boxes_per_image,
               detection_scores_batched, detection_classes_batched,
               detection_masks_batched, num_det_boxes_per_image,
               original_image_spatial_shape):
        self.add_single_ground_truth_image_info(
            image_id, {
                'groundtruth_boxes':
                    groundtruth_boxes[:num_gt_box],
                'groundtruth_classes':
                    groundtruth_classes[:num_gt_box],
                'groundtruth_instance_masks':
                    groundtruth_instance_masks[
                        :num_gt_box,
                        :original_image_shape[0],
                        :original_image_shape[1]],
                'groundtruth_is_crowd':
                    groundtruth_is_crowd[:num_gt_box]
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
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    original_image_spatial_shape = eval_dict[
        input_data_fields.original_image_spatial_shape]
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

    return tf.py_func(update_op, [
        image_id, groundtruth_boxes, groundtruth_classes,
        groundtruth_instance_masks, groundtruth_is_crowd,
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


class CocoPanopticSegmentationEvaluator(
    object_detection_evaluation.DetectionEvaluator):
  """Class to evaluate PQ (panoptic quality) metric on COCO dataset.

  More details about this metric: https://arxiv.org/pdf/1801.00868.pdf.
  """

  def __init__(self,
               categories,
               include_metrics_per_category=False,
               iou_threshold=0.5,
               ioa_threshold=0.5):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
      include_metrics_per_category: If True, include metrics for each category.
      iou_threshold: intersection-over-union threshold for mask matching (with
        normal groundtruths).
      ioa_threshold: intersection-over-area threshold for mask matching with
        "is_crowd" groundtruths.
    """
    super(CocoPanopticSegmentationEvaluator, self).__init__(categories)
    self._groundtruth_masks = {}
    self._groundtruth_class_labels = {}
    self._groundtruth_is_crowd = {}
    self._predicted_masks = {}
    self._predicted_class_labels = {}
    self._include_metrics_per_category = include_metrics_per_category
    self._iou_threshold = iou_threshold
    self._ioa_threshold = ioa_threshold

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._groundtruth_masks.clear()
    self._groundtruth_class_labels.clear()
    self._groundtruth_is_crowd.clear()
    self._predicted_masks.clear()
    self._predicted_class_labels.clear()

  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    If the image has already been added, a warning is logged, and groundtruth is
    ignored.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary containing -
        InputDataFields.groundtruth_classes: integer numpy array of shape
          [num_masks] containing 1-indexed groundtruth classes for the mask.
        InputDataFields.groundtruth_instance_masks: uint8 numpy array of shape
          [num_masks, image_height, image_width] containing groundtruth masks.
          The elements of the array must be in {0, 1}.
        InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
          shape [num_boxes] containing iscrowd flag for groundtruth boxes.
    """

    if image_id in self._groundtruth_masks:
      tf.logging.warning(
          'Ignoring groundtruth with image %s, since it has already been '
          'added to the ground truth database.', image_id)
      return

    self._groundtruth_masks[image_id] = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_instance_masks]
    self._groundtruth_class_labels[image_id] = groundtruth_dict[
        standard_fields.InputDataFields.groundtruth_classes]
    groundtruth_is_crowd = groundtruth_dict.get(
        standard_fields.InputDataFields.groundtruth_is_crowd)
    # Drop groundtruth_is_crowd if empty tensor.
    if groundtruth_is_crowd is not None and not groundtruth_is_crowd.size > 0:
      groundtruth_is_crowd = None
    if groundtruth_is_crowd is not None:
      self._groundtruth_is_crowd[image_id] = groundtruth_is_crowd

  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    If a detection has already been added for this image id, a warning is
    logged, and the detection is skipped.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary containing -
        DetectionResultFields.detection_classes: integer numpy array of shape
          [num_masks] containing 1-indexed detection classes for the masks.
        DetectionResultFields.detection_masks: optional uint8 numpy array of
          shape [num_masks, image_height, image_width] containing instance
          masks. The elements of the array must be in {0, 1}.

    Raises:
      ValueError: If results and groundtruth shape don't match.
    """

    if image_id not in self._groundtruth_masks:
      raise ValueError('Missing groundtruth for image id: {}'.format(image_id))

    detection_masks = detections_dict[
        standard_fields.DetectionResultFields.detection_masks]
    self._predicted_masks[image_id] = detection_masks
    self._predicted_class_labels[image_id] = detections_dict[
        standard_fields.DetectionResultFields.detection_classes]
    groundtruth_mask_shape = self._groundtruth_masks[image_id].shape
    if groundtruth_mask_shape[1:] != detection_masks.shape[1:]:
      raise ValueError("The shape of results doesn't match groundtruth.")

  def evaluate(self):
    """Evaluates the detection masks and returns a dictionary of coco metrics.

    Returns:
      A dictionary holding -

      1. summary_metric:
      'PanopticQuality@%.2fIOU': mean panoptic quality averaged over classes at
        the required IOU.
      'SegmentationQuality@%.2fIOU': mean segmentation quality averaged over
        classes at the required IOU.
      'RecognitionQuality@%.2fIOU': mean recognition quality averaged over
        classes at the required IOU.
      'NumValidClasses': number of valid classes. A valid class should have at
        least one normal (is_crowd=0) groundtruth mask or one predicted mask.
      'NumTotalClasses': number of total classes.

      2. per_category_pq: if include_metrics_per_category is True, category
      specific results with keys of the form:
      'PanopticQuality@%.2fIOU_ByCategory/category'.
    """
    # Evaluate and accumulate the iou/tp/fp/fn.
    sum_tp_iou, sum_num_tp, sum_num_fp, sum_num_fn = self._evaluate_all_masks()
    # Compute PQ metric for each category and average over all classes.
    mask_metrics = self._compute_panoptic_metrics(sum_tp_iou, sum_num_tp,
                                                  sum_num_fp, sum_num_fn)
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
        'num_gt_masks_per_image' and 'num_det_masks_per_image' to properly unpad
        the tensors from the batch.

    Returns:
      a dictionary of metric names to tuple of value_op and update_op that can
      be used as eval metric ops in tf.estimator.EstimatorSpec. Note that all
      update ops  must be run together and similarly all value ops must be run
      together to guarantee correct behaviour.
    """

    def update_op(image_id_batched, groundtruth_classes_batched,
                  groundtruth_instance_masks_batched,
                  groundtruth_is_crowd_batched, num_gt_masks_per_image,
                  detection_classes_batched, detection_masks_batched,
                  num_det_masks_per_image):
      """Update op for metrics."""
      for (image_id, groundtruth_classes, groundtruth_instance_masks,
           groundtruth_is_crowd, num_gt_mask, detection_classes,
           detection_masks, num_det_mask) in zip(
               image_id_batched, groundtruth_classes_batched,
               groundtruth_instance_masks_batched, groundtruth_is_crowd_batched,
               num_gt_masks_per_image, detection_classes_batched,
               detection_masks_batched, num_det_masks_per_image):

        self.add_single_ground_truth_image_info(
            image_id, {
                'groundtruth_classes':
                    groundtruth_classes[:num_gt_mask],
                'groundtruth_instance_masks':
                    groundtruth_instance_masks[:num_gt_mask],
                'groundtruth_is_crowd':
                    groundtruth_is_crowd[:num_gt_mask]
            })
        self.add_single_detected_image_info(
            image_id, {
                'detection_classes': detection_classes[:num_det_mask],
                'detection_masks': detection_masks[:num_det_mask]
            })

    # Unpack items from the evaluation dictionary.
    (image_id, groundtruth_classes, groundtruth_instance_masks,
     groundtruth_is_crowd, num_gt_masks_per_image, detection_classes,
     detection_masks, num_det_masks_per_image
    ) = self._unpack_evaluation_dictionary_items(eval_dict)

    update_op = tf.py_func(update_op, [
        image_id, groundtruth_classes, groundtruth_instance_masks,
        groundtruth_is_crowd, num_gt_masks_per_image, detection_classes,
        detection_masks, num_det_masks_per_image
    ], [])

    metric_names = [
        'PanopticQuality@%.2fIOU' % self._iou_threshold,
        'SegmentationQuality@%.2fIOU' % self._iou_threshold,
        'RecognitionQuality@%.2fIOU' % self._iou_threshold
    ]
    if self._include_metrics_per_category:
      for category_dict in self._categories:
        metric_names.append('PanopticQuality@%.2fIOU_ByCategory/%s' %
                            (self._iou_threshold, category_dict['name']))

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

  def _evaluate_all_masks(self):
    """Evaluate all masks and compute sum iou/TP/FP/FN."""

    sum_num_tp = {category['id']: 0 for category in self._categories}
    sum_num_fp = sum_num_tp.copy()
    sum_num_fn = sum_num_tp.copy()
    sum_tp_iou = sum_num_tp.copy()

    for image_id in self._groundtruth_class_labels:
      # Separate normal and is_crowd groundtruth
      crowd_gt_indices = self._groundtruth_is_crowd.get(image_id)
      (normal_gt_masks, normal_gt_classes, crowd_gt_masks,
       crowd_gt_classes) = self._separate_normal_and_crowd_labels(
           crowd_gt_indices, self._groundtruth_masks[image_id],
           self._groundtruth_class_labels[image_id])

      # Mask matching to normal GT.
      predicted_masks = self._predicted_masks[image_id]
      predicted_class_labels = self._predicted_class_labels[image_id]
      (overlaps, pred_matched,
       gt_matched) = self._match_predictions_to_groundtruths(
           predicted_masks,
           predicted_class_labels,
           normal_gt_masks,
           normal_gt_classes,
           self._iou_threshold,
           is_crowd=False,
           with_replacement=False)

      # Accumulate true positives.
      for (class_id, is_matched, overlap) in zip(predicted_class_labels,
                                                 pred_matched, overlaps):
        if is_matched:
          sum_num_tp[class_id] += 1
          sum_tp_iou[class_id] += overlap

      # Accumulate false negatives.
      for (class_id, is_matched) in zip(normal_gt_classes, gt_matched):
        if not is_matched:
          sum_num_fn[class_id] += 1

      # Match remaining predictions to crowd gt.
      remained_pred_indices = np.logical_not(pred_matched)
      remained_pred_masks = predicted_masks[remained_pred_indices, :, :]
      remained_pred_classes = predicted_class_labels[remained_pred_indices]
      _, pred_matched, _ = self._match_predictions_to_groundtruths(
          remained_pred_masks,
          remained_pred_classes,
          crowd_gt_masks,
          crowd_gt_classes,
          self._ioa_threshold,
          is_crowd=True,
          with_replacement=True)

      # Accumulate false positives
      for (class_id, is_matched) in zip(remained_pred_classes, pred_matched):
        if not is_matched:
          sum_num_fp[class_id] += 1
    return sum_tp_iou, sum_num_tp, sum_num_fp, sum_num_fn

  def _compute_panoptic_metrics(self, sum_tp_iou, sum_num_tp, sum_num_fp,
                                sum_num_fn):
    """Compute PQ metric for each category and average over all classes.

    Args:
      sum_tp_iou: dict, summed true positive intersection-over-union (IoU) for
        each class, keyed by class_id.
      sum_num_tp: the total number of true positives for each class, keyed by
        class_id.
      sum_num_fp: the total number of false positives for each class, keyed by
        class_id.
      sum_num_fn: the total number of false negatives for each class, keyed by
        class_id.

    Returns:
      mask_metrics: a dictionary containing averaged metrics over all classes,
        and per-category metrics if required.
    """
    mask_metrics = {}
    sum_pq = 0
    sum_sq = 0
    sum_rq = 0
    num_valid_classes = 0
    for category in self._categories:
      class_id = category['id']
      (panoptic_quality, segmentation_quality,
       recognition_quality) = self._compute_panoptic_metrics_single_class(
           sum_tp_iou[class_id], sum_num_tp[class_id], sum_num_fp[class_id],
           sum_num_fn[class_id])
      if panoptic_quality is not None:
        sum_pq += panoptic_quality
        sum_sq += segmentation_quality
        sum_rq += recognition_quality
        num_valid_classes += 1
        if self._include_metrics_per_category:
          mask_metrics['PanopticQuality@%.2fIOU_ByCategory/%s' %
                       (self._iou_threshold,
                        category['name'])] = panoptic_quality
    mask_metrics['PanopticQuality@%.2fIOU' %
                 self._iou_threshold] = sum_pq / num_valid_classes
    mask_metrics['SegmentationQuality@%.2fIOU' %
                 self._iou_threshold] = sum_sq / num_valid_classes
    mask_metrics['RecognitionQuality@%.2fIOU' %
                 self._iou_threshold] = sum_rq / num_valid_classes
    mask_metrics['NumValidClasses'] = num_valid_classes
    mask_metrics['NumTotalClasses'] = len(self._categories)
    return mask_metrics

  def _compute_panoptic_metrics_single_class(self, sum_tp_iou, num_tp, num_fp,
                                             num_fn):
    """Compute panoptic metrics: panoptic/segmentation/recognition quality.

    More computation details in https://arxiv.org/pdf/1801.00868.pdf.
    Args:
      sum_tp_iou: summed true positive intersection-over-union (IoU) for a
        specific class.
      num_tp: the total number of true positives for a specific class.
      num_fp: the total number of false positives for a specific class.
      num_fn: the total number of false negatives for a specific class.

    Returns:
      panoptic_quality: sum_tp_iou / (num_tp + 0.5*num_fp + 0.5*num_fn).
      segmentation_quality: sum_tp_iou / num_tp.
      recognition_quality: num_tp / (num_tp + 0.5*num_fp + 0.5*num_fn).
    """
    denominator = num_tp + 0.5 * num_fp + 0.5 * num_fn
    # Calculate metric only if there is at least one GT or one prediction.
    if denominator > 0:
      recognition_quality = num_tp / denominator
      if num_tp > 0:
        segmentation_quality = sum_tp_iou / num_tp
      else:
        # If there is no TP for this category.
        segmentation_quality = 0
      panoptic_quality = segmentation_quality * recognition_quality
      return panoptic_quality, segmentation_quality, recognition_quality
    else:
      return None, None, None

  def _separate_normal_and_crowd_labels(self, crowd_gt_indices,
                                        groundtruth_masks, groundtruth_classes):
    """Separate normal and crowd groundtruth class_labels and masks.

    Args:
      crowd_gt_indices: None or array of shape [num_groundtruths]. If None, all
        groundtruths are treated as normal ones.
      groundtruth_masks: array of shape [num_groundtruths, height, width].
      groundtruth_classes: array of shape [num_groundtruths].

    Returns:
      normal_gt_masks: array of shape [num_normal_groundtruths, height, width].
      normal_gt_classes: array of shape [num_normal_groundtruths].
      crowd_gt_masks: array of shape [num_crowd_groundtruths, height, width].
      crowd_gt_classes: array of shape [num_crowd_groundtruths].
    Raises:
      ValueError: if the shape of groundtruth classes doesn't match groundtruth
        masks or if the shape of crowd_gt_indices.
    """
    if groundtruth_masks.shape[0] != groundtruth_classes.shape[0]:
      raise ValueError(
          "The number of masks doesn't match the number of labels.")
    if crowd_gt_indices is None:
      # All gts are treated as normal
      crowd_gt_indices = np.zeros(groundtruth_masks.shape, dtype=bool)
    else:
      if groundtruth_masks.shape[0] != crowd_gt_indices.shape[0]:
        raise ValueError(
            "The number of masks doesn't match the number of is_crowd labels.")
      crowd_gt_indices = crowd_gt_indices.astype(bool)
    normal_gt_indices = np.logical_not(crowd_gt_indices)
    if normal_gt_indices.size:
      normal_gt_masks = groundtruth_masks[normal_gt_indices, :, :]
      normal_gt_classes = groundtruth_classes[normal_gt_indices]
      crowd_gt_masks = groundtruth_masks[crowd_gt_indices, :, :]
      crowd_gt_classes = groundtruth_classes[crowd_gt_indices]
    else:
      # No groundtruths available, groundtruth_masks.shape = (0, h, w)
      normal_gt_masks = groundtruth_masks
      normal_gt_classes = groundtruth_classes
      crowd_gt_masks = groundtruth_masks
      crowd_gt_classes = groundtruth_classes
    return normal_gt_masks, normal_gt_classes, crowd_gt_masks, crowd_gt_classes

  def _match_predictions_to_groundtruths(self,
                                         predicted_masks,
                                         predicted_classes,
                                         groundtruth_masks,
                                         groundtruth_classes,
                                         matching_threshold,
                                         is_crowd=False,
                                         with_replacement=False):
    """Match the predicted masks to groundtruths.

    Args:
      predicted_masks: array of shape [num_predictions, height, width].
      predicted_classes: array of shape [num_predictions].
      groundtruth_masks: array of shape [num_groundtruths, height, width].
      groundtruth_classes: array of shape [num_groundtruths].
      matching_threshold: if the overlap between a prediction and a groundtruth
        is larger than this threshold, the prediction is true positive.
      is_crowd: whether the groundtruths are crowd annotation or not. If True,
        use intersection over area (IoA) as the overlapping metric; otherwise
        use intersection over union (IoU).
      with_replacement: whether a groundtruth can be matched to multiple
        predictions. By default, for normal groundtruths, only 1-1 matching is
        allowed for normal groundtruths; for crowd groundtruths, 1-to-many must
        be allowed.

    Returns:
      best_overlaps: array of shape [num_predictions]. Values representing the
      IoU
        or IoA with best matched groundtruth.
      pred_matched: array of shape [num_predictions]. Boolean value representing
        whether the ith prediction is matched to a groundtruth.
      gt_matched: array of shape [num_groundtruth]. Boolean value representing
        whether the ith groundtruth is matched to a prediction.
    Raises:
      ValueError: if the shape of groundtruth/predicted masks doesn't match
        groundtruth/predicted classes.
    """
    if groundtruth_masks.shape[0] != groundtruth_classes.shape[0]:
      raise ValueError(
          "The number of GT masks doesn't match the number of labels.")
    if predicted_masks.shape[0] != predicted_classes.shape[0]:
      raise ValueError(
          "The number of predicted masks doesn't match the number of labels.")
    gt_matched = np.zeros(groundtruth_classes.shape, dtype=bool)
    pred_matched = np.zeros(predicted_classes.shape, dtype=bool)
    best_overlaps = np.zeros(predicted_classes.shape)
    for pid in range(predicted_classes.shape[0]):
      best_overlap = 0
      matched_gt_id = -1
      for gid in range(groundtruth_classes.shape[0]):
        if predicted_classes[pid] == groundtruth_classes[gid]:
          if (not with_replacement) and gt_matched[gid]:
            continue
          if not is_crowd:
            overlap = np_mask_ops.iou(predicted_masks[pid:pid + 1],
                                      groundtruth_masks[gid:gid + 1])[0, 0]
          else:
            overlap = np_mask_ops.ioa(groundtruth_masks[gid:gid + 1],
                                      predicted_masks[pid:pid + 1])[0, 0]
          if overlap >= matching_threshold and overlap > best_overlap:
            matched_gt_id = gid
            best_overlap = overlap
      if matched_gt_id >= 0:
        gt_matched[matched_gt_id] = True
        pred_matched[pid] = True
        best_overlaps[pid] = best_overlap
    return best_overlaps, pred_matched, gt_matched

  def _unpack_evaluation_dictionary_items(self, eval_dict):
    """Unpack items from the evaluation dictionary."""
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    image_id = eval_dict[input_data_fields.key]
    groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
    groundtruth_instance_masks = eval_dict[
        input_data_fields.groundtruth_instance_masks]
    groundtruth_is_crowd = eval_dict.get(input_data_fields.groundtruth_is_crowd,
                                         None)
    num_gt_masks_per_image = eval_dict.get(
        input_data_fields.num_groundtruth_boxes, None)
    detection_classes = eval_dict[detection_fields.detection_classes]
    detection_masks = eval_dict[detection_fields.detection_masks]
    num_det_masks_per_image = eval_dict.get(detection_fields.num_detections,
                                            None)
    if groundtruth_is_crowd is None:
      groundtruth_is_crowd = tf.zeros_like(groundtruth_classes, dtype=tf.bool)

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)
      groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
      groundtruth_instance_masks = tf.expand_dims(groundtruth_instance_masks, 0)
      groundtruth_is_crowd = tf.expand_dims(groundtruth_is_crowd, 0)
      detection_classes = tf.expand_dims(detection_classes, 0)
      detection_masks = tf.expand_dims(detection_masks, 0)

      if num_gt_masks_per_image is None:
        num_gt_masks_per_image = tf.shape(groundtruth_classes)[1:2]
      else:
        num_gt_masks_per_image = tf.expand_dims(num_gt_masks_per_image, 0)

      if num_det_masks_per_image is None:
        num_det_masks_per_image = tf.shape(detection_classes)[1:2]
      else:
        num_det_masks_per_image = tf.expand_dims(num_det_masks_per_image, 0)
    else:
      if num_gt_masks_per_image is None:
        num_gt_masks_per_image = tf.tile(
            tf.shape(groundtruth_classes)[1:2],
            multiples=tf.shape(groundtruth_classes)[0:1])
      if num_det_masks_per_image is None:
        num_det_masks_per_image = tf.tile(
            tf.shape(detection_classes)[1:2],
            multiples=tf.shape(detection_classes)[0:1])
    return (image_id, groundtruth_classes, groundtruth_instance_masks,
            groundtruth_is_crowd, num_gt_masks_per_image, detection_classes,
            detection_masks, num_det_masks_per_image)
