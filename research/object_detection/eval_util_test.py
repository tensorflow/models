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
"""Tests for eval_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import six
from six.moves import range
import tensorflow as tf

from object_detection import eval_util
from object_detection.core import standard_fields as fields
from object_detection.metrics import coco_evaluation
from object_detection.protos import eval_pb2
from object_detection.utils import test_case


class EvalUtilTest(test_case.TestCase, parameterized.TestCase):

  def _get_categories_list(self):
    return [{'id': 1, 'name': 'person'},
            {'id': 2, 'name': 'dog'},
            {'id': 3, 'name': 'cat'}]

  def _get_categories_list_with_keypoints(self):
    return [{
        'id': 1,
        'name': 'person',
        'keypoints': {
            'left_eye': 0,
            'right_eye': 3
        }
    }, {
        'id': 2,
        'name': 'dog',
        'keypoints': {
            'tail_start': 1,
            'mouth': 2
        }
    }, {
        'id': 3,
        'name': 'cat'
    }]

  def _make_evaluation_dict(self,
                            resized_groundtruth_masks=False,
                            batch_size=1,
                            max_gt_boxes=None,
                            scale_to_absolute=False):
    input_data_fields = fields.InputDataFields
    detection_fields = fields.DetectionResultFields

    image = tf.zeros(shape=[batch_size, 20, 20, 3], dtype=tf.uint8)
    if batch_size == 1:
      key = tf.constant('image1')
    else:
      key = tf.constant([str(i) for i in range(batch_size)])
    detection_boxes = tf.tile(tf.constant([[[0., 0., 1., 1.]]]),
                              multiples=[batch_size, 1, 1])
    detection_scores = tf.tile(tf.constant([[0.8]]), multiples=[batch_size, 1])
    detection_classes = tf.tile(tf.constant([[0]]), multiples=[batch_size, 1])
    detection_masks = tf.tile(tf.ones(shape=[1, 1, 20, 20], dtype=tf.float32),
                              multiples=[batch_size, 1, 1, 1])
    num_detections = tf.ones([batch_size])
    groundtruth_boxes = tf.constant([[0., 0., 1., 1.]])
    groundtruth_classes = tf.constant([1])
    groundtruth_instance_masks = tf.ones(shape=[1, 20, 20], dtype=tf.uint8)
    groundtruth_keypoints = tf.constant([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    if resized_groundtruth_masks:
      groundtruth_instance_masks = tf.ones(shape=[1, 10, 10], dtype=tf.uint8)

    if batch_size > 1:
      groundtruth_boxes = tf.tile(tf.expand_dims(groundtruth_boxes, 0),
                                  multiples=[batch_size, 1, 1])
      groundtruth_classes = tf.tile(tf.expand_dims(groundtruth_classes, 0),
                                    multiples=[batch_size, 1])
      groundtruth_instance_masks = tf.tile(
          tf.expand_dims(groundtruth_instance_masks, 0),
          multiples=[batch_size, 1, 1, 1])
      groundtruth_keypoints = tf.tile(
          tf.expand_dims(groundtruth_keypoints, 0),
          multiples=[batch_size, 1, 1])

    detections = {
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
        detection_fields.num_detections: num_detections
    }
    groundtruth = {
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_keypoints: groundtruth_keypoints,
        input_data_fields.groundtruth_instance_masks: groundtruth_instance_masks
    }
    if batch_size > 1:
      return eval_util.result_dict_for_batched_example(
          image, key, detections, groundtruth,
          scale_to_absolute=scale_to_absolute,
          max_gt_boxes=max_gt_boxes)
    else:
      return eval_util.result_dict_for_single_example(
          image, key, detections, groundtruth,
          scale_to_absolute=scale_to_absolute)

  @parameterized.parameters(
      {'batch_size': 1, 'max_gt_boxes': None, 'scale_to_absolute': True},
      {'batch_size': 8, 'max_gt_boxes': [1], 'scale_to_absolute': True},
      {'batch_size': 1, 'max_gt_boxes': None, 'scale_to_absolute': False},
      {'batch_size': 8, 'max_gt_boxes': [1], 'scale_to_absolute': False}
  )
  def test_get_eval_metric_ops_for_coco_detections(self, batch_size=1,
                                                   max_gt_boxes=None,
                                                   scale_to_absolute=False):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend(['coco_detection_metrics'])
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict(batch_size=batch_size,
                                           max_gt_boxes=max_gt_boxes,
                                           scale_to_absolute=scale_to_absolute)
    metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        eval_config, categories, eval_dict)
    _, update_op = metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      metrics = {}
      for key, (value_op, _) in six.iteritems(metric_ops):
        metrics[key] = value_op
      sess.run(update_op)
      metrics = sess.run(metrics)
      self.assertAlmostEqual(1.0, metrics['DetectionBoxes_Precision/mAP'])
      self.assertNotIn('DetectionMasks_Precision/mAP', metrics)

  @parameterized.parameters(
      {'batch_size': 1, 'max_gt_boxes': None, 'scale_to_absolute': True},
      {'batch_size': 8, 'max_gt_boxes': [1], 'scale_to_absolute': True},
      {'batch_size': 1, 'max_gt_boxes': None, 'scale_to_absolute': False},
      {'batch_size': 8, 'max_gt_boxes': [1], 'scale_to_absolute': False}
  )
  def test_get_eval_metric_ops_for_coco_detections_and_masks(
      self, batch_size=1, max_gt_boxes=None, scale_to_absolute=False):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend(
        ['coco_detection_metrics', 'coco_mask_metrics'])
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict(batch_size=batch_size,
                                           max_gt_boxes=max_gt_boxes,
                                           scale_to_absolute=scale_to_absolute)
    metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        eval_config, categories, eval_dict)
    _, update_op_boxes = metric_ops['DetectionBoxes_Precision/mAP']
    _, update_op_masks = metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      metrics = {}
      for key, (value_op, _) in six.iteritems(metric_ops):
        metrics[key] = value_op
      sess.run(update_op_boxes)
      sess.run(update_op_masks)
      metrics = sess.run(metrics)
      self.assertAlmostEqual(1.0, metrics['DetectionBoxes_Precision/mAP'])
      self.assertAlmostEqual(1.0, metrics['DetectionMasks_Precision/mAP'])

  @parameterized.parameters(
      {'batch_size': 1, 'max_gt_boxes': None, 'scale_to_absolute': True},
      {'batch_size': 8, 'max_gt_boxes': [1], 'scale_to_absolute': True},
      {'batch_size': 1, 'max_gt_boxes': None, 'scale_to_absolute': False},
      {'batch_size': 8, 'max_gt_boxes': [1], 'scale_to_absolute': False}
  )
  def test_get_eval_metric_ops_for_coco_detections_and_resized_masks(
      self, batch_size=1, max_gt_boxes=None, scale_to_absolute=False):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend(
        ['coco_detection_metrics', 'coco_mask_metrics'])
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict(batch_size=batch_size,
                                           max_gt_boxes=max_gt_boxes,
                                           scale_to_absolute=scale_to_absolute,
                                           resized_groundtruth_masks=True)
    metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        eval_config, categories, eval_dict)
    _, update_op_boxes = metric_ops['DetectionBoxes_Precision/mAP']
    _, update_op_masks = metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      metrics = {}
      for key, (value_op, _) in six.iteritems(metric_ops):
        metrics[key] = value_op
      sess.run(update_op_boxes)
      sess.run(update_op_masks)
      metrics = sess.run(metrics)
      self.assertAlmostEqual(1.0, metrics['DetectionBoxes_Precision/mAP'])
      self.assertAlmostEqual(1.0, metrics['DetectionMasks_Precision/mAP'])

  def test_get_eval_metric_ops_raises_error_with_unsupported_metric(self):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend(['unsupported_metric'])
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict()
    with self.assertRaises(ValueError):
      eval_util.get_eval_metric_ops_for_evaluators(
          eval_config, categories, eval_dict)

  def test_get_eval_metric_ops_for_evaluators(self):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend([
        'coco_detection_metrics', 'coco_mask_metrics',
        'precision_at_recall_detection_metrics'
    ])
    eval_config.include_metrics_per_category = True
    eval_config.recall_lower_bound = 0.2
    eval_config.recall_upper_bound = 0.6

    evaluator_options = eval_util.evaluator_options_from_eval_config(
        eval_config)
    self.assertTrue(evaluator_options['coco_detection_metrics']
                    ['include_metrics_per_category'])
    self.assertTrue(
        evaluator_options['coco_mask_metrics']['include_metrics_per_category'])
    self.assertAlmostEqual(
        evaluator_options['precision_at_recall_detection_metrics']
        ['recall_lower_bound'], eval_config.recall_lower_bound)
    self.assertAlmostEqual(
        evaluator_options['precision_at_recall_detection_metrics']
        ['recall_upper_bound'], eval_config.recall_upper_bound)

  def test_get_evaluator_with_evaluator_options(self):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend(
        ['coco_detection_metrics', 'precision_at_recall_detection_metrics'])
    eval_config.include_metrics_per_category = True
    eval_config.recall_lower_bound = 0.2
    eval_config.recall_upper_bound = 0.6
    categories = self._get_categories_list()

    evaluator_options = eval_util.evaluator_options_from_eval_config(
        eval_config)
    evaluator = eval_util.get_evaluators(eval_config, categories,
                                         evaluator_options)

    self.assertTrue(evaluator[0]._include_metrics_per_category)
    self.assertAlmostEqual(evaluator[1]._recall_lower_bound,
                           eval_config.recall_lower_bound)
    self.assertAlmostEqual(evaluator[1]._recall_upper_bound,
                           eval_config.recall_upper_bound)

  def test_get_evaluator_with_no_evaluator_options(self):
    eval_config = eval_pb2.EvalConfig()
    eval_config.metrics_set.extend(
        ['coco_detection_metrics', 'precision_at_recall_detection_metrics'])
    eval_config.include_metrics_per_category = True
    eval_config.recall_lower_bound = 0.2
    eval_config.recall_upper_bound = 0.6
    categories = self._get_categories_list()

    evaluator = eval_util.get_evaluators(
        eval_config, categories, evaluator_options=None)

    # Even though we are setting eval_config.include_metrics_per_category = True
    # and bounds on recall, these options are never passed into the
    # DetectionEvaluator constructor (via `evaluator_options`).
    self.assertFalse(evaluator[0]._include_metrics_per_category)
    self.assertAlmostEqual(evaluator[1]._recall_lower_bound, 0.0)
    self.assertAlmostEqual(evaluator[1]._recall_upper_bound, 1.0)

  def test_get_evaluator_with_keypoint_metrics(self):
    eval_config = eval_pb2.EvalConfig()
    person_keypoints_metric = eval_config.parameterized_metric.add()
    person_keypoints_metric.coco_keypoint_metrics.class_label = 'person'
    person_keypoints_metric.coco_keypoint_metrics.keypoint_label_to_sigmas[
        'left_eye'] = 0.1
    person_keypoints_metric.coco_keypoint_metrics.keypoint_label_to_sigmas[
        'right_eye'] = 0.2
    dog_keypoints_metric = eval_config.parameterized_metric.add()
    dog_keypoints_metric.coco_keypoint_metrics.class_label = 'dog'
    dog_keypoints_metric.coco_keypoint_metrics.keypoint_label_to_sigmas[
        'tail_start'] = 0.3
    dog_keypoints_metric.coco_keypoint_metrics.keypoint_label_to_sigmas[
        'mouth'] = 0.4
    categories = self._get_categories_list_with_keypoints()

    evaluator = eval_util.get_evaluators(
        eval_config, categories, evaluator_options=None)

    # Verify keypoint evaluator class variables.
    self.assertLen(evaluator, 3)
    self.assertFalse(evaluator[0]._include_metrics_per_category)
    self.assertEqual(evaluator[1]._category_name, 'person')
    self.assertEqual(evaluator[2]._category_name, 'dog')
    self.assertAllEqual(evaluator[1]._keypoint_ids, [0, 3])
    self.assertAllEqual(evaluator[2]._keypoint_ids, [1, 2])
    self.assertAllClose([0.1, 0.2], evaluator[1]._oks_sigmas)
    self.assertAllClose([0.3, 0.4], evaluator[2]._oks_sigmas)

  def test_get_evaluator_with_unmatched_label(self):
    eval_config = eval_pb2.EvalConfig()
    person_keypoints_metric = eval_config.parameterized_metric.add()
    person_keypoints_metric.coco_keypoint_metrics.class_label = 'unmatched'
    person_keypoints_metric.coco_keypoint_metrics.keypoint_label_to_sigmas[
        'kpt'] = 0.1
    categories = self._get_categories_list_with_keypoints()

    evaluator = eval_util.get_evaluators(
        eval_config, categories, evaluator_options=None)
    self.assertLen(evaluator, 1)
    self.assertNotIsInstance(
        evaluator[0], coco_evaluation.CocoKeypointEvaluator)

  def test_padded_image_result_dict(self):

    input_data_fields = fields.InputDataFields
    detection_fields = fields.DetectionResultFields
    key = tf.constant([str(i) for i in range(2)])

    detection_boxes = np.array([[[0., 0., 1., 1.]], [[0.0, 0.0, 0.5, 0.5]]],
                               dtype=np.float32)
    detection_keypoints = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
                                   dtype=np.float32)
    detections = {
        detection_fields.detection_boxes:
            tf.constant(detection_boxes),
        detection_fields.detection_scores:
            tf.constant([[1.], [1.]]),
        detection_fields.detection_classes:
            tf.constant([[1], [2]]),
        detection_fields.num_detections:
            tf.constant([1, 1]),
        detection_fields.detection_keypoints:
            tf.tile(
                tf.reshape(
                    tf.constant(detection_keypoints), shape=[1, 1, 3, 2]),
                multiples=[2, 1, 1, 1])
    }

    gt_boxes = detection_boxes
    groundtruth = {
        input_data_fields.groundtruth_boxes:
            tf.constant(gt_boxes),
        input_data_fields.groundtruth_classes:
            tf.constant([[1.], [1.]]),
        input_data_fields.groundtruth_keypoints:
            tf.tile(
                tf.reshape(
                    tf.constant(detection_keypoints), shape=[1, 1, 3, 2]),
                multiples=[2, 1, 1, 1])
    }

    image = tf.zeros((2, 100, 100, 3), dtype=tf.float32)

    true_image_shapes = tf.constant([[100, 100, 3], [50, 100, 3]])
    original_image_spatial_shapes = tf.constant([[200, 200], [150, 300]])

    result = eval_util.result_dict_for_batched_example(
        image, key, detections, groundtruth,
        scale_to_absolute=True,
        true_image_shapes=true_image_shapes,
        original_image_spatial_shapes=original_image_spatial_shapes,
        max_gt_boxes=tf.constant(1))

    with self.test_session() as sess:
      result = sess.run(result)
      self.assertAllEqual(
          [[[0., 0., 200., 200.]], [[0.0, 0.0, 150., 150.]]],
          result[input_data_fields.groundtruth_boxes])
      self.assertAllClose([[[[0., 0.], [100., 100.], [200., 200.]]],
                           [[[0., 0.], [150., 150.], [300., 300.]]]],
                          result[input_data_fields.groundtruth_keypoints])

      # Predictions from the model are not scaled.
      self.assertAllEqual(
          [[[0., 0., 200., 200.]], [[0.0, 0.0, 75., 150.]]],
          result[detection_fields.detection_boxes])
      self.assertAllClose([[[[0., 0.], [100., 100.], [200., 200.]]],
                           [[[0., 0.], [75., 150.], [150., 300.]]]],
                          result[detection_fields.detection_keypoints])


if __name__ == '__main__':
  tf.test.main()
