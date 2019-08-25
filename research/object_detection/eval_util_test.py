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

import six
from six.moves import range
import tensorflow as tf

from object_detection import eval_util
from object_detection.core import standard_fields as fields
from object_detection.protos import eval_pb2
from object_detection.utils import test_case


class EvalUtilTest(test_case.TestCase, parameterized.TestCase):

  def _get_categories_list(self):
    return [{'id': 1, 'name': 'person'},
            {'id': 2, 'name': 'dog'},
            {'id': 3, 'name': 'cat'}]

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


if __name__ == '__main__':
  tf.test.main()
