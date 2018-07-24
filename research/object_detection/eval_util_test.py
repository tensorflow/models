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

import tensorflow as tf


from object_detection import eval_util
from object_detection.core import standard_fields as fields


class EvalUtilTest(tf.test.TestCase):

  def _get_categories_list(self):
    return [{'id': 0, 'name': 'person'},
            {'id': 1, 'name': 'dog'},
            {'id': 2, 'name': 'cat'}]

  def _make_evaluation_dict(self, resized_groundtruth_masks=False):
    input_data_fields = fields.InputDataFields
    detection_fields = fields.DetectionResultFields

    image = tf.zeros(shape=[1, 20, 20, 3], dtype=tf.uint8)
    key = tf.constant('image1')
    detection_boxes = tf.constant([[[0., 0., 1., 1.]]])
    detection_scores = tf.constant([[0.8]])
    detection_classes = tf.constant([[0]])
    detection_masks = tf.ones(shape=[1, 1, 20, 20], dtype=tf.float32)
    num_detections = tf.constant([1])
    groundtruth_boxes = tf.constant([[0., 0., 1., 1.]])
    groundtruth_classes = tf.constant([1])
    groundtruth_instance_masks = tf.ones(shape=[1, 20, 20], dtype=tf.uint8)
    if resized_groundtruth_masks:
      groundtruth_instance_masks = tf.ones(shape=[1, 10, 10], dtype=tf.uint8)
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
    return eval_util.result_dict_for_single_example(image, key, detections,
                                                    groundtruth)

  def test_get_eval_metric_ops_for_coco_detections(self):
    evaluation_metrics = ['coco_detection_metrics']
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict()
    metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        evaluation_metrics, categories, eval_dict)
    _, update_op = metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      metrics = {}
      for key, (value_op, _) in metric_ops.iteritems():
        metrics[key] = value_op
      sess.run(update_op)
      metrics = sess.run(metrics)
      print(metrics)
      self.assertAlmostEqual(1.0, metrics['DetectionBoxes_Precision/mAP'])
      self.assertNotIn('DetectionMasks_Precision/mAP', metrics)

  def test_get_eval_metric_ops_for_coco_detections_and_masks(self):
    evaluation_metrics = ['coco_detection_metrics',
                          'coco_mask_metrics']
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict()
    metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        evaluation_metrics, categories, eval_dict)
    _, update_op_boxes = metric_ops['DetectionBoxes_Precision/mAP']
    _, update_op_masks = metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      metrics = {}
      for key, (value_op, _) in metric_ops.iteritems():
        metrics[key] = value_op
      sess.run(update_op_boxes)
      sess.run(update_op_masks)
      metrics = sess.run(metrics)
      self.assertAlmostEqual(1.0, metrics['DetectionBoxes_Precision/mAP'])
      self.assertAlmostEqual(1.0, metrics['DetectionMasks_Precision/mAP'])

  def test_get_eval_metric_ops_for_coco_detections_and_resized_masks(self):
    evaluation_metrics = ['coco_detection_metrics',
                          'coco_mask_metrics']
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict(resized_groundtruth_masks=True)
    metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
        evaluation_metrics, categories, eval_dict)
    _, update_op_boxes = metric_ops['DetectionBoxes_Precision/mAP']
    _, update_op_masks = metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      metrics = {}
      for key, (value_op, _) in metric_ops.iteritems():
        metrics[key] = value_op
      sess.run(update_op_boxes)
      sess.run(update_op_masks)
      metrics = sess.run(metrics)
      self.assertAlmostEqual(1.0, metrics['DetectionBoxes_Precision/mAP'])
      self.assertAlmostEqual(1.0, metrics['DetectionMasks_Precision/mAP'])

  def test_get_eval_metric_ops_raises_error_with_unsupported_metric(self):
    evaluation_metrics = ['unsupported_metrics']
    categories = self._get_categories_list()
    eval_dict = self._make_evaluation_dict()
    with self.assertRaises(ValueError):
      eval_util.get_eval_metric_ops_for_evaluators(
          evaluation_metrics, categories, eval_dict)


if __name__ == '__main__':
  tf.test.main()
