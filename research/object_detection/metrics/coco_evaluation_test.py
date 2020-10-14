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
"""Tests for tensorflow_models.object_detection.metrics.coco_evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import tf_version


def _get_categories_list():
  return [{
      'id': 1,
      'name': 'person'
  }, {
      'id': 2,
      'name': 'dog'
  }, {
      'id': 3,
      'name': 'cat'
  }]


def _get_category_keypoints_dict():
  return {
      'person': [{
          'id': 0,
          'name': 'left_eye'
      }, {
          'id': 3,
          'name': 'right_eye'
      }],
      'dog': [{
          'id': 1,
          'name': 'tail_start'
      }, {
          'id': 2,
          'name': 'mouth'
      }]
  }


class CocoDetectionEvaluationTest(tf.test.TestCase):

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    """Tests that mAP is calculated correctly on GT and Detections."""
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image3',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[25., 25., 50., 50.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image3',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[25., 25., 50., 50.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsSkipCrowd(self):
    """Tests computing mAP with is_crowd GT boxes skipped."""
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.], [99., 99., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1, 2]),
            standard_fields.InputDataFields.groundtruth_is_crowd:
                np.array([0, 1])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsEmptyCrowd(self):
    """Tests computing mAP with empty is_crowd array passed in."""
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_is_crowd:
                np.array([])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

  def testRejectionOnDuplicateGroundtruth(self):
    """Tests that groundtruth cannot be added more than once for an image."""
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    #  Add groundtruth
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    coco_evaluator.add_single_ground_truth_image_info(image_key1, {
        standard_fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes1,
        standard_fields.InputDataFields.groundtruth_classes:
            groundtruth_class_labels1
    })
    groundtruth_lists_len = len(coco_evaluator._groundtruth_list)

    # Add groundtruth with the same image id.
    coco_evaluator.add_single_ground_truth_image_info(image_key1, {
        standard_fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes1,
        standard_fields.InputDataFields.groundtruth_classes:
            groundtruth_class_labels1
    })
    self.assertEqual(groundtruth_lists_len,
                     len(coco_evaluator._groundtruth_list))

  def testRejectionOnDuplicateDetections(self):
    """Tests that detections cannot be added more than once for an image."""
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    #  Add groundtruth
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[99., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    detections_lists_len = len(coco_evaluator._detection_boxes_list)
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',  # Note that this image id was previously added.
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1])
        })
    self.assertEqual(detections_lists_len,
                     len(coco_evaluator._detection_boxes_list))

  def testExceptionRaisedWithMissingGroundtruth(self):
    """Tests that exception is raised for detection with missing groundtruth."""
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    with self.assertRaises(ValueError):
      coco_evaluator.add_single_detected_image_info(
          image_id='image1',
          detections_dict={
              standard_fields.DetectionResultFields.detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              standard_fields.DetectionResultFields.detection_scores:
                  np.array([.8]),
              standard_fields.DetectionResultFields.detection_classes:
                  np.array([1])
          })


@unittest.skipIf(tf_version.is_tf2(), 'Only Supported in TF1.X')
class CocoEvaluationPyFuncTest(tf.test.TestCase):

  def _MatchingGroundtruthAndDetections(self, coco_evaluator):
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      sess.run(update_op,
               feed_dict={
                   image_id: 'image1',
                   groundtruth_boxes: np.array([[100., 100., 200., 200.]]),
                   groundtruth_classes: np.array([1]),
                   detection_boxes: np.array([[100., 100., 200., 200.]]),
                   detection_scores: np.array([.8]),
                   detection_classes: np.array([1])
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image2',
                   groundtruth_boxes: np.array([[50., 50., 100., 100.]]),
                   groundtruth_classes: np.array([3]),
                   detection_boxes: np.array([[50., 50., 100., 100.]]),
                   detection_scores: np.array([.7]),
                   detection_classes: np.array([3])
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image3',
                   groundtruth_boxes: np.array([[25., 25., 50., 50.]]),
                   groundtruth_classes: np.array([2]),
                   detection_boxes: np.array([[25., 25., 50., 50.]]),
                   detection_scores: np.array([.9]),
                   detection_classes: np.array([2])
               })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_boxes_list)
    self.assertFalse(coco_evaluator._image_ids)

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    self._MatchingGroundtruthAndDetections(coco_evaluator)

  # Configured to skip unmatched detector predictions with
  # groundtruth_labeled_classes, but reverts to fully-labeled eval since there
  # are no groundtruth_labeled_classes set.
  def testGetMAPWithSkipUnmatchedPredictionsIgnoreGrountruthLabeledClasses(
      self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list(), skip_predictions_for_unlabeled_class=True)
    self._MatchingGroundtruthAndDetections(coco_evaluator)

  # Test skipping unmatched detector predictions with
  # groundtruth_labeled_classes.
  def testGetMAPWithSkipUnmatchedPredictions(self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list(), skip_predictions_for_unlabeled_class=True)
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_labeled_classes = tf.placeholder(tf.float32, shape=(None))
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key:
            image_id,
        input_data_fields.groundtruth_boxes:
            groundtruth_boxes,
        input_data_fields.groundtruth_classes:
            groundtruth_classes,
        input_data_fields.groundtruth_labeled_classes:
            groundtruth_labeled_classes,
        detection_fields.detection_boxes:
            detection_boxes,
        detection_fields.detection_scores:
            detection_scores,
        detection_fields.detection_classes:
            detection_classes
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              # Only class 1 is exhaustively labeled for image1.
              groundtruth_labeled_classes:
                  np.array([1]),
              detection_boxes:
                  np.array([[100., 100., 200., 200.], [100., 100., 200.,
                                                       200.]]),
              detection_scores:
                  np.array([.8, .95]),
              detection_classes:
                  np.array([1, 2])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id: 'image2',
              groundtruth_boxes: np.array([[50., 50., 100., 100.]]),
              groundtruth_classes: np.array([3]),
              groundtruth_labeled_classes: np.array([3]),
              detection_boxes: np.array([[50., 50., 100., 100.]]),
              detection_scores: np.array([.7]),
              detection_classes: np.array([3])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id: 'image3',
              groundtruth_boxes: np.array([[25., 25., 50., 50.]]),
              groundtruth_classes: np.array([2]),
              groundtruth_labeled_classes: np.array([2]),
              detection_boxes: np.array([[25., 25., 50., 50.]]),
              detection_scores: np.array([.9]),
              detection_classes: np.array([2])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_boxes_list)
    self.assertFalse(coco_evaluator._image_ids)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsIsAnnotated(self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    is_annotated = tf.placeholder(tf.bool, shape=())
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        'is_annotated': is_annotated,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      sess.run(update_op,
               feed_dict={
                   image_id: 'image1',
                   groundtruth_boxes: np.array([[100., 100., 200., 200.]]),
                   groundtruth_classes: np.array([1]),
                   is_annotated: True,
                   detection_boxes: np.array([[100., 100., 200., 200.]]),
                   detection_scores: np.array([.8]),
                   detection_classes: np.array([1])
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image2',
                   groundtruth_boxes: np.array([[50., 50., 100., 100.]]),
                   groundtruth_classes: np.array([3]),
                   is_annotated: True,
                   detection_boxes: np.array([[50., 50., 100., 100.]]),
                   detection_scores: np.array([.7]),
                   detection_classes: np.array([3])
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image3',
                   groundtruth_boxes: np.array([[25., 25., 50., 50.]]),
                   groundtruth_classes: np.array([2]),
                   is_annotated: True,
                   detection_boxes: np.array([[25., 25., 50., 50.]]),
                   detection_scores: np.array([.9]),
                   detection_classes: np.array([2])
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image4',
                   groundtruth_boxes: np.zeros((0, 4)),
                   groundtruth_classes: np.zeros((0)),
                   is_annotated: False,  # Note that this image isn't annotated.
                   detection_boxes: np.array([[25., 25., 50., 50.],
                                              [25., 25., 70., 50.],
                                              [25., 25., 80., 50.],
                                              [25., 25., 90., 50.]]),
                   detection_scores: np.array([0.6, 0.7, 0.8, 0.9]),
                   detection_classes: np.array([1, 2, 2, 3])
               })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_boxes_list)
    self.assertFalse(coco_evaluator._image_ids)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsPadded(self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.], [-1, -1, -1, -1]]),
              groundtruth_classes:
                  np.array([1, -1]),
              detection_boxes:
                  np.array([[100., 100., 200., 200.], [0., 0., 0., 0.]]),
              detection_scores:
                  np.array([.8, 0.]),
              detection_classes:
                  np.array([1, -1])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image2',
              groundtruth_boxes:
                  np.array([[50., 50., 100., 100.], [-1, -1, -1, -1]]),
              groundtruth_classes:
                  np.array([3, -1]),
              detection_boxes:
                  np.array([[50., 50., 100., 100.], [0., 0., 0., 0.]]),
              detection_scores:
                  np.array([.7, 0.]),
              detection_classes:
                  np.array([3, -1])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image3',
              groundtruth_boxes:
                  np.array([[25., 25., 50., 50.], [10., 10., 15., 15.]]),
              groundtruth_classes:
                  np.array([2, 2]),
              detection_boxes:
                  np.array([[25., 25., 50., 50.], [10., 10., 15., 15.]]),
              detection_scores:
                  np.array([.95, .9]),
              detection_classes:
                  np.array([2, 2])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 0.83333331)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_boxes_list)
    self.assertFalse(coco_evaluator._image_ids)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsBatched(self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    batch_size = 3
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      sess.run(update_op,
               feed_dict={
                   image_id: ['image1', 'image2', 'image3'],
                   groundtruth_boxes: np.array([[[100., 100., 200., 200.]],
                                                [[50., 50., 100., 100.]],
                                                [[25., 25., 50., 50.]]]),
                   groundtruth_classes: np.array([[1], [3], [2]]),
                   detection_boxes: np.array([[[100., 100., 200., 200.]],
                                              [[50., 50., 100., 100.]],
                                              [[25., 25., 50., 50.]]]),
                   detection_scores: np.array([[.8], [.7], [.9]]),
                   detection_classes: np.array([[1], [3], [2]])
               })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_boxes_list)
    self.assertFalse(coco_evaluator._image_ids)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsPaddedBatches(self):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
        _get_categories_list())
    batch_size = 3
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    num_gt_boxes_per_image = tf.placeholder(tf.int32, shape=(None))
    detection_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    num_det_boxes_per_image = tf.placeholder(tf.int32, shape=(None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        'num_groundtruth_boxes_per_image': num_gt_boxes_per_image,
        'num_det_boxes_per_image': num_det_boxes_per_image
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionBoxes_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id: ['image1', 'image2', 'image3'],
              groundtruth_boxes:
                  np.array([[[100., 100., 200., 200.], [-1, -1, -1, -1]],
                            [[50., 50., 100., 100.], [-1, -1, -1, -1]],
                            [[25., 25., 50., 50.], [10., 10., 15., 15.]]]),
              groundtruth_classes:
                  np.array([[1, -1], [3, -1], [2, 2]]),
              num_gt_boxes_per_image:
                  np.array([1, 1, 2]),
              detection_boxes:
                  np.array([[[100., 100., 200., 200.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.]],
                            [[50., 50., 100., 100.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.]],
                            [[25., 25., 50., 50.],
                             [10., 10., 15., 15.],
                             [10., 10., 15., 15.]]]),
              detection_scores:
                  np.array([[.8, 0., 0.], [.7, 0., 0.], [.95, .9, 0.9]]),
              detection_classes:
                  np.array([[1, -1, -1], [3, -1, -1], [2, 2, 2]]),
              num_det_boxes_per_image:
                  np.array([1, 1, 3]),
          })

    # Check the number of bounding boxes added.
    self.assertEqual(len(coco_evaluator._groundtruth_list), 4)
    self.assertEqual(len(coco_evaluator._detection_boxes_list), 5)

    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 0.83333331)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_boxes_list)
    self.assertFalse(coco_evaluator._image_ids)


class CocoKeypointEvaluationTest(tf.test.TestCase):

  def testGetOneMAPWithMatchingKeypoints(self):
    """Tests that correct mAP for keypoints is calculated."""
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[150., 160.], [float('nan'),
                                          float('nan')],
                           [float('nan'), float('nan')], [170., 180.]]]),
            standard_fields.InputDataFields.groundtruth_keypoint_visibilities:
                np.array([[2, 0, 0, 2]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[75., 76.], [float('nan'),
                                        float('nan')],
                           [float('nan'), float('nan')], [77., 78.]]]),
            standard_fields.InputDataFields.groundtruth_keypoint_visibilities:
                np.array([[2, 0, 0, 2]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)

  def testGroundtruthListValues(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[150., 160.], [float('nan'), float('nan')],
                           [float('nan'), float('nan')], [170., 180.]]]),
            standard_fields.InputDataFields.groundtruth_keypoint_visibilities:
                np.array([[2, 0, 0, 2]]),
            standard_fields.InputDataFields.groundtruth_area: np.array([15.])
        })
    gt_dict = coco_evaluator._groundtruth_list[0]
    self.assertEqual(gt_dict['id'], 1)
    self.assertAlmostEqual(gt_dict['bbox'], [100.0, 100.0, 100.0, 100.0])
    self.assertAlmostEqual(
        gt_dict['keypoints'], [160.0, 150.0, 2, 180.0, 170.0, 2])
    self.assertEqual(gt_dict['num_keypoints'], 2)
    self.assertAlmostEqual(gt_dict['area'], 15.0)

  def testKeypointVisibilitiesAreOptional(self):
    """Tests that evaluator works when visibilities aren't provided."""
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[150., 160.], [float('nan'),
                                          float('nan')],
                           [float('nan'), float('nan')], [170., 180.]]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[75., 76.], [float('nan'),
                                        float('nan')],
                           [float('nan'), float('nan')], [77., 78.]]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)

  def testFiltersDetectionsFromOtherCategories(self):
    """Tests that the evaluator ignores detections from other categories."""
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=2, category_keypoints=category_keypoint_dict['person'],
        class_text='dog')
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[150., 160.], [170., 180.], [110., 120.],
                           [130., 140.]]]),
            standard_fields.InputDataFields.groundtruth_keypoint_visibilities:
                np.array([[2, 2, 2, 2]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.9]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[150., 160.], [170., 180.], [110., 120.],
                           [130., 140.]]])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/dog'],
                           -1.0)

  def testHandlesUnlabeledKeypointData(self):
    """Tests that the evaluator handles missing keypoints GT."""
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[150., 160.], [float('nan'),
                                          float('nan')],
                           [float('nan'), float('nan')], [170., 180.]]]),
            standard_fields.InputDataFields.groundtruth_keypoint_visibilities:
                np.array([[0, 0, 0, 2]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[50., 60.], [1., 2.], [3., 4.], [170., 180.]]])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)

  def testIgnoresCrowdAnnotations(self):
    """Tests that the evaluator ignores GT marked as crowd."""
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_is_crowd:
                np.array([1]),
            standard_fields.InputDataFields.groundtruth_keypoints:
                np.array([[[150., 160.], [float('nan'),
                                          float('nan')],
                           [float('nan'), float('nan')], [170., 180.]]]),
            standard_fields.InputDataFields.groundtruth_keypoint_visibilities:
                np.array([[2, 0, 0, 2]])
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_keypoints:
                np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           -1.0)


@unittest.skipIf(tf_version.is_tf2(), 'Only Supported in TF1.X')
class CocoKeypointEvaluationPyFuncTest(tf.test.TestCase):

  def testGetOneMAPWithMatchingKeypoints(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_keypoints: groundtruth_keypoints,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_keypoints: detection_keypoints,
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[150., 160.], [float('nan'),
                                            float('nan')],
                             [float('nan'), float('nan')], [170., 180.]]]),
              detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              detection_scores:
                  np.array([.8]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image2',
              groundtruth_boxes:
                  np.array([[50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[75., 76.], [float('nan'),
                                          float('nan')],
                             [float('nan'), float('nan')], [77., 78.]]]),
              detection_boxes:
                  np.array([[50., 50., 100., 100.]]),
              detection_scores:
                  np.array([.7]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], 1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], 1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)

  def testGetOneMAPWithMatchingKeypointsAndVisibilities(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))
    groundtruth_keypoint_visibilities = tf.placeholder(
        tf.float32, shape=(None, 4))
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key:
            image_id,
        input_data_fields.groundtruth_boxes:
            groundtruth_boxes,
        input_data_fields.groundtruth_classes:
            groundtruth_classes,
        input_data_fields.groundtruth_keypoints:
            groundtruth_keypoints,
        input_data_fields.groundtruth_keypoint_visibilities:
            groundtruth_keypoint_visibilities,
        detection_fields.detection_boxes:
            detection_boxes,
        detection_fields.detection_scores:
            detection_scores,
        detection_fields.detection_classes:
            detection_classes,
        detection_fields.detection_keypoints:
            detection_keypoints,
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[150., 160.], [float('nan'),
                                            float('nan')],
                             [float('nan'), float('nan')], [170., 180.]]]),
              groundtruth_keypoint_visibilities:
                  np.array([[0, 0, 0, 2]]),
              detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              detection_scores:
                  np.array([.8]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[50., 60.], [1., 2.], [3., 4.], [170., 180.]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], -1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], -1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)

  def testGetOneMAPWithMatchingKeypointsIsAnnotated(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))
    is_annotated = tf.placeholder(tf.bool, shape=())
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_keypoints: groundtruth_keypoints,
        'is_annotated': is_annotated,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_keypoints: detection_keypoints,
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[150., 160.], [float('nan'),
                                            float('nan')],
                             [float('nan'), float('nan')], [170., 180.]]]),
              is_annotated:
                  True,
              detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              detection_scores:
                  np.array([.8]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image2',
              groundtruth_boxes:
                  np.array([[50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[75., 76.], [float('nan'),
                                          float('nan')],
                             [float('nan'), float('nan')], [77., 78.]]]),
              is_annotated:
                  True,
              detection_boxes:
                  np.array([[50., 50., 100., 100.]]),
              detection_scores:
                  np.array([.7]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image3',
              groundtruth_boxes:
                  np.zeros((0, 4)),
              groundtruth_classes:
                  np.zeros((0)),
              groundtruth_keypoints:
                  np.zeros((0, 4, 2)),
              is_annotated:
                  False,  # Note that this image isn't annotated.
              detection_boxes:
                  np.array([[25., 25., 50., 50.], [25., 25., 70., 50.],
                            [25., 25., 80., 50.], [25., 25., 90., 50.]]),
              detection_scores:
                  np.array([0.6, 0.7, 0.8, 0.9]),
              detection_classes:
                  np.array([1, 2, 2, 3]),
              detection_keypoints:
                  np.array([[[0., 0.], [0., 0.], [0., 0.], [0., 0.]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], 1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], 1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)

  def testGetOneMAPWithMatchingKeypointsBatched(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    batch_size = 2
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    groundtruth_keypoints = tf.placeholder(
        tf.float32, shape=(batch_size, None, 4, 2))
    detection_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_keypoints = tf.placeholder(
        tf.float32, shape=(batch_size, None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_keypoints: groundtruth_keypoints,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_keypoints: detection_keypoints
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id: ['image1', 'image2'],
              groundtruth_boxes:
                  np.array([[[100., 100., 200., 200.]], [[50., 50., 100.,
                                                          100.]]]),
              groundtruth_classes:
                  np.array([[1], [3]]),
              groundtruth_keypoints:
                  np.array([[[[150., 160.], [float('nan'),
                                             float('nan')],
                              [float('nan'), float('nan')], [170., 180.]]],
                            [[[75., 76.], [float('nan'),
                                           float('nan')],
                              [float('nan'), float('nan')], [77., 78.]]]]),
              detection_boxes:
                  np.array([[[100., 100., 200., 200.]], [[50., 50., 100.,
                                                          100.]]]),
              detection_scores:
                  np.array([[.8], [.7]]),
              detection_classes:
                  np.array([[1], [3]]),
              detection_keypoints:
                  np.array([[[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]],
                            [[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], -1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], -1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)


class CocoMaskEvaluationTest(tf.test.TestCase):

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
            np.pad(np.ones([1, 100, 100], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
            np.pad(np.ones([1, 100, 100], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
            np.pad(np.ones([1, 50, 50], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
            np.pad(np.ones([1, 50, 50], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image3',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[25., 25., 50., 50.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
            np.pad(np.ones([1, 25, 25], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image3',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                np.array([[25., 25., 50., 50.]]),
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
                # The value of 5 is equivalent to 1, since masks will be
                # thresholded and binarized before evaluation.
                np.pad(5 * np.ones([1, 25, 25], dtype=np.uint8),
                       ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
    coco_evaluator.clear()
    self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
    self.assertFalse(coco_evaluator._image_ids_with_detections)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_masks_list)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsSkipCrowd(self):
    """Tests computing mAP with is_crowd GT boxes skipped."""
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(
        _get_categories_list())
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.], [99., 99., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array([1, 2]),
            standard_fields.InputDataFields.groundtruth_is_crowd:
                np.array([0, 1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
                np.concatenate(
                    [np.pad(np.ones([1, 100, 100], dtype=np.uint8),
                            ((0, 0), (100, 56), (100, 56)), mode='constant'),
                     np.pad(np.ones([1, 101, 101], dtype=np.uint8),
                            ((0, 0), (99, 56), (99, 56)), mode='constant')],
                    axis=0)
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
                np.pad(np.ones([1, 100, 100], dtype=np.uint8),
                       ((0, 0), (100, 56), (100, 56)), mode='constant')
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)


@unittest.skipIf(tf_version.is_tf2(), 'Only Supported in TF1.X')
class CocoMaskEvaluationPyFuncTest(tf.test.TestCase):

  def testAddEvalDict(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_masks = tf.placeholder(tf.uint8, shape=(None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }
    update_op = coco_evaluator.add_eval_dict(eval_dict)
    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.], [50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1, 2]),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant')
                  ]),
              detection_scores:
                  np.array([.9, .8]),
              detection_classes:
                  np.array([2, 1]),
              detection_masks:
                  np.stack([
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant'),
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                  ])
          })
      self.assertLen(coco_evaluator._groundtruth_list, 2)
      self.assertLen(coco_evaluator._detection_masks_list, 2)

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_masks = tf.placeholder(tf.uint8, shape=(None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.], [50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1, 2]),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant')
                  ]),
              detection_scores:
                  np.array([.9, .8]),
              detection_classes:
                  np.array([2, 1]),
              detection_masks:
                  np.stack([
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant'),
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                  ])
          })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image2',
                   groundtruth_boxes: np.array([[50., 50., 100., 100.]]),
                   groundtruth_classes: np.array([1]),
                   groundtruth_masks: np.pad(np.ones([1, 50, 50],
                                                     dtype=np.uint8),
                                             ((0, 0), (10, 10), (10, 10)),
                                             mode='constant'),
                   detection_scores: np.array([.8]),
                   detection_classes: np.array([1]),
                   detection_masks: np.pad(np.ones([1, 50, 50], dtype=np.uint8),
                                           ((0, 0), (10, 10), (10, 10)),
                                           mode='constant')
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image3',
                   groundtruth_boxes: np.array([[25., 25., 50., 50.]]),
                   groundtruth_classes: np.array([1]),
                   groundtruth_masks: np.pad(np.ones([1, 25, 25],
                                                     dtype=np.uint8),
                                             ((0, 0), (10, 10), (10, 10)),
                                             mode='constant'),
                   detection_scores: np.array([.8]),
                   detection_classes: np.array([1]),
                   detection_masks: np.pad(np.ones([1, 25, 25],
                                                   dtype=np.uint8),
                                           ((0, 0), (10, 10), (10, 10)),
                                           mode='constant')
               })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._image_ids_with_detections)
    self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
    self.assertFalse(coco_evaluator._detection_masks_list)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsBatched(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    batch_size = 3
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    groundtruth_masks = tf.placeholder(
        tf.uint8, shape=(batch_size, None, None, None))
    detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_masks = tf.placeholder(
        tf.uint8, shape=(batch_size, None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id: ['image1', 'image2', 'image3'],
              groundtruth_boxes:
                  np.array([[[100., 100., 200., 200.]],
                            [[50., 50., 100., 100.]],
                            [[25., 25., 50., 50.]]]),
              groundtruth_classes:
                  np.array([[1], [1], [1]]),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          np.ones([1, 100, 100], dtype=np.uint8),
                          ((0, 0), (0, 0), (0, 0)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 50, 50], dtype=np.uint8),
                          ((0, 0), (25, 25), (25, 25)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 25, 25], dtype=np.uint8),
                          ((0, 0), (37, 38), (37, 38)),
                          mode='constant')
                  ],
                           axis=0),
              detection_scores:
                  np.array([[.8], [.8], [.8]]),
              detection_classes:
                  np.array([[1], [1], [1]]),
              detection_masks:
                  np.stack([
                      np.pad(
                          np.ones([1, 100, 100], dtype=np.uint8),
                          ((0, 0), (0, 0), (0, 0)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 50, 50], dtype=np.uint8),
                          ((0, 0), (25, 25), (25, 25)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 25, 25], dtype=np.uint8),
                          ((0, 0), (37, 38), (37, 38)),
                          mode='constant')
                  ],
                           axis=0)
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._image_ids_with_detections)
    self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
    self.assertFalse(coco_evaluator._detection_masks_list)


def _get_panoptic_test_data():
  # image1 contains 3 people in gt, (2 normal annotation and 1 "is_crowd"
  # annotation), and 3 people in prediction.
  gt_masks1 = np.zeros((3, 50, 50), dtype=np.uint8)
  result_masks1 = np.zeros((3, 50, 50), dtype=np.uint8)
  gt_masks1[0, 10:20, 20:30] = 1
  result_masks1[0, 10:18, 20:30] = 1
  gt_masks1[1, 25:30, 25:35] = 1
  result_masks1[1, 18:25, 25:30] = 1
  gt_masks1[2, 40:50, 40:50] = 1
  result_masks1[2, 47:50, 47:50] = 1
  gt_class1 = np.array([1, 1, 1])
  gt_is_crowd1 = np.array([0, 0, 1])
  result_class1 = np.array([1, 1, 1])

  # image2 contains 1 dog and 1 cat in gt, while 1 person and 1 dog in
  # prediction.
  gt_masks2 = np.zeros((2, 30, 40), dtype=np.uint8)
  result_masks2 = np.zeros((2, 30, 40), dtype=np.uint8)
  gt_masks2[0, 5:15, 20:35] = 1
  gt_masks2[1, 20:30, 0:10] = 1
  result_masks2[0, 20:25, 10:15] = 1
  result_masks2[1, 6:15, 15:35] = 1
  gt_class2 = np.array([2, 3])
  gt_is_crowd2 = np.array([0, 0])
  result_class2 = np.array([1, 2])

  gt_class = [gt_class1, gt_class2]
  gt_masks = [gt_masks1, gt_masks2]
  gt_is_crowd = [gt_is_crowd1, gt_is_crowd2]
  result_class = [result_class1, result_class2]
  result_masks = [result_masks1, result_masks2]
  return gt_class, gt_masks, gt_is_crowd, result_class, result_masks


class CocoPanopticEvaluationTest(tf.test.TestCase):

  def test_panoptic_quality(self):
    pq_evaluator = coco_evaluation.CocoPanopticSegmentationEvaluator(
        _get_categories_list(), include_metrics_per_category=True)
    (gt_class, gt_masks, gt_is_crowd, result_class,
     result_masks) = _get_panoptic_test_data()

    for i in range(2):
      pq_evaluator.add_single_ground_truth_image_info(
          image_id='image%d' % i,
          groundtruth_dict={
              standard_fields.InputDataFields.groundtruth_classes:
                  gt_class[i],
              standard_fields.InputDataFields.groundtruth_instance_masks:
                  gt_masks[i],
              standard_fields.InputDataFields.groundtruth_is_crowd:
                  gt_is_crowd[i]
          })

      pq_evaluator.add_single_detected_image_info(
          image_id='image%d' % i,
          detections_dict={
              standard_fields.DetectionResultFields.detection_classes:
                  result_class[i],
              standard_fields.DetectionResultFields.detection_masks:
                  result_masks[i]
          })

    metrics = pq_evaluator.evaluate()
    self.assertAlmostEqual(metrics['PanopticQuality@0.50IOU_ByCategory/person'],
                           0.32)
    self.assertAlmostEqual(metrics['PanopticQuality@0.50IOU_ByCategory/dog'],
                           135.0 / 195)
    self.assertAlmostEqual(metrics['PanopticQuality@0.50IOU_ByCategory/cat'], 0)
    self.assertAlmostEqual(metrics['SegmentationQuality@0.50IOU'],
                           (0.8 + 135.0 / 195) / 3)
    self.assertAlmostEqual(metrics['RecognitionQuality@0.50IOU'], (0.4 + 1) / 3)
    self.assertAlmostEqual(metrics['PanopticQuality@0.50IOU'],
                           (0.32 + 135.0 / 195) / 3)
    self.assertEqual(metrics['NumValidClasses'], 3)
    self.assertEqual(metrics['NumTotalClasses'], 3)


@unittest.skipIf(tf_version.is_tf2(), 'Only Supported in TF1.X')
class CocoPanopticEvaluationPyFuncTest(tf.test.TestCase):

  def testPanopticQualityNoBatch(self):
    pq_evaluator = coco_evaluation.CocoPanopticSegmentationEvaluator(
        _get_categories_list(), include_metrics_per_category=True)

    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_classes = tf.placeholder(tf.int32, shape=(None))
    groundtruth_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
    groundtruth_is_crowd = tf.placeholder(tf.int32, shape=(None))
    detection_classes = tf.placeholder(tf.int32, shape=(None))
    detection_masks = tf.placeholder(tf.uint8, shape=(None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        input_data_fields.groundtruth_is_crowd: groundtruth_is_crowd,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }

    eval_metric_ops = pq_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['PanopticQuality@0.50IOU']
    (gt_class, gt_masks, gt_is_crowd, result_class,
     result_masks) = _get_panoptic_test_data()

    with self.test_session() as sess:
      for i in range(2):
        sess.run(
            update_op,
            feed_dict={
                image_id: 'image%d' % i,
                groundtruth_classes: gt_class[i],
                groundtruth_masks: gt_masks[i],
                groundtruth_is_crowd: gt_is_crowd[i],
                detection_classes: result_class[i],
                detection_masks: result_masks[i]
            })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['PanopticQuality@0.50IOU'],
                           (0.32 + 135.0 / 195) / 3)

  def testPanopticQualityBatched(self):
    pq_evaluator = coco_evaluation.CocoPanopticSegmentationEvaluator(
        _get_categories_list(), include_metrics_per_category=True)
    batch_size = 2
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_classes = tf.placeholder(tf.int32, shape=(batch_size, None))
    groundtruth_masks = tf.placeholder(
        tf.uint8, shape=(batch_size, None, None, None))
    groundtruth_is_crowd = tf.placeholder(tf.int32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.int32, shape=(batch_size, None))
    detection_masks = tf.placeholder(
        tf.uint8, shape=(batch_size, None, None, None))
    num_gt_masks_per_image = tf.placeholder(tf.int32, shape=(batch_size))
    num_det_masks_per_image = tf.placeholder(tf.int32, shape=(batch_size))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        input_data_fields.groundtruth_is_crowd: groundtruth_is_crowd,
        input_data_fields.num_groundtruth_boxes: num_gt_masks_per_image,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
        detection_fields.num_detections: num_det_masks_per_image,
    }

    eval_metric_ops = pq_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['PanopticQuality@0.50IOU']
    (gt_class, gt_masks, gt_is_crowd, result_class,
     result_masks) = _get_panoptic_test_data()
    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id: ['image0', 'image1'],
              groundtruth_classes:
                  np.stack([
                      gt_class[0],
                      np.pad(gt_class[1], (0, 1), mode='constant')
                  ],
                           axis=0),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          gt_masks[0], ((0, 0), (0, 10), (0, 10)),
                          mode='constant'),
                      np.pad(
                          gt_masks[1], ((0, 1), (0, 30), (0, 20)),
                          mode='constant'),
                  ],
                           axis=0),
              groundtruth_is_crowd:
                  np.stack([
                      gt_is_crowd[0],
                      np.pad(gt_is_crowd[1], (0, 1), mode='constant')
                  ],
                           axis=0),
              num_gt_masks_per_image: np.array([3, 2]),
              detection_classes:
                  np.stack([
                      result_class[0],
                      np.pad(result_class[1], (0, 1), mode='constant')
                  ],
                           axis=0),
              detection_masks:
                  np.stack([
                      np.pad(
                          result_masks[0], ((0, 0), (0, 10), (0, 10)),
                          mode='constant'),
                      np.pad(
                          result_masks[1], ((0, 1), (0, 30), (0, 20)),
                          mode='constant'),
                  ],
                           axis=0),
              num_det_masks_per_image: np.array([3, 2]),
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['PanopticQuality@0.50IOU'],
                           (0.32 + 135.0 / 195) / 3)


if __name__ == '__main__':
  tf.test.main()
