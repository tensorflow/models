# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow_models.object_detection.metrics.calibration_evaluation."""  # pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import tensorflow.compat.v1 as tf
from object_detection.core import standard_fields
from object_detection.metrics import calibration_evaluation
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


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class CalibrationDetectionEvaluationTest(tf.test.TestCase):

  def _get_ece(self, ece_op, update_op):
    """Return scalar expected calibration error."""
    with self.test_session() as sess:
      metrics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
      sess.run(tf.variables_initializer(var_list=metrics_vars))
      _ = sess.run(update_op)
    return sess.run(ece_op)

  def testGetECEWithMatchingGroundtruthAndDetections(self):
    """Tests that ECE is calculated correctly when box matches exist."""
    calibration_evaluator = calibration_evaluation.CalibrationDetectionEvaluator(
        _get_categories_list(), iou_threshold=0.5)
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    # All gt and detection boxes match.
    base_eval_dict = {
        input_data_fields.key:
            tf.constant(['image_1', 'image_2', 'image_3']),
        input_data_fields.groundtruth_boxes:
            tf.constant([[[100., 100., 200., 200.]],
                         [[50., 50., 100., 100.]],
                         [[25., 25., 50., 50.]]],
                        dtype=tf.float32),
        detection_fields.detection_boxes:
            tf.constant([[[100., 100., 200., 200.]],
                         [[50., 50., 100., 100.]],
                         [[25., 25., 50., 50.]]],
                        dtype=tf.float32),
        input_data_fields.groundtruth_classes:
            tf.constant([[1], [2], [3]], dtype=tf.int64),
        # Note that, in the zero ECE case, the detection class for image_2
        # should NOT match groundtruth, since the detection score is zero.
        detection_fields.detection_scores:
            tf.constant([[1.0], [0.0], [1.0]], dtype=tf.float32)
    }

    # Zero ECE (perfectly calibrated).
    zero_ece_eval_dict = base_eval_dict.copy()
    zero_ece_eval_dict[detection_fields.detection_classes] = tf.constant(
        [[1], [1], [3]], dtype=tf.int64)
    zero_ece_op, zero_ece_update_op = (
        calibration_evaluator.get_estimator_eval_metric_ops(zero_ece_eval_dict)
        ['CalibrationError/ExpectedCalibrationError'])
    zero_ece = self._get_ece(zero_ece_op, zero_ece_update_op)
    self.assertAlmostEqual(zero_ece, 0.0)

    # ECE of 1 (poorest calibration).
    one_ece_eval_dict = base_eval_dict.copy()
    one_ece_eval_dict[detection_fields.detection_classes] = tf.constant(
        [[3], [2], [1]], dtype=tf.int64)
    one_ece_op, one_ece_update_op = (
        calibration_evaluator.get_estimator_eval_metric_ops(one_ece_eval_dict)
        ['CalibrationError/ExpectedCalibrationError'])
    one_ece = self._get_ece(one_ece_op, one_ece_update_op)
    self.assertAlmostEqual(one_ece, 1.0)

  def testGetECEWithUnmatchedGroundtruthAndDetections(self):
    """Tests that ECE is correctly calculated when boxes are unmatched."""
    calibration_evaluator = calibration_evaluation.CalibrationDetectionEvaluator(
        _get_categories_list(), iou_threshold=0.5)
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    # No gt and detection boxes match.
    eval_dict = {
        input_data_fields.key:
            tf.constant(['image_1', 'image_2', 'image_3']),
        input_data_fields.groundtruth_boxes:
            tf.constant([[[100., 100., 200., 200.]],
                         [[50., 50., 100., 100.]],
                         [[25., 25., 50., 50.]]],
                        dtype=tf.float32),
        detection_fields.detection_boxes:
            tf.constant([[[50., 50., 100., 100.]],
                         [[25., 25., 50., 50.]],
                         [[100., 100., 200., 200.]]],
                        dtype=tf.float32),
        input_data_fields.groundtruth_classes:
            tf.constant([[1], [2], [3]], dtype=tf.int64),
        detection_fields.detection_classes:
            tf.constant([[1], [1], [3]], dtype=tf.int64),
        # Detection scores of zero when boxes are unmatched = ECE of zero.
        detection_fields.detection_scores:
            tf.constant([[0.0], [0.0], [0.0]], dtype=tf.float32)
    }

    ece_op, update_op = calibration_evaluator.get_estimator_eval_metric_ops(
        eval_dict)['CalibrationError/ExpectedCalibrationError']
    ece = self._get_ece(ece_op, update_op)
    self.assertAlmostEqual(ece, 0.0)

  def testGetECEWithBatchedDetections(self):
    """Tests that ECE is correct with multiple detections per image."""
    calibration_evaluator = calibration_evaluation.CalibrationDetectionEvaluator(
        _get_categories_list(), iou_threshold=0.5)
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    # Note that image_2 has mismatched classes and detection scores but should
    # still produce ECE of 0 because detection scores are also 0.
    eval_dict = {
        input_data_fields.key:
            tf.constant(['image_1', 'image_2', 'image_3']),
        input_data_fields.groundtruth_boxes:
            tf.constant([[[100., 100., 200., 200.], [50., 50., 100., 100.]],
                         [[50., 50., 100., 100.], [100., 100., 200., 200.]],
                         [[25., 25., 50., 50.], [100., 100., 200., 200.]]],
                        dtype=tf.float32),
        detection_fields.detection_boxes:
            tf.constant([[[100., 100., 200., 200.], [50., 50., 100., 100.]],
                         [[50., 50., 100., 100.], [25., 25., 50., 50.]],
                         [[25., 25., 50., 50.], [100., 100., 200., 200.]]],
                        dtype=tf.float32),
        input_data_fields.groundtruth_classes:
            tf.constant([[1, 2], [2, 3], [3, 1]], dtype=tf.int64),
        detection_fields.detection_classes:
            tf.constant([[1, 2], [1, 1], [3, 1]], dtype=tf.int64),
        detection_fields.detection_scores:
            tf.constant([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=tf.float32)
    }

    ece_op, update_op = calibration_evaluator.get_estimator_eval_metric_ops(
        eval_dict)['CalibrationError/ExpectedCalibrationError']
    ece = self._get_ece(ece_op, update_op)
    self.assertAlmostEqual(ece, 0.0)

  def testGetECEWhenImagesFilteredByIsAnnotated(self):
    """Tests that ECE is correct when detections filtered by is_annotated."""
    calibration_evaluator = calibration_evaluation.CalibrationDetectionEvaluator(
        _get_categories_list(), iou_threshold=0.5)
    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    # ECE will be 0 only if the third image is filtered by is_annotated.
    eval_dict = {
        input_data_fields.key:
            tf.constant(['image_1', 'image_2', 'image_3']),
        input_data_fields.groundtruth_boxes:
            tf.constant([[[100., 100., 200., 200.]],
                         [[50., 50., 100., 100.]],
                         [[25., 25., 50., 50.]]],
                        dtype=tf.float32),
        detection_fields.detection_boxes:
            tf.constant([[[100., 100., 200., 200.]],
                         [[50., 50., 100., 100.]],
                         [[25., 25., 50., 50.]]],
                        dtype=tf.float32),
        input_data_fields.groundtruth_classes:
            tf.constant([[1], [2], [1]], dtype=tf.int64),
        detection_fields.detection_classes:
            tf.constant([[1], [1], [3]], dtype=tf.int64),
        detection_fields.detection_scores:
            tf.constant([[1.0], [0.0], [1.0]], dtype=tf.float32),
        'is_annotated': tf.constant([True, True, False], dtype=tf.bool)
    }

    ece_op, update_op = calibration_evaluator.get_estimator_eval_metric_ops(
        eval_dict)['CalibrationError/ExpectedCalibrationError']
    ece = self._get_ece(ece_op, update_op)
    self.assertAlmostEqual(ece, 0.0)

if __name__ == '__main__':
  tf.test.main()
