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
"""Tests for tensorflow_models.object_detection.utils.vrd_evaluation."""

import numpy as np
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.utils import vrd_evaluation


class VRDRelationDetectionEvaluatorTest(tf.test.TestCase):

  def test_vrdrelation_evaluator(self):
    self.vrd_eval = vrd_evaluation.VRDRelationDetectionEvaluator()

    image_key1 = 'img1'
    groundtruth_box_tuples1 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2]), ([0, 0, 1, 1], [1, 2, 2, 3])],
        dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples1 = np.array(
        [(1, 2, 3), (1, 4, 3)], dtype=vrd_evaluation.label_data_type)
    groundtruth_verified_labels1 = np.array([1, 2, 3, 4, 5], dtype=int)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key1, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_box_tuples1,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_tuples1,
            standard_fields.InputDataFields.groundtruth_image_classes:
                groundtruth_verified_labels1
        })

    image_key2 = 'img2'
    groundtruth_box_tuples2 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples2 = np.array(
        [(1, 4, 3)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key2, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_box_tuples2,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_tuples2,
        })

    image_key3 = 'img3'
    groundtruth_box_tuples3 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples3 = np.array(
        [(1, 2, 4)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key3, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_box_tuples3,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_tuples3,
        })

    image_key = 'img1'
    detected_box_tuples = np.array(
        [([0, 0.3, 1, 1], [1.1, 1, 2, 2]), ([0, 0, 1, 1], [1, 1, 2, 2]),
         ([0.5, 0, 1, 1], [1, 1, 3, 3])],
        dtype=vrd_evaluation.vrd_box_data_type)
    detected_class_tuples = np.array(
        [(1, 2, 5), (1, 2, 3), (1, 6, 3)], dtype=vrd_evaluation.label_data_type)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    self.vrd_eval.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                detected_box_tuples,
            standard_fields.DetectionResultFields.detection_scores:
                detected_scores,
            standard_fields.DetectionResultFields.detection_classes:
                detected_class_tuples
        })

    metrics = self.vrd_eval.evaluate()

    self.assertAlmostEqual(metrics['VRDMetric_Relationships_weightedAP@0.5IOU'],
                           0.25)
    self.assertAlmostEqual(metrics['VRDMetric_Relationships_mAP@0.5IOU'],
                           0.1666666666666666)
    self.assertAlmostEqual(metrics['VRDMetric_Relationships_AP@0.5IOU/3'],
                           0.3333333333333333)
    self.assertAlmostEqual(metrics['VRDMetric_Relationships_AP@0.5IOU/4'], 0)
    self.assertAlmostEqual(metrics['VRDMetric_Relationships_Recall@50@0.5IOU'],
                           0.25)
    self.assertAlmostEqual(metrics['VRDMetric_Relationships_Recall@100@0.5IOU'],
                           0.25)
    self.vrd_eval.clear()
    self.assertFalse(self.vrd_eval._image_ids)


class VRDPhraseDetectionEvaluatorTest(tf.test.TestCase):

  def test_vrdphrase_evaluator(self):
    self.vrd_eval = vrd_evaluation.VRDPhraseDetectionEvaluator()

    image_key1 = 'img1'
    groundtruth_box_tuples1 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2]), ([0, 0, 1, 1], [1, 2, 2, 3])],
        dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples1 = np.array(
        [(1, 2, 3), (1, 4, 3)], dtype=vrd_evaluation.label_data_type)
    groundtruth_verified_labels1 = np.array([1, 2, 3, 4, 5], dtype=int)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key1, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_box_tuples1,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_tuples1,
            standard_fields.InputDataFields.groundtruth_image_classes:
                groundtruth_verified_labels1
        })

    image_key2 = 'img2'
    groundtruth_box_tuples2 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples2 = np.array(
        [(1, 4, 3)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key2, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_box_tuples2,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_tuples2,
        })

    image_key3 = 'img3'
    groundtruth_box_tuples3 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples3 = np.array(
        [(1, 2, 4)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key3, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_box_tuples3,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_tuples3,
        })

    image_key = 'img1'
    detected_box_tuples = np.array(
        [([0, 0.3, 0.5, 0.5], [0.3, 0.3, 1.0, 1.0]),
         ([0, 0, 1.2, 1.2], [0.0, 0.0, 2.0, 2.0]),
         ([0.5, 0, 1, 1], [1, 1, 3, 3])],
        dtype=vrd_evaluation.vrd_box_data_type)
    detected_class_tuples = np.array(
        [(1, 2, 5), (1, 2, 3), (1, 6, 3)], dtype=vrd_evaluation.label_data_type)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    self.vrd_eval.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                detected_box_tuples,
            standard_fields.DetectionResultFields.detection_scores:
                detected_scores,
            standard_fields.DetectionResultFields.detection_classes:
                detected_class_tuples
        })

    metrics = self.vrd_eval.evaluate()

    self.assertAlmostEqual(metrics['VRDMetric_Phrases_weightedAP@0.5IOU'], 0.25)
    self.assertAlmostEqual(metrics['VRDMetric_Phrases_mAP@0.5IOU'],
                           0.1666666666666666)
    self.assertAlmostEqual(metrics['VRDMetric_Phrases_AP@0.5IOU/3'],
                           0.3333333333333333)
    self.assertAlmostEqual(metrics['VRDMetric_Phrases_AP@0.5IOU/4'], 0)
    self.assertAlmostEqual(metrics['VRDMetric_Phrases_Recall@50@0.5IOU'], 0.25)
    self.assertAlmostEqual(metrics['VRDMetric_Phrases_Recall@100@0.5IOU'], 0.25)

    self.vrd_eval.clear()
    self.assertFalse(self.vrd_eval._image_ids)


class VRDDetectionEvaluationTest(tf.test.TestCase):

  def setUp(self):

    self.vrd_eval = vrd_evaluation._VRDDetectionEvaluation(
        matching_iou_threshold=0.5)

    image_key1 = 'img1'
    groundtruth_box_tuples1 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2]), ([0, 0, 1, 1], [1, 2, 2, 3])],
        dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples1 = np.array(
        [(1, 2, 3), (1, 4, 3)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key1, groundtruth_box_tuples1, groundtruth_class_tuples1)

    image_key2 = 'img2'
    groundtruth_box_tuples2 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples2 = np.array(
        [(1, 4, 3)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key2, groundtruth_box_tuples2, groundtruth_class_tuples2)

    image_key3 = 'img3'
    groundtruth_box_tuples3 = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=vrd_evaluation.vrd_box_data_type)
    groundtruth_class_tuples3 = np.array(
        [(1, 2, 4)], dtype=vrd_evaluation.label_data_type)
    self.vrd_eval.add_single_ground_truth_image_info(
        image_key3, groundtruth_box_tuples3, groundtruth_class_tuples3)

    image_key = 'img1'
    detected_box_tuples = np.array(
        [([0, 0.3, 1, 1], [1.1, 1, 2, 2]), ([0, 0, 1, 1], [1, 1, 2, 2])],
        dtype=vrd_evaluation.vrd_box_data_type)
    detected_class_tuples = np.array(
        [(1, 2, 3), (1, 2, 3)], dtype=vrd_evaluation.label_data_type)
    detected_scores = np.array([0.7, 0.8], dtype=float)
    self.vrd_eval.add_single_detected_image_info(
        image_key, detected_box_tuples, detected_scores, detected_class_tuples)

    metrics = self.vrd_eval.evaluate()
    expected_weighted_average_precision = 0.25
    expected_mean_average_precision = 0.16666666666666
    expected_precision = np.array([1., 0.5], dtype=float)
    expected_recall = np.array([0.25, 0.25], dtype=float)
    expected_recall_50 = 0.25
    expected_recall_100 = 0.25
    expected_median_rank_50 = 0
    expected_median_rank_100 = 0

    self.assertAlmostEqual(expected_weighted_average_precision,
                           metrics.weighted_average_precision)
    self.assertAlmostEqual(expected_mean_average_precision,
                           metrics.mean_average_precision)
    self.assertAlmostEqual(expected_mean_average_precision,
                           metrics.mean_average_precision)

    self.assertAllClose(expected_precision, metrics.precisions)
    self.assertAllClose(expected_recall, metrics.recalls)
    self.assertAlmostEqual(expected_recall_50, metrics.recall_50)
    self.assertAlmostEqual(expected_recall_100, metrics.recall_100)
    self.assertAlmostEqual(expected_median_rank_50, metrics.median_rank_50)
    self.assertAlmostEqual(expected_median_rank_100, metrics.median_rank_100)


if __name__ == '__main__':
  tf.test.main()
