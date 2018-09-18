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

"""Tests for object_detection.utils.object_detection_evaluation."""

import numpy as np
import tensorflow as tf

from object_detection.core import standard_fields
from object_detection.utils import object_detection_evaluation


class OpenImagesV2EvaluationTest(tf.test.TestCase):

  def test_returns_correct_metric_values(self):
    categories = [{
        'id': 1,
        'name': 'cat'
    }, {
        'id': 2,
        'name': 'dog'
    }, {
        'id': 3,
        'name': 'elephant'
    }]

    oiv2_evaluator = object_detection_evaluation.OpenImagesDetectionEvaluator(
        categories)
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array(
        [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    oiv2_evaluator.add_single_ground_truth_image_info(image_key1, {
        standard_fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes1,
        standard_fields.InputDataFields.groundtruth_classes:
            groundtruth_class_labels1,
        standard_fields.InputDataFields.groundtruth_group_of:
            np.array([], dtype=bool)
    })
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array(
        [[10, 10, 11, 11], [500, 500, 510, 510], [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([1, 1, 3], dtype=int)
    groundtruth_is_group_of_list2 = np.array([False, True, False], dtype=bool)
    oiv2_evaluator.add_single_ground_truth_image_info(image_key2, {
        standard_fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes2,
        standard_fields.InputDataFields.groundtruth_classes:
            groundtruth_class_labels2,
        standard_fields.InputDataFields.groundtruth_group_of:
            groundtruth_is_group_of_list2
    })
    image_key3 = 'img3'
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([2], dtype=int)
    oiv2_evaluator.add_single_ground_truth_image_info(image_key3, {
        standard_fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes3,
        standard_fields.InputDataFields.groundtruth_classes:
            groundtruth_class_labels3
    })
    # Add detections
    image_key = 'img2'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        dtype=float)
    detected_class_labels = np.array([1, 1, 3], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    oiv2_evaluator.add_single_detected_image_info(image_key, {
        standard_fields.DetectionResultFields.detection_boxes:
            detected_boxes,
        standard_fields.DetectionResultFields.detection_scores:
            detected_scores,
        standard_fields.DetectionResultFields.detection_classes:
            detected_class_labels
    })
    metrics = oiv2_evaluator.evaluate()
    self.assertAlmostEqual(
        metrics['OpenImagesV2_PerformanceByCategory/AP@0.5IOU/dog'], 0.0)
    self.assertAlmostEqual(
        metrics['OpenImagesV2_PerformanceByCategory/AP@0.5IOU/elephant'], 0.0)
    self.assertAlmostEqual(
        metrics['OpenImagesV2_PerformanceByCategory/AP@0.5IOU/cat'], 0.16666666)
    self.assertAlmostEqual(metrics['OpenImagesV2_Precision/mAP@0.5IOU'],
                           0.05555555)
    oiv2_evaluator.clear()
    self.assertFalse(oiv2_evaluator._image_ids)


class OpenImagesDetectionChallengeEvaluatorTest(tf.test.TestCase):

  def test_returns_correct_metric_values(self):
    categories = [{
        'id': 1,
        'name': 'cat'
    }, {
        'id': 2,
        'name': 'dog'
    }, {
        'id': 3,
        'name': 'elephant'
    }]
    oivchallenge_evaluator = (
        object_detection_evaluation.OpenImagesDetectionChallengeEvaluator(
            categories, group_of_weight=0.5))

    image_key = 'img1'
    groundtruth_boxes = np.array(
        [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], dtype=float)
    groundtruth_class_labels = np.array([1, 3, 1], dtype=int)
    groundtruth_is_group_of_list = np.array([False, False, True], dtype=bool)
    groundtruth_verified_labels = np.array([1, 2, 3], dtype=int)
    oivchallenge_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels,
            standard_fields.InputDataFields.groundtruth_group_of:
                groundtruth_is_group_of_list,
            standard_fields.InputDataFields.groundtruth_image_classes:
                groundtruth_verified_labels,
        })
    image_key = 'img2'
    groundtruth_boxes = np.array(
        [[10, 10, 11, 11], [500, 500, 510, 510], [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels = np.array([1, 1, 3], dtype=int)
    groundtruth_is_group_of_list = np.array([False, False, True], dtype=bool)
    oivchallenge_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels,
            standard_fields.InputDataFields.groundtruth_group_of:
                groundtruth_is_group_of_list
        })
    image_key = 'img3'
    groundtruth_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels = np.array([2], dtype=int)
    oivchallenge_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels
        })
    image_key = 'img1'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120]], dtype=float)
    detected_class_labels = np.array([2, 2], dtype=int)
    detected_scores = np.array([0.7, 0.8], dtype=float)
    oivchallenge_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                detected_boxes,
            standard_fields.DetectionResultFields.detection_scores:
                detected_scores,
            standard_fields.DetectionResultFields.detection_classes:
                detected_class_labels
        })
    image_key = 'img2'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220],
         [10, 10, 11, 11]],
        dtype=float)
    detected_class_labels = np.array([1, 1, 2, 3], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.5, 0.9], dtype=float)
    oivchallenge_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                detected_boxes,
            standard_fields.DetectionResultFields.detection_scores:
                detected_scores,
            standard_fields.DetectionResultFields.detection_classes:
                detected_class_labels
        })
    image_key = 'img3'
    detected_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    detected_class_labels = np.array([2], dtype=int)
    detected_scores = np.array([0.5], dtype=float)
    oivchallenge_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                detected_boxes,
            standard_fields.DetectionResultFields.detection_scores:
                detected_scores,
            standard_fields.DetectionResultFields.detection_classes:
                detected_class_labels
        })
    metrics = oivchallenge_evaluator.evaluate()

    self.assertAlmostEqual(
        metrics['OpenImagesChallenge2018_PerformanceByCategory/AP@0.5IOU/dog'],
        0.3333333333)
    self.assertAlmostEqual(
        metrics[
            'OpenImagesChallenge2018_PerformanceByCategory/AP@0.5IOU/elephant'],
        0.333333333333)
    self.assertAlmostEqual(
        metrics['OpenImagesChallenge2018_PerformanceByCategory/AP@0.5IOU/cat'],
        0.142857142857)
    self.assertAlmostEqual(
        metrics['OpenImagesChallenge2018_Precision/mAP@0.5IOU'], 0.269841269)

    oivchallenge_evaluator.clear()
    self.assertFalse(oivchallenge_evaluator._image_ids)


class PascalEvaluationTest(tf.test.TestCase):

  def test_returns_correct_metric_values_on_boxes(self):
    categories = [{'id': 1, 'name': 'cat'},
                  {'id': 2, 'name': 'dog'},
                  {'id': 3, 'name': 'elephant'}]
    #  Add groundtruth
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key1,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels1,
         standard_fields.InputDataFields.groundtruth_difficult:
         np.array([], dtype=bool)})
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([1, 1, 3], dtype=int)
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key2,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes2,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels2,
         standard_fields.InputDataFields.groundtruth_difficult:
         groundtruth_is_difficult_list2})
    image_key3 = 'img3'
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([2], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key3,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes3,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels3})

    # Add detections
    image_key = 'img2'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        dtype=float)
    detected_class_labels = np.array([1, 1, 3], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    pascal_evaluator.add_single_detected_image_info(
        image_key,
        {standard_fields.DetectionResultFields.detection_boxes: detected_boxes,
         standard_fields.DetectionResultFields.detection_scores:
         detected_scores,
         standard_fields.DetectionResultFields.detection_classes:
         detected_class_labels})

    metrics = pascal_evaluator.evaluate()
    self.assertAlmostEqual(
        metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/dog'], 0.0)
    self.assertAlmostEqual(
        metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/elephant'], 0.0)
    self.assertAlmostEqual(
        metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/cat'], 0.16666666)
    self.assertAlmostEqual(metrics['PascalBoxes_Precision/mAP@0.5IOU'],
                           0.05555555)
    pascal_evaluator.clear()
    self.assertFalse(pascal_evaluator._image_ids)

  def test_returns_correct_metric_values_on_masks(self):
    categories = [{'id': 1, 'name': 'cat'},
                  {'id': 2, 'name': 'dog'},
                  {'id': 3, 'name': 'elephant'}]
    #  Add groundtruth
    pascal_evaluator = (
        object_detection_evaluation.PascalInstanceSegmentationEvaluator(
            categories))
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    groundtruth_masks_1_0 = np.array([[1, 0, 0, 0],
                                      [1, 0, 0, 0],
                                      [1, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks_1_1 = np.array([[0, 0, 1, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 1, 0]], dtype=np.uint8)
    groundtruth_masks_1_2 = np.array([[0, 1, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 1, 0, 0]], dtype=np.uint8)
    groundtruth_masks1 = np.stack(
        [groundtruth_masks_1_0, groundtruth_masks_1_1, groundtruth_masks_1_2],
        axis=0)

    pascal_evaluator.add_single_ground_truth_image_info(
        image_key1, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes1,
            standard_fields.InputDataFields.groundtruth_instance_masks:
                groundtruth_masks1,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels1,
            standard_fields.InputDataFields.groundtruth_difficult:
                np.array([], dtype=bool)
        })
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([1, 1, 3], dtype=int)
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    groundtruth_masks_2_0 = np.array([[1, 1, 1, 1],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks_2_1 = np.array([[0, 0, 0, 0],
                                      [1, 1, 1, 1],
                                      [0, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks_2_2 = np.array([[0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [1, 1, 1, 1]], dtype=np.uint8)
    groundtruth_masks2 = np.stack(
        [groundtruth_masks_2_0, groundtruth_masks_2_1, groundtruth_masks_2_2],
        axis=0)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key2, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes2,
            standard_fields.InputDataFields.groundtruth_instance_masks:
                groundtruth_masks2,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels2,
            standard_fields.InputDataFields.groundtruth_difficult:
                groundtruth_is_difficult_list2
        })
    image_key3 = 'img3'
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([2], dtype=int)
    groundtruth_masks_3_0 = np.array([[1, 1, 1, 1],
                                      [1, 1, 1, 1],
                                      [1, 1, 1, 1]], dtype=np.uint8)
    groundtruth_masks3 = np.stack([groundtruth_masks_3_0], axis=0)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key3, {
            standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes3,
            standard_fields.InputDataFields.groundtruth_instance_masks:
                groundtruth_masks3,
            standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels3
        })

    # Add detections
    image_key = 'img2'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        dtype=float)
    detected_class_labels = np.array([1, 1, 3], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    detected_masks_0 = np.array([[1, 1, 1, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_1 = np.array([[1, 0, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_2 = np.array([[0, 1, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 0, 0]], dtype=np.uint8)
    detected_masks = np.stack(
        [detected_masks_0, detected_masks_1, detected_masks_2], axis=0)

    pascal_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                detected_boxes,
            standard_fields.DetectionResultFields.detection_masks:
                detected_masks,
            standard_fields.DetectionResultFields.detection_scores:
                detected_scores,
            standard_fields.DetectionResultFields.detection_classes:
                detected_class_labels
        })

    metrics = pascal_evaluator.evaluate()

    self.assertAlmostEqual(
        metrics['PascalMasks_PerformanceByCategory/AP@0.5IOU/dog'], 0.0)
    self.assertAlmostEqual(
        metrics['PascalMasks_PerformanceByCategory/AP@0.5IOU/elephant'], 0.0)
    self.assertAlmostEqual(
        metrics['PascalMasks_PerformanceByCategory/AP@0.5IOU/cat'], 0.16666666)
    self.assertAlmostEqual(metrics['PascalMasks_Precision/mAP@0.5IOU'],
                           0.05555555)
    pascal_evaluator.clear()
    self.assertFalse(pascal_evaluator._image_ids)

  def test_value_error_on_duplicate_images(self):
    categories = [{'id': 1, 'name': 'cat'},
                  {'id': 2, 'name': 'dog'},
                  {'id': 3, 'name': 'elephant'}]
    #  Add groundtruth
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key1,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels1})
    with self.assertRaises(ValueError):
      pascal_evaluator.add_single_ground_truth_image_info(
          image_key1,
          {standard_fields.InputDataFields.groundtruth_boxes:
           groundtruth_boxes1,
           standard_fields.InputDataFields.groundtruth_classes:
           groundtruth_class_labels1})


class WeightedPascalEvaluationTest(tf.test.TestCase):

  def setUp(self):
    self.categories = [{'id': 1, 'name': 'cat'},
                       {'id': 2, 'name': 'dog'},
                       {'id': 3, 'name': 'elephant'}]

  def create_and_add_common_ground_truth(self):
    #  Add groundtruth
    self.wp_eval = (
        object_detection_evaluation.WeightedPascalDetectionEvaluator(
            self.categories))

    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    self.wp_eval.add_single_ground_truth_image_info(
        image_key1,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels1})
    # add 'img2' separately
    image_key3 = 'img3'
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([2], dtype=int)
    self.wp_eval.add_single_ground_truth_image_info(
        image_key3,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes3,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels3})

  def add_common_detected(self):
    image_key = 'img2'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        dtype=float)
    detected_class_labels = np.array([1, 1, 3], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    self.wp_eval.add_single_detected_image_info(
        image_key,
        {standard_fields.DetectionResultFields.detection_boxes: detected_boxes,
         standard_fields.DetectionResultFields.detection_scores:
         detected_scores,
         standard_fields.DetectionResultFields.detection_classes:
         detected_class_labels})

  def test_returns_correct_metric_values(self):
    self.create_and_add_common_ground_truth()
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([1, 1, 3], dtype=int)
    self.wp_eval.add_single_ground_truth_image_info(
        image_key2,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes2,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels2
        })
    self.add_common_detected()

    metrics = self.wp_eval.evaluate()
    self.assertAlmostEqual(
        metrics[self.wp_eval._metric_prefix +
                'PerformanceByCategory/AP@0.5IOU/dog'], 0.0)
    self.assertAlmostEqual(
        metrics[self.wp_eval._metric_prefix +
                'PerformanceByCategory/AP@0.5IOU/elephant'], 0.0)
    self.assertAlmostEqual(
        metrics[self.wp_eval._metric_prefix +
                'PerformanceByCategory/AP@0.5IOU/cat'], 0.5 / 4)
    self.assertAlmostEqual(metrics[self.wp_eval._metric_prefix +
                                   'Precision/mAP@0.5IOU'],
                           1. / (4 + 1 + 2) / 3)
    self.wp_eval.clear()
    self.assertFalse(self.wp_eval._image_ids)

  def test_returns_correct_metric_values_with_difficult_list(self):
    self.create_and_add_common_ground_truth()
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([1, 1, 3], dtype=int)
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    self.wp_eval.add_single_ground_truth_image_info(
        image_key2,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes2,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels2,
         standard_fields.InputDataFields.groundtruth_difficult:
         groundtruth_is_difficult_list2
        })
    self.add_common_detected()

    metrics = self.wp_eval.evaluate()
    self.assertAlmostEqual(
        metrics[self.wp_eval._metric_prefix +
                'PerformanceByCategory/AP@0.5IOU/dog'], 0.0)
    self.assertAlmostEqual(
        metrics[self.wp_eval._metric_prefix +
                'PerformanceByCategory/AP@0.5IOU/elephant'], 0.0)
    self.assertAlmostEqual(
        metrics[self.wp_eval._metric_prefix +
                'PerformanceByCategory/AP@0.5IOU/cat'], 0.5 / 3)
    self.assertAlmostEqual(metrics[self.wp_eval._metric_prefix +
                                   'Precision/mAP@0.5IOU'],
                           1. / (3 + 1 + 2) / 3)
    self.wp_eval.clear()
    self.assertFalse(self.wp_eval._image_ids)

  def test_value_error_on_duplicate_images(self):
    #  Add groundtruth
    self.wp_eval = (
        object_detection_evaluation.WeightedPascalDetectionEvaluator(
            self.categories))
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    self.wp_eval.add_single_ground_truth_image_info(
        image_key1,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels1})
    with self.assertRaises(ValueError):
      self.wp_eval.add_single_ground_truth_image_info(
          image_key1,
          {standard_fields.InputDataFields.groundtruth_boxes:
           groundtruth_boxes1,
           standard_fields.InputDataFields.groundtruth_classes:
           groundtruth_class_labels1})


class ObjectDetectionEvaluationTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 3
    self.od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
        num_groundtruth_classes)

    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([0, 2, 0], dtype=int)
    self.od_eval.add_single_ground_truth_image_info(
        image_key1, groundtruth_boxes1, groundtruth_class_labels1)
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([0, 0, 2], dtype=int)
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    groundtruth_is_group_of_list2 = np.array([False, False, True], dtype=bool)
    self.od_eval.add_single_ground_truth_image_info(
        image_key2, groundtruth_boxes2, groundtruth_class_labels2,
        groundtruth_is_difficult_list2, groundtruth_is_group_of_list2)

    image_key3 = 'img3'
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([1], dtype=int)
    self.od_eval.add_single_ground_truth_image_info(
        image_key3, groundtruth_boxes3, groundtruth_class_labels3)

    image_key = 'img2'
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        dtype=float)
    detected_class_labels = np.array([0, 0, 2], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    self.od_eval.add_single_detected_image_info(
        image_key, detected_boxes, detected_scores, detected_class_labels)

  def test_value_error_on_zero_classes(self):
    with self.assertRaises(ValueError):
      object_detection_evaluation.ObjectDetectionEvaluation(
          num_groundtruth_classes=0)

  def test_add_single_ground_truth_image_info(self):
    expected_num_gt_instances_per_class = np.array([3, 1, 1], dtype=int)
    expected_num_gt_imgs_per_class = np.array([2, 1, 2], dtype=int)
    self.assertTrue(np.array_equal(expected_num_gt_instances_per_class,
                                   self.od_eval.num_gt_instances_per_class))
    self.assertTrue(np.array_equal(expected_num_gt_imgs_per_class,
                                   self.od_eval.num_gt_imgs_per_class))
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    self.assertTrue(np.allclose(self.od_eval.groundtruth_boxes['img2'],
                                groundtruth_boxes2))
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    self.assertTrue(np.allclose(
        self.od_eval.groundtruth_is_difficult_list['img2'],
        groundtruth_is_difficult_list2))
    groundtruth_is_group_of_list2 = np.array([False, False, True], dtype=bool)
    self.assertTrue(
        np.allclose(self.od_eval.groundtruth_is_group_of_list['img2'],
                    groundtruth_is_group_of_list2))

    groundtruth_class_labels1 = np.array([0, 2, 0], dtype=int)
    self.assertTrue(np.array_equal(self.od_eval.groundtruth_class_labels[
        'img1'], groundtruth_class_labels1))

  def test_add_single_detected_image_info(self):
    expected_scores_per_class = [[np.array([0.8, 0.7], dtype=float)], [],
                                 [np.array([0.9], dtype=float)]]
    expected_tp_fp_labels_per_class = [[np.array([0, 1], dtype=bool)], [],
                                       [np.array([0], dtype=bool)]]
    expected_num_images_correctly_detected_per_class = np.array([0, 0, 0],
                                                                dtype=int)
    for i in range(self.od_eval.num_class):
      for j in range(len(expected_scores_per_class[i])):
        self.assertTrue(np.allclose(expected_scores_per_class[i][j],
                                    self.od_eval.scores_per_class[i][j]))
        self.assertTrue(np.array_equal(expected_tp_fp_labels_per_class[i][
            j], self.od_eval.tp_fp_labels_per_class[i][j]))
    self.assertTrue(np.array_equal(
        expected_num_images_correctly_detected_per_class,
        self.od_eval.num_images_correctly_detected_per_class))

  def test_evaluate(self):
    (average_precision_per_class, mean_ap, precisions_per_class,
     recalls_per_class, corloc_per_class,
     mean_corloc) = self.od_eval.evaluate()
    expected_precisions_per_class = [np.array([0, 0.5], dtype=float),
                                     np.array([], dtype=float),
                                     np.array([0], dtype=float)]
    expected_recalls_per_class = [
        np.array([0, 1. / 3.], dtype=float), np.array([], dtype=float),
        np.array([0], dtype=float)
    ]
    expected_average_precision_per_class = np.array([1. / 6., 0, 0],
                                                    dtype=float)
    expected_corloc_per_class = np.array([0, np.divide(0, 0), 0], dtype=float)
    expected_mean_ap = 1. / 18
    expected_mean_corloc = 0.0
    for i in range(self.od_eval.num_class):
      self.assertTrue(np.allclose(expected_precisions_per_class[i],
                                  precisions_per_class[i]))
      self.assertTrue(np.allclose(expected_recalls_per_class[i],
                                  recalls_per_class[i]))
    self.assertTrue(np.allclose(expected_average_precision_per_class,
                                average_precision_per_class))
    self.assertTrue(np.allclose(expected_corloc_per_class, corloc_per_class))
    self.assertAlmostEqual(expected_mean_ap, mean_ap)
    self.assertAlmostEqual(expected_mean_corloc, mean_corloc)


if __name__ == '__main__':
  tf.test.main()
