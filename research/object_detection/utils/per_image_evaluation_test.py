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

"""Tests for object_detection.utils.per_image_evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from object_detection.utils import per_image_evaluation


class SingleClassTpFpWithDifficultBoxesTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
        nms_max_output_boxes)

    self.detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                   dtype=float)
    self.detected_scores = np.array([0.6, 0.8, 0.5], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_1 = np.array([[1, 0, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_2 = np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 0, 0]], dtype=np.uint8)
    self.detected_masks = np.stack(
        [detected_masks_0, detected_masks_1, detected_masks_2], axis=0)
    self.groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 10, 10]],
                                      dtype=float)
    groundtruth_masks_0 = np.array([[1, 1, 0, 0],
                                    [1, 1, 0, 0],
                                    [0, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks_1 = np.array([[0, 0, 0, 1],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 1]], dtype=np.uint8)
    self.groundtruth_masks = np.stack(
        [groundtruth_masks_0, groundtruth_masks_1], axis=0)

  def test_match_to_gt_box_0(self):
    groundtruth_groundtruth_is_difficult_list = np.array([False, True],
                                                         dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, False], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, True, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_match_to_gt_mask_0(self):
    groundtruth_groundtruth_is_difficult_list = np.array([False, True],
                                                         dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, False], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=self.groundtruth_masks)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([True, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_match_to_gt_box_1(self):
    groundtruth_groundtruth_is_difficult_list = np.array([True, False],
                                                         dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, False], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = np.array([0.8, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_match_to_gt_mask_1(self):
    groundtruth_groundtruth_is_difficult_list = np.array([True, False],
                                                         dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, False], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=self.groundtruth_masks)
    expected_scores = np.array([0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class SingleClassTpFpWithGroupOfBoxesTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
        nms_max_output_boxes)

    self.detected_boxes = np.array(
        [[0, 0, 1, 1], [0, 0, 2, 1], [0, 0, 3, 1]], dtype=float)
    self.detected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_1 = np.array([[1, 0, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_2 = np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 0, 0]], dtype=np.uint8)
    self.detected_masks = np.stack(
        [detected_masks_0, detected_masks_1, detected_masks_2], axis=0)

    self.groundtruth_boxes = np.array(
        [[0, 0, 1, 1], [0, 0, 5, 5], [10, 10, 20, 20]], dtype=float)
    groundtruth_masks_0 = np.array([[1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [1, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks_1 = np.array([[0, 0, 1, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 1, 0]], dtype=np.uint8)
    groundtruth_masks_2 = np.array([[0, 1, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 1, 0, 0]], dtype=np.uint8)
    self.groundtruth_masks = np.stack(
        [groundtruth_masks_0, groundtruth_masks_1, groundtruth_masks_2], axis=0)

  def test_match_to_non_group_of_and_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, True, True], dtype=bool)
    expected_scores = np.array([0.8], dtype=float)
    expected_tp_fp_labels = np.array([True], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_match_to_non_group_of_and_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, True, True], dtype=bool)
    expected_scores = np.array([0.6], dtype=float)
    expected_tp_fp_labels = np.array([True], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=self.groundtruth_masks)

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_match_two_to_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [True, False, True], dtype=bool)
    expected_scores = np.array([0.5], dtype=float)
    expected_tp_fp_labels = np.array([False], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_match_two_to_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [True, False, True], dtype=bool)
    expected_scores = np.array([0.8], dtype=float)
    expected_tp_fp_labels = np.array([True], dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=self.groundtruth_masks)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class SingleClassTpFpWithGroupOfBoxesTestWeighted(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.group_of_weight = 0.5
    self.eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
        nms_max_output_boxes, self.group_of_weight)

    self.detected_boxes = np.array(
        [[0, 0, 1, 1], [0, 0, 2, 1], [0, 0, 3, 1]], dtype=float)
    self.detected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    detected_masks_0 = np.array(
        [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_1 = np.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_2 = np.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    self.detected_masks = np.stack(
        [detected_masks_0, detected_masks_1, detected_masks_2], axis=0)

    self.groundtruth_boxes = np.array(
        [[0, 0, 1, 1], [0, 0, 5, 5], [10, 10, 20, 20]], dtype=float)
    groundtruth_masks_0 = np.array(
        [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks_1 = np.array(
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)
    groundtruth_masks_2 = np.array(
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
    self.groundtruth_masks = np.stack(
        [groundtruth_masks_0, groundtruth_masks_1, groundtruth_masks_2], axis=0)

  def test_match_to_non_group_of_and_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, True, True], dtype=bool)
    expected_scores = np.array([0.8, 0.6], dtype=float)
    expected_tp_fp_labels = np.array([1.0, self.group_of_weight], dtype=float)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_match_to_non_group_of_and_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, True, True], dtype=bool)
    expected_scores = np.array([0.6, 0.8, 0.5], dtype=float)
    expected_tp_fp_labels = np.array(
        [1.0, self.group_of_weight, self.group_of_weight], dtype=float)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=self.groundtruth_masks)

    tf.logging.info(
        "test_mask_match_to_non_group_of_and_group_of_box {} {}".format(
            tp_fp_labels, expected_tp_fp_labels))

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_match_two_to_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [True, False, True], dtype=bool)
    expected_scores = np.array([0.5, 0.8], dtype=float)
    expected_tp_fp_labels = np.array([0.0, self.group_of_weight], dtype=float)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)

    tf.logging.info("test_match_two_to_group_of_box {} {}".format(
        tp_fp_labels, expected_tp_fp_labels))

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_match_two_to_group_of_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array(
        [False, False, False], dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [True, False, True], dtype=bool)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array(
        [1.0, self.group_of_weight, self.group_of_weight], dtype=float)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=self.groundtruth_masks)

    tf.logging.info("test_mask_match_two_to_group_of_box {} {}".format(
        tp_fp_labels, expected_tp_fp_labels))

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class SingleClassTpFpNoDifficultBoxesTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold_high_iou = 0.5
    matching_iou_threshold_low_iou = 0.1
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.eval_high_iou = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold_high_iou,
        nms_iou_threshold, nms_max_output_boxes)

    self.eval_low_iou = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold_low_iou,
        nms_iou_threshold, nms_max_output_boxes)

    self.detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                   dtype=float)
    self.detected_scores = np.array([0.6, 0.8, 0.5], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_1 = np.array([[1, 0, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.uint8)
    detected_masks_2 = np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 0, 0]], dtype=np.uint8)
    self.detected_masks = np.stack(
        [detected_masks_0, detected_masks_1, detected_masks_2], axis=0)

  def test_no_true_positives(self):
    groundtruth_boxes = np.array([[100, 100, 105, 105]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False], dtype=bool)
    scores, tp_fp_labels = self.eval_high_iou._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_no_true_positives(self):
    groundtruth_boxes = np.array([[100, 100, 105, 105]], dtype=float)
    groundtruth_masks_0 = np.array([[1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1]], dtype=np.uint8)
    groundtruth_masks = np.stack([groundtruth_masks_0], axis=0)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False], dtype=bool)
    scores, tp_fp_labels = self.eval_high_iou._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=groundtruth_masks)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_one_true_positives_with_large_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False], dtype=bool)
    scores, tp_fp_labels = self.eval_high_iou._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, True, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_one_true_positives_with_large_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_masks_0 = np.array([[1, 0, 0, 0],
                                    [1, 1, 0, 0],
                                    [0, 0, 0, 0]], dtype=np.uint8)
    groundtruth_masks = np.stack([groundtruth_masks_0], axis=0)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False], dtype=bool)
    scores, tp_fp_labels = self.eval_high_iou._compute_tp_fp_for_single_class(
        self.detected_boxes,
        self.detected_scores,
        groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=self.detected_masks,
        groundtruth_masks=groundtruth_masks)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([True, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_one_true_positives_with_very_small_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False], dtype=bool)
    scores, tp_fp_labels = self.eval_low_iou._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([True, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_two_true_positives_with_large_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3.5, 3.5]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, False], dtype=bool)
    scores, tp_fp_labels = self.eval_high_iou._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, True, True], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class SingleClassTpFpEmptyMaskAndBoxesTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold_iou = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold_iou, nms_iou_threshold,
        nms_max_output_boxes)

  def test_mask_tp_and_ignore(self):
    # GT: one box with mask, one without
    # Det: One mask matches gt1, one matches box gt2 and is ignored
    groundtruth_boxes = np.array([[0, 0, 2, 3], [0, 0, 2, 2]], dtype=float)
    groundtruth_mask_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_mask_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_masks = np.stack([groundtruth_mask_0, groundtruth_mask_1],
                                 axis=0)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False, False],
                                                        dtype=bool)

    detected_boxes = np.array([[0, 0, 2, 3], [0, 0, 2, 2]], dtype=float)
    detected_scores = np.array([0.6, 0.8], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks_1 = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks = np.stack([detected_masks_0, detected_masks_1], axis=0)

    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        detected_boxes, detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list, detected_masks,
        groundtruth_masks)
    expected_scores = np.array([0.6], dtype=float)
    expected_tp_fp_labels = np.array([True], dtype=bool)

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_mask_one_tp_one_fp(self):
    # GT: one box with mask, one without
    # Det: one mask matches gt1, one is fp (box does not match)
    groundtruth_boxes = np.array([[0, 0, 2, 3], [2, 2, 4, 4]], dtype=float)
    groundtruth_mask_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_mask_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_masks = np.stack([groundtruth_mask_0, groundtruth_mask_1],
                                 axis=0)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False, False],
                                                        dtype=bool)

    detected_boxes = np.array([[0, 0, 2, 3], [0, 0, 2, 2]], dtype=float)
    detected_scores = np.array([0.6, 0.8], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks_1 = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks = np.stack([detected_masks_0, detected_masks_1], axis=0)

    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        detected_boxes,
        detected_scores,
        groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)
    expected_scores = np.array([0.8, 0.6], dtype=float)
    expected_tp_fp_labels = np.array([False, True], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_two_mask_one_gt_one_ignore(self):
    # GT: one box with mask, one without.
    # Det: two mask matches same gt, one is tp, one is passed down to box match
    # and ignored.
    groundtruth_boxes = np.array([[0, 0, 2, 3], [0, 0, 2, 3]], dtype=float)
    groundtruth_mask_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_mask_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_masks = np.stack([groundtruth_mask_0, groundtruth_mask_1],
                                 axis=0)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False, False],
                                                        dtype=bool)

    detected_boxes = np.array([[0, 0, 2, 3], [0, 0, 2, 3]], dtype=float)
    detected_scores = np.array([0.6, 0.8], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks = np.stack([detected_masks_0, detected_masks_1], axis=0)

    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        detected_boxes,
        detected_scores,
        groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)
    expected_scores = np.array([0.8], dtype=float)
    expected_tp_fp_labels = np.array([True], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_two_mask_one_gt_one_fp(self):
    # GT: one box with mask, one without.
    # Det: two mask matches same gt, one is tp, one is passed down to box match
    # and is fp.
    groundtruth_boxes = np.array([[0, 0, 2, 3], [2, 3, 4, 6]], dtype=float)
    groundtruth_mask_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_mask_1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                  dtype=np.uint8)
    groundtruth_masks = np.stack([groundtruth_mask_0, groundtruth_mask_1],
                                 axis=0)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
    groundtruth_groundtruth_is_group_of_list = np.array([False, False],
                                                        dtype=bool)

    detected_boxes = np.array([[0, 0, 2, 3], [0, 0, 2, 3]], dtype=float)
    detected_scores = np.array([0.6, 0.8], dtype=float)
    detected_masks_0 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                dtype=np.uint8)
    detected_masks = np.stack([detected_masks_0, detected_masks_1], axis=0)

    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        detected_boxes,
        detected_scores,
        groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list,
        detected_masks=detected_masks,
        groundtruth_masks=groundtruth_masks)
    expected_scores = np.array([0.8, 0.6], dtype=float)
    expected_tp_fp_labels = np.array([True, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class MultiClassesTpFpTest(tf.test.TestCase):

  def test_tp_fp(self):
    num_groundtruth_classes = 3
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    eval1 = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes,
                                                    matching_iou_threshold,
                                                    nms_iou_threshold,
                                                    nms_max_output_boxes)
    detected_boxes = np.array([[0, 0, 1, 1], [10, 10, 5, 5], [0, 0, 2, 2],
                               [5, 10, 10, 5], [10, 5, 5, 10], [0, 0, 3, 3]],
                              dtype=float)
    detected_scores = np.array([0.8, 0.1, 0.8, 0.9, 0.7, 0.8], dtype=float)
    detected_class_labels = np.array([0, 1, 1, 2, 0, 2], dtype=int)
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3.5, 3.5]], dtype=float)
    groundtruth_class_labels = np.array([0, 2], dtype=int)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=float)
    groundtruth_groundtruth_is_group_of_list = np.array(
        [False, False], dtype=bool)
    scores, tp_fp_labels, _ = eval1.compute_object_detection_metrics(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels,
        groundtruth_groundtruth_is_difficult_list,
        groundtruth_groundtruth_is_group_of_list)
    expected_scores = [np.array([0.8], dtype=float)] * 3
    expected_tp_fp_labels = [np.array([True]), np.array([False]), np.array([True
                                                                           ])]
    for i in range(len(expected_scores)):
      self.assertTrue(np.allclose(expected_scores[i], scores[i]))
      self.assertTrue(np.array_equal(expected_tp_fp_labels[i], tp_fp_labels[i]))


class CorLocTest(tf.test.TestCase):

  def test_compute_corloc_with_normal_iou_threshold(self):
    num_groundtruth_classes = 3
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    eval1 = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes,
                                                    matching_iou_threshold,
                                                    nms_iou_threshold,
                                                    nms_max_output_boxes)
    detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3],
                               [0, 0, 5, 5]], dtype=float)
    detected_scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=float)
    detected_class_labels = np.array([0, 1, 0, 2], dtype=int)
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3, 3], [0, 0, 6, 6]],
                                 dtype=float)
    groundtruth_class_labels = np.array([0, 0, 2], dtype=int)

    is_class_correctly_detected_in_image = eval1._compute_cor_loc(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels)
    expected_result = np.array([1, 0, 1], dtype=int)
    self.assertTrue(np.array_equal(expected_result,
                                   is_class_correctly_detected_in_image))

  def test_compute_corloc_with_very_large_iou_threshold(self):
    num_groundtruth_classes = 3
    matching_iou_threshold = 0.9
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    eval1 = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes,
                                                    matching_iou_threshold,
                                                    nms_iou_threshold,
                                                    nms_max_output_boxes)
    detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3],
                               [0, 0, 5, 5]], dtype=float)
    detected_scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=float)
    detected_class_labels = np.array([0, 1, 0, 2], dtype=int)
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3, 3], [0, 0, 6, 6]],
                                 dtype=float)
    groundtruth_class_labels = np.array([0, 0, 2], dtype=int)

    is_class_correctly_detected_in_image = eval1._compute_cor_loc(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels)
    expected_result = np.array([1, 0, 0], dtype=int)
    self.assertTrue(np.array_equal(expected_result,
                                   is_class_correctly_detected_in_image))


if __name__ == "__main__":
  tf.test.main()
