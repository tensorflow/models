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

from object_detection.utils import object_detection_evaluation


class ObjectDetectionEvaluationTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 3
    self.od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
        num_groundtruth_classes)

    image_key1 = "img1"
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([0, 2, 0], dtype=int)
    self.od_eval.add_single_ground_truth_image_info(
        image_key1, groundtruth_boxes1, groundtruth_class_labels1)
    image_key2 = "img2"
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([0, 0, 2], dtype=int)
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    self.od_eval.add_single_ground_truth_image_info(
        image_key2, groundtruth_boxes2, groundtruth_class_labels2,
        groundtruth_is_difficult_list2)
    image_key3 = "img3"
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([1], dtype=int)
    self.od_eval.add_single_ground_truth_image_info(
        image_key3, groundtruth_boxes3, groundtruth_class_labels3)

    image_key = "img2"
    detected_boxes = np.array(
        [[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        dtype=float)
    detected_class_labels = np.array([0, 0, 2], dtype=int)
    detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    self.od_eval.add_single_detected_image_info(
        image_key, detected_boxes, detected_scores, detected_class_labels)

  def test_add_single_ground_truth_image_info(self):
    expected_num_gt_instances_per_class = np.array([3, 1, 2], dtype=int)
    expected_num_gt_imgs_per_class = np.array([2, 1, 2], dtype=int)
    self.assertTrue(np.array_equal(expected_num_gt_instances_per_class,
                                   self.od_eval.num_gt_instances_per_class))
    self.assertTrue(np.array_equal(expected_num_gt_imgs_per_class,
                                   self.od_eval.num_gt_imgs_per_class))
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    self.assertTrue(np.allclose(self.od_eval.groundtruth_boxes["img2"],
                                groundtruth_boxes2))
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    self.assertTrue(np.allclose(
        self.od_eval.groundtruth_is_difficult_list["img2"],
        groundtruth_is_difficult_list2))
    groundtruth_class_labels1 = np.array([0, 2, 0], dtype=int)
    self.assertTrue(np.array_equal(self.od_eval.groundtruth_class_labels[
        "img1"], groundtruth_class_labels1))

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


if __name__ == "__main__":
  tf.test.main()
