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
"""Tests for object_detection.utils.per_image_vrd_evaluation."""
import numpy as np
import tensorflow as tf

from object_detection.utils import per_image_vrd_evaluation


class SingleClassPerImageVrdEvaluationTest(tf.test.TestCase):

  def setUp(self):
    matching_iou_threshold = 0.5
    self.eval = per_image_vrd_evaluation.PerImageVRDEvaluation(
        matching_iou_threshold)
    box_data_type = np.dtype([('subject', 'f4', (4,)), ('object', 'f4', (4,))])

    self.detected_box_tuples = np.array(
        [([0, 0, 1.1, 1], [1, 1, 2, 2]), ([0, 0, 1, 1], [1, 1, 2, 2]),
         ([1, 1, 2, 2], [0, 0, 1.1, 1])],
        dtype=box_data_type)
    self.detected_scores = np.array([0.8, 0.2, 0.1], dtype=float)
    self.groundtruth_box_tuples = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2])], dtype=box_data_type)

  def test_tp_fp_eval(self):
    tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_box_tuples, self.groundtruth_box_tuples)
    expected_tp_fp_labels = np.array([True, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_tp_fp_eval_empty_gt(self):
    box_data_type = np.dtype([('subject', 'f4', (4,)), ('object', 'f4', (4,))])

    tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_box_tuples, np.array([], dtype=box_data_type))
    expected_tp_fp_labels = np.array([False, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class MultiClassPerImageVrdEvaluationTest(tf.test.TestCase):

  def setUp(self):
    matching_iou_threshold = 0.5
    self.eval = per_image_vrd_evaluation.PerImageVRDEvaluation(
        matching_iou_threshold)
    box_data_type = np.dtype([('subject', 'f4', (4,)), ('object', 'f4', (4,))])
    label_data_type = np.dtype([('subject', 'i4'), ('object', 'i4'),
                                ('relation', 'i4')])

    self.detected_box_tuples = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2]), ([0, 0, 1.1, 1], [1, 1, 2, 2]),
         ([1, 1, 2, 2], [0, 0, 1.1, 1]), ([0, 0, 1, 1], [3, 4, 5, 6])],
        dtype=box_data_type)
    self.detected_class_tuples = np.array(
        [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 4, 5)], dtype=label_data_type)
    self.detected_scores = np.array([0.2, 0.8, 0.1, 0.5], dtype=float)

    self.groundtruth_box_tuples = np.array(
        [([0, 0, 1, 1], [1, 1, 2, 2]), ([1, 1, 2, 2], [0, 0, 1.1, 1]),
         ([0, 0, 1, 1], [3, 4, 5, 5.5])],
        dtype=box_data_type)
    self.groundtruth_class_tuples = np.array(
        [(1, 2, 3), (1, 7, 3), (1, 4, 5)], dtype=label_data_type)

  def test_tp_fp_eval(self):
    scores, tp_fp_labels, mapping = self.eval.compute_detection_tp_fp(
        self.detected_box_tuples, self.detected_scores,
        self.detected_class_tuples, self.groundtruth_box_tuples,
        self.groundtruth_class_tuples)

    expected_scores = np.array([0.8, 0.5, 0.2, 0.1], dtype=float)
    expected_tp_fp_labels = np.array([True, True, False, False], dtype=bool)
    expected_mapping = np.array([1, 3, 0, 2])

    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))
    self.assertTrue(np.allclose(expected_mapping, mapping))


if __name__ == '__main__':
  tf.test.main()
