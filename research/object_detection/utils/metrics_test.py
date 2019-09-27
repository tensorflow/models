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
"""Tests for object_detection.metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.utils import metrics


class MetricsTest(tf.test.TestCase):

  def test_compute_cor_loc(self):
    num_gt_imgs_per_class = np.array([100, 1, 5, 1, 1], dtype=int)
    num_images_correctly_detected_per_class = np.array(
        [10, 0, 1, 0, 0], dtype=int)
    corloc = metrics.compute_cor_loc(num_gt_imgs_per_class,
                                     num_images_correctly_detected_per_class)
    expected_corloc = np.array([0.1, 0, 0.2, 0, 0], dtype=float)
    self.assertTrue(np.allclose(corloc, expected_corloc))

  def test_compute_cor_loc_nans(self):
    num_gt_imgs_per_class = np.array([100, 0, 0, 1, 1], dtype=int)
    num_images_correctly_detected_per_class = np.array(
        [10, 0, 1, 0, 0], dtype=int)
    corloc = metrics.compute_cor_loc(num_gt_imgs_per_class,
                                     num_images_correctly_detected_per_class)
    expected_corloc = np.array([0.1, np.nan, np.nan, 0, 0], dtype=float)
    self.assertAllClose(corloc, expected_corloc)

  def test_compute_precision_recall(self):
    num_gt = 10
    scores = np.array([0.4, 0.3, 0.6, 0.2, 0.7, 0.1], dtype=float)
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    labels_float_type = np.array([0, 1, 1, 0, 0, 1], dtype=float)
    accumulated_tp_count = np.array([0, 1, 1, 2, 2, 3], dtype=float)
    expected_precision = accumulated_tp_count / np.array([1, 2, 3, 4, 5, 6])
    expected_recall = accumulated_tp_count / num_gt

    precision, recall = metrics.compute_precision_recall(scores, labels, num_gt)
    precision_float_type, recall_float_type = metrics.compute_precision_recall(
        scores, labels_float_type, num_gt)

    self.assertAllClose(precision, expected_precision)
    self.assertAllClose(recall, expected_recall)
    self.assertAllClose(precision_float_type, expected_precision)
    self.assertAllClose(recall_float_type, expected_recall)

  def test_compute_precision_recall_float(self):
    num_gt = 10
    scores = np.array([0.4, 0.3, 0.6, 0.2, 0.7, 0.1], dtype=float)
    labels_float = np.array([0, 1, 1, 0.5, 0, 1], dtype=float)
    expected_precision = np.array(
        [0., 0.5, 0.33333333, 0.5, 0.55555556, 0.63636364], dtype=float)
    expected_recall = np.array([0., 0.1, 0.1, 0.2, 0.25, 0.35], dtype=float)
    precision, recall = metrics.compute_precision_recall(
        scores, labels_float, num_gt)
    self.assertAllClose(precision, expected_precision)
    self.assertAllClose(recall, expected_recall)

  def test_compute_average_precision(self):
    precision = np.array([0.8, 0.76, 0.9, 0.65, 0.7, 0.5, 0.55, 0], dtype=float)
    recall = np.array([0.3, 0.3, 0.4, 0.4, 0.45, 0.45, 0.5, 0.5], dtype=float)
    processed_precision = np.array(
        [0.9, 0.9, 0.9, 0.7, 0.7, 0.55, 0.55, 0], dtype=float)
    recall_interval = np.array([0.3, 0, 0.1, 0, 0.05, 0, 0.05, 0], dtype=float)
    expected_mean_ap = np.sum(recall_interval * processed_precision)
    mean_ap = metrics.compute_average_precision(precision, recall)
    self.assertAlmostEqual(expected_mean_ap, mean_ap)

  def test_compute_precision_recall_and_ap_no_groundtruth(self):
    num_gt = 0
    scores = np.array([0.4, 0.3, 0.6, 0.2, 0.7, 0.1], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 0], dtype=bool)
    expected_precision = None
    expected_recall = None
    precision, recall = metrics.compute_precision_recall(scores, labels, num_gt)
    self.assertEquals(precision, expected_precision)
    self.assertEquals(recall, expected_recall)
    ap = metrics.compute_average_precision(precision, recall)
    self.assertTrue(np.isnan(ap))

  def test_compute_recall_at_k(self):
    num_gt = 4
    tp_fp = [
        np.array([1, 0, 0], dtype=float),
        np.array([0, 1], dtype=float),
        np.array([0, 0, 0, 0, 0], dtype=float)
    ]
    tp_fp_bool = [
        np.array([True, False, False], dtype=bool),
        np.array([False, True], dtype=float),
        np.array([False, False, False, False, False], dtype=float)
    ]

    recall_1 = metrics.compute_recall_at_k(tp_fp, num_gt, 1)
    recall_3 = metrics.compute_recall_at_k(tp_fp, num_gt, 3)
    recall_5 = metrics.compute_recall_at_k(tp_fp, num_gt, 5)

    recall_3_bool = metrics.compute_recall_at_k(tp_fp_bool, num_gt, 3)

    self.assertAlmostEqual(recall_1, 0.25)
    self.assertAlmostEqual(recall_3, 0.5)
    self.assertAlmostEqual(recall_3_bool, 0.5)
    self.assertAlmostEqual(recall_5, 0.5)

  def test_compute_median_rank_at_k(self):
    tp_fp = [
        np.array([1, 0, 0], dtype=float),
        np.array([0, 0.1], dtype=float),
        np.array([0, 0, 0, 0, 0], dtype=float)
    ]
    tp_fp_bool = [
        np.array([True, False, False], dtype=bool),
        np.array([False, True], dtype=float),
        np.array([False, False, False, False, False], dtype=float)
    ]

    median_ranks_1 = metrics.compute_median_rank_at_k(tp_fp, 1)
    median_ranks_3 = metrics.compute_median_rank_at_k(tp_fp, 3)
    median_ranks_5 = metrics.compute_median_rank_at_k(tp_fp, 5)
    median_ranks_3_bool = metrics.compute_median_rank_at_k(tp_fp_bool, 3)

    self.assertEquals(median_ranks_1, 0)
    self.assertEquals(median_ranks_3, 0.5)
    self.assertEquals(median_ranks_3_bool, 0.5)
    self.assertEquals(median_ranks_5, 0.5)


if __name__ == '__main__':
  tf.test.main()
