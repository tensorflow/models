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

import numpy as np
import tensorflow as tf

from object_detection.utils import metrics


class MetricsTest(tf.test.TestCase):

  def test_compute_cor_loc(self):
    num_gt_imgs_per_class = np.array([100, 1, 5, 1, 1], dtype=int)
    num_images_correctly_detected_per_class = np.array([10, 0, 1, 0, 0],
                                                       dtype=int)
    corloc = metrics.compute_cor_loc(num_gt_imgs_per_class,
                                     num_images_correctly_detected_per_class)
    expected_corloc = np.array([0.1, 0, 0.2, 0, 0], dtype=float)
    self.assertTrue(np.allclose(corloc, expected_corloc))

  def test_compute_cor_loc_nans(self):
    num_gt_imgs_per_class = np.array([100, 0, 0, 1, 1], dtype=int)
    num_images_correctly_detected_per_class = np.array([10, 0, 1, 0, 0],
                                                       dtype=int)
    corloc = metrics.compute_cor_loc(num_gt_imgs_per_class,
                                     num_images_correctly_detected_per_class)
    expected_corloc = np.array([0.1, np.nan, np.nan, 0, 0], dtype=float)
    self.assertAllClose(corloc, expected_corloc)

  def test_compute_precision_recall(self):
    num_gt = 10
    scores = np.array([0.4, 0.3, 0.6, 0.2, 0.7, 0.1], dtype=float)
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    accumulated_tp_count = np.array([0, 1, 1, 2, 2, 3], dtype=float)
    expected_precision = accumulated_tp_count / np.array([1, 2, 3, 4, 5, 6])
    expected_recall = accumulated_tp_count / num_gt
    precision, recall = metrics.compute_precision_recall(scores, labels, num_gt)
    self.assertAllClose(precision, expected_precision)
    self.assertAllClose(recall, expected_recall)

  def test_compute_average_precision(self):
    precision = np.array([0.8, 0.76, 0.9, 0.65, 0.7, 0.5, 0.55, 0], dtype=float)
    recall = np.array([0.3, 0.3, 0.4, 0.4, 0.45, 0.45, 0.5, 0.5], dtype=float)
    processed_precision = np.array([0.9, 0.9, 0.9, 0.7, 0.7, 0.55, 0.55, 0],
                                   dtype=float)
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


if __name__ == '__main__':
  tf.test.main()
