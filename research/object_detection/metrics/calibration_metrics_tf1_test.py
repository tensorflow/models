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
"""Tests for calibration_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from object_detection.metrics import calibration_metrics
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class CalibrationLibTest(tf.test.TestCase):

  @staticmethod
  def _get_calibration_placeholders():
    """Returns TF placeholders for y_true and y_pred."""
    return (tf.placeholder(tf.int64, shape=(None)),
            tf.placeholder(tf.float32, shape=(None)))

  def test_expected_calibration_error_all_bins_filled(self):
    """Test expected calibration error when all bins contain predictions."""
    y_true, y_pred = self._get_calibration_placeholders()
    expected_ece_op, update_op = calibration_metrics.expected_calibration_error(
        y_true, y_pred, nbins=2)
    with self.test_session() as sess:
      metrics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
      sess.run(tf.variables_initializer(var_list=metrics_vars))
      # Bin calibration errors (|confidence - accuracy| * bin_weight):
      # - [0,0.5): |0.2 - 0.333| * (3/5) = 0.08
      # - [0.5, 1]: |0.75 - 0.5| * (2/5) = 0.1
      sess.run(
          update_op,
          feed_dict={
              y_pred: np.array([0., 0.2, 0.4, 0.5, 1.0]),
              y_true: np.array([0, 0, 1, 0, 1])
          })
    actual_ece = 0.08 + 0.1
    expected_ece = sess.run(expected_ece_op)
    self.assertAlmostEqual(actual_ece, expected_ece)

  def test_expected_calibration_error_all_bins_not_filled(self):
    """Test expected calibration error when no predictions for one bin."""
    y_true, y_pred = self._get_calibration_placeholders()
    expected_ece_op, update_op = calibration_metrics.expected_calibration_error(
        y_true, y_pred, nbins=2)
    with self.test_session() as sess:
      metrics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
      sess.run(tf.variables_initializer(var_list=metrics_vars))
      # Bin calibration errors (|confidence - accuracy| * bin_weight):
      # - [0,0.5): |0.2 - 0.333| * (3/5) = 0.08
      # - [0.5, 1]: |0.75 - 0.5| * (2/5) = 0.1
      sess.run(
          update_op,
          feed_dict={
              y_pred: np.array([0., 0.2, 0.4]),
              y_true: np.array([0, 0, 1])
          })
    actual_ece = np.abs(0.2 - (1 / 3.))
    expected_ece = sess.run(expected_ece_op)
    self.assertAlmostEqual(actual_ece, expected_ece)

  def test_expected_calibration_error_with_multiple_data_streams(self):
    """Test expected calibration error when multiple data batches provided."""
    y_true, y_pred = self._get_calibration_placeholders()
    expected_ece_op, update_op = calibration_metrics.expected_calibration_error(
        y_true, y_pred, nbins=2)
    with self.test_session() as sess:
      metrics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
      sess.run(tf.variables_initializer(var_list=metrics_vars))
      # Identical data to test_expected_calibration_error_all_bins_filled,
      # except split over three batches.
      sess.run(
          update_op,
          feed_dict={
              y_pred: np.array([0., 0.2]),
              y_true: np.array([0, 0])
          })
      sess.run(
          update_op,
          feed_dict={
              y_pred: np.array([0.4, 0.5]),
              y_true: np.array([1, 0])
          })
      sess.run(
          update_op, feed_dict={
              y_pred: np.array([1.0]),
              y_true: np.array([1])
          })
    actual_ece = 0.08 + 0.1
    expected_ece = sess.run(expected_ece_op)
    self.assertAlmostEqual(actual_ece, expected_ece)


if __name__ == '__main__':
  tf.test.main()
