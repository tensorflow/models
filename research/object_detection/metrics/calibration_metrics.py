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
"""Object detection calibration metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import metrics_impl


def _safe_div(numerator, denominator):
  """Divides two tensors element-wise, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  t = tf.truediv(numerator, denominator)
  zero = tf.zeros_like(t, dtype=denominator.dtype)
  condition = tf.greater(denominator, zero)
  zero = tf.cast(zero, t.dtype)
  return tf.where(condition, t, zero)


def _ece_from_bins(bin_counts, bin_true_sum, bin_preds_sum, name):
  """Calculates Expected Calibration Error from accumulated statistics."""
  bin_accuracies = _safe_div(bin_true_sum, bin_counts)
  bin_confidences = _safe_div(bin_preds_sum, bin_counts)
  abs_bin_errors = tf.abs(bin_accuracies - bin_confidences)
  bin_weights = _safe_div(bin_counts, tf.reduce_sum(bin_counts))
  return tf.reduce_sum(abs_bin_errors * bin_weights, name=name)


def expected_calibration_error(y_true, y_pred, nbins=20):
  """Calculates Expected Calibration Error (ECE).

  ECE is a scalar summary statistic of calibration error. It is the
  sample-weighted average of the difference between the predicted and true
  probabilities of a positive detection across uniformly-spaced model
  confidences [0, 1]. See referenced paper for a thorough explanation.

  Reference:
    Guo, et. al, "On Calibration of Modern Neural Networks"
    Page 2, Expected Calibration Error (ECE).
    https://arxiv.org/pdf/1706.04599.pdf

  This function creates three local variables, `bin_counts`, `bin_true_sum`, and
  `bin_preds_sum` that are used to compute ECE.  For estimation of the metric
  over a stream of data, the function creates an `update_op` operation that
  updates these variables and returns the ECE.

  Args:
    y_true: 1-D tf.int64 Tensor of binarized ground truth, corresponding to each
      prediction in y_pred.
    y_pred: 1-D tf.float32 tensor of model confidence scores in range
      [0.0, 1.0].
    nbins: int specifying the number of uniformly-spaced bins into which y_pred
      will be bucketed.

  Returns:
    value_op: A value metric op that returns ece.
    update_op: An operation that increments the `bin_counts`, `bin_true_sum`,
      and `bin_preds_sum` variables appropriately and whose value matches `ece`.

  Raises:
    InvalidArgumentError: if y_pred is not in [0.0, 1.0].
  """
  bin_counts = metrics_impl.metric_variable(
      [nbins], tf.float32, name='bin_counts')
  bin_true_sum = metrics_impl.metric_variable(
      [nbins], tf.float32, name='true_sum')
  bin_preds_sum = metrics_impl.metric_variable(
      [nbins], tf.float32, name='preds_sum')

  with tf.control_dependencies([
      tf.assert_greater_equal(y_pred, 0.0),
      tf.assert_less_equal(y_pred, 1.0),
  ]):
    bin_ids = tf.histogram_fixed_width_bins(y_pred, [0.0, 1.0], nbins=nbins)

  with tf.control_dependencies([bin_ids]):
    update_bin_counts_op = tf.assign_add(
        bin_counts, tf.cast(tf.bincount(bin_ids, minlength=nbins),
                            dtype=tf.float32))
    update_bin_true_sum_op = tf.assign_add(
        bin_true_sum,
        tf.cast(tf.bincount(bin_ids, weights=y_true, minlength=nbins),
                dtype=tf.float32))
    update_bin_preds_sum_op = tf.assign_add(
        bin_preds_sum,
        tf.cast(tf.bincount(bin_ids, weights=y_pred, minlength=nbins),
                dtype=tf.float32))

  ece_update_op = _ece_from_bins(
      update_bin_counts_op,
      update_bin_true_sum_op,
      update_bin_preds_sum_op,
      name='update_op')
  ece = _ece_from_bins(bin_counts, bin_true_sum, bin_preds_sum, name='value')
  return ece, ece_update_op
