# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""IOU Metrics used for semantic segmentation models."""

import numpy as np
import tensorflow as tf


class PerClassIoU(tf.keras.metrics.Metric):
  """Computes the per-class Intersection-Over-Union metric.

  Mean Intersection-Over-Union is a common evaluation metric for semantic image
  segmentation, which first computes the IOU for each semantic class.
  IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
  The predictions are accumulated in a confusion matrix, weighted by
  `sample_weight` and the metric is then calculated from it.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Example:

  >>> # cm = [[1, 1],
  >>> #        [1, 1]]
  >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
  >>> # iou = true_positives / (sum_row + sum_col - true_positives))
  >>> # result = [(1 / (2 + 2 - 1), 1 / (2 + 2 - 1)] = 0.33
  >>> m = tf.keras.metrics.MeanIoU(num_classes=2)
  >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
  >>> m.result().numpy()
  [0.33333334, 0.33333334]

  """

  def __init__(self, num_classes, name=None, dtype=None):
    """Initializes `PerClassIoU`.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    """

    super(PerClassIoU, self).__init__(name=name, dtype=dtype)
    self.num_classes = num_classes

    # Variable to accumulate the predictions in the confusion matrix.
    self.total_cm = self.add_weight(
        'total_confusion_matrix',
        shape=(num_classes, num_classes),
        initializer=tf.compat.v1.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      IOU per class.
    """

    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)

    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
      y_pred = tf.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = tf.reshape(y_true, [-1])

    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self._dtype)
      if sample_weight.shape.ndims > 1:
        sample_weight = tf.reshape(sample_weight, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        self.num_classes,
        weights=sample_weight,
        dtype=self._dtype)
    return self.total_cm.assign_add(current_cm)

  def result(self):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.cast(
        tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = tf.cast(
        tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = tf.cast(
        tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    return tf.math.divide_no_nan(true_positives, denominator)

  def reset_states(self):
    tf.keras.backend.set_value(
        self.total_cm, np.zeros((self.num_classes, self.num_classes)))

  def get_config(self):
    config = {'num_classes': self.num_classes}
    base_config = super(PerClassIoU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
