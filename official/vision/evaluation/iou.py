# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import tensorflow as tf, tf_keras


class PerClassIoU(tf_keras.metrics.MeanIoU):
  """Computes the per-class Intersection-Over-Union metric.

  This metric computes the IOU for each semantic class.
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
  >>> m = tf_keras.metrics.MeanIoU(num_classes=2)
  >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
  >>> m.result().numpy()
  [0.33333334, 0.33333334]
  """

  def result(self):
    """Compute IoU for each class via the confusion matrix."""
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


class PerClassIoUV2(tf_keras.metrics.Metric):
  """Computes the per-class Intersection-Over-Union metric.

  This implementation converts predictions and ground-truth to binary masks,
  and uses logical AND and OR to compute intersection and union, which is much
  faster than the PerClassIoU (using confusion matrix) above on TPU, but slower
  on CPU and GPU.
  """

  def __init__(self,
               num_classes: int,
               name: Optional[str] = None,
               dtype: Optional[Union[str, tf.dtypes.DType]] = tf.float32,
               shape: Optional[Sequence[int]] = None,
               sparse_y_true: bool = False,
               sparse_y_pred: bool = False,
               axis: int = -1):
    """Initialization for PerClassIoU.

    Args:
      num_classes: `int`, number of classes.
      name: `str`, name of the metric instance.
      dtype: data type of the metric result.
      shape: shape of the metrics result.
      sparse_y_true: whether ground truth labels are encoded using integers or
        dense one-hot vectors.
      sparse_y_pred: whether predictions are encoded using integers or dense
        one-hot vectors.
      axis: (Optional) Defaults to -1. The dimension containing the one-hot
        values.
    """
    super().__init__(name=name, dtype=dtype)
    self.num_classes = num_classes
    self.sparse_y_true = sparse_y_true
    self.sparse_y_pred = sparse_y_pred
    self.axis = axis

    # Variable to accumulate the intersection & union.
    # intersection = true_positives
    if not shape:
      shape = [num_classes]
    self.intersection_per_class = self.add_weight(
        'intersection_per_class', shape, initializer='zeros', dtype=tf.float32)
    # union = true_positives + false_positive + false_negative
    self.union_per_class = self.add_weight(
        'union_per_class', shape, initializer='zeros', dtype=tf.float32)

  def reset_state(self):
    """Resets all of the metric state variables."""
    self.intersection_per_class.assign(
        tf.zeros_like(self.intersection_per_class)
    )
    self.union_per_class.assign(tf.zeros_like(self.union_per_class))

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    """Updates metric state by accumulating the variables.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
    """

    if self.sparse_y_true:
      # Shape: (..., num_classes, ...)
      y_true = tf.one_hot(
          tf.cast(y_true, dtype=tf.int32),
          self.num_classes,
          axis=self.axis,
          on_value=True,
          off_value=False,
      )
    if self.sparse_y_pred:
      # Shape: (..., num_classes, ...)
      y_pred = tf.one_hot(
          tf.cast(y_pred, dtype=tf.int32),
          self.num_classes,
          axis=self.axis,
          on_value=True,
          off_value=False,
      )

    one_hot_axis = self.axis if self.axis >= 0 else (
        len(y_true.get_shape().as_list()) + self.axis)
    # Reduce sum the leading dimensions.
    # Shape: (num_classes, ...)
    current_intersection = tf.math.count_nonzero(
        y_pred & y_true, axis=np.arange(one_hot_axis), dtype=tf.float32
    )
    # Shape: (num_classes, ...)
    current_union = tf.math.count_nonzero(
        y_pred | y_true, axis=np.arange(one_hot_axis), dtype=tf.float32
    )

    self.intersection_per_class.assign_add(
        tf.cast(current_intersection, self.intersection_per_class.dtype))
    self.union_per_class.assign_add(
        tf.cast(current_union, self.union_per_class.dtype))

  def result(self) -> tf.Tensor:
    """Computes IoU for each class."""
    return tf.cast(
        tf.math.divide_no_nan(self.intersection_per_class,
                              self.union_per_class), self.dtype)

  def get_config(self) -> Dict[str, Any]:
    """Returns the serializable config of the metric."""
    return {
        'num_classes': self.num_classes,
        'name': self.name,
        'dtype': self.dtype,
        'sparse_y_true': self.sparse_y_true,
        'sparse_y_pred': self.sparse_y_pred,
        'axis': self.axis,
    }
