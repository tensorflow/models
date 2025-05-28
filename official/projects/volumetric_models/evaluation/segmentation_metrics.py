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

"""Metrics for segmentation."""
from typing import Optional

import tensorflow as tf, tf_keras
from official.projects.volumetric_models.losses import segmentation_losses


class DiceScore:
  """Dice score metric for semantic segmentation.

  This class follows the same function interface as tf_keras.metrics.Metric but
  does not derive from tf_keras.metrics.Metric or utilize its functions. The
  reason is a tf_keras.metrics.Metric object does not run well on CPU while
  created on GPU, when running with MirroredStrategy. The same interface allows
  for minimal change to the upstream tasks.

  Attributes:
    name: The name of the metric.
    dtype: The dtype of the metric, for example, tf.float32.
  """

  def __init__(self,
               num_classes: int,
               metric_type: Optional[str] = None,
               per_class_metric: bool = False,
               name: Optional[str] = None,
               dtype: Optional[str] = None):
    """Constructs segmentation evaluator class.

    Args:
      num_classes: The number of classes.
      metric_type: An optional `str` of type of dice scores.
      per_class_metric: Whether to report per-class metric.
      name: A `str`, name of the metric instance..
      dtype: The data type of the metric result.
    """
    self._num_classes = num_classes
    self._per_class_metric = per_class_metric
    self._dice_op_overall = segmentation_losses.SegmentationLossDiceScore(
        metric_type=metric_type)
    self._dice_scores_overall = tf.Variable(0.0)
    self._count = tf.Variable(0.0)

    if self._per_class_metric:
      # Always use raw dice score for per-class metrics, so metric_type is None
      # by default.
      self._dice_op_per_class = segmentation_losses.SegmentationLossDiceScore()

      self._dice_scores_per_class = [
          tf.Variable(0.0) for _ in range(num_classes)
      ]
      self._count_per_class = [tf.Variable(0.0) for _ in range(num_classes)]

    self.name = name
    self.dtype = dtype

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
    """Updates metric state.

    Args:
      y_true: The true labels of size [batch, width, height, volume,
        num_classes].
      y_pred: The prediction of size [batch, width, height, volume,
        num_classes].

    Raises:
      ValueError: If number of classes from groundtruth label does not equal to
        `num_classes`.
    """
    if self._num_classes != y_true.get_shape()[-1]:
      raise ValueError(
          'The number of classes from groundtruth labels and `num_classes` '
          'should equal, but they are {0} and {1}.'.format(
              self._num_classes,
              y_true.get_shape()[-1]))

    # If both y_pred and y_true are all 0s, we skip computing the metrics;
    # otherwise the averaged metrics will be erroneously lower.
    if tf.reduce_sum(y_true) != 0 or tf.reduce_sum(y_pred) != 0:
      self._count.assign_add(1.)
      self._dice_scores_overall.assign_add(
          1 - self._dice_op_overall(y_pred, y_true))
      if self._per_class_metric:
        for class_id in range(self._num_classes):
          if tf.reduce_sum(y_true[..., class_id]) != 0 or tf.reduce_sum(
              y_pred[..., class_id]) != 0:
            self._count_per_class[class_id].assign_add(1.)
            self._dice_scores_per_class[class_id].assign_add(
                1 - self._dice_op_per_class(y_pred[...,
                                                   class_id], y_true[...,
                                                                     class_id]))

  def result(self) -> tf.Tensor:
    """Computes and returns the metric.

    The first one is `generalized` or `adaptive` overall dice score, depending
    on `metric_type`. If `per_class_metric` is True, `num_classes` elements are
    also appended to the overall metric, as the per-class raw dice scores.

    Returns:
      The resulting dice scores.
    """
    if self._per_class_metric:
      dice_scores = [
          tf.math.divide_no_nan(self._dice_scores_overall, self._count)
      ]
      for class_id in range(self._num_classes):
        dice_scores.append(
            tf.math.divide_no_nan(self._dice_scores_per_class[class_id],
                                  self._count_per_class[class_id]))
      return tf.stack(dice_scores)
    else:
      return tf.math.divide_no_nan(self._dice_scores_overall, self._count)

  def reset_states(self):
    """Resets the metrcis to the initial state."""
    self._count = tf.Variable(0.0)
    self._dice_scores_overall = tf.Variable(0.0)
    if self._per_class_metric:
      for class_id in range(self._num_classes):
        self._dice_scores_per_class[class_id] = tf.Variable(0.0)
        self._count_per_class[class_id] = tf.Variable(0.0)
