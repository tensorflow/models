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

"""Losses used for segmentation models."""

from typing import Optional, Sequence
import tensorflow as tf


class SegmentationLossDiceScore(object):
  """Semantic segmentation loss using generalized dice score.

  Dice score (DSC) is a similarity measure that equals twice the number of
  elements common to both sets divided by the sum of the number of elements
  in each set. It is commonly used to evaluate segmentation performance to
  measure the overlap of predicted and groundtruth regions.
  (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)

  Generalized dice score is the dice score weighted by the volume of groundtruth
  labels per class. Adaptive dice score adds weights to generalized dice score.
  It assigns larger weights to lower dice score, so that wrong predictions
  contribute more to the total loss. Model will then be trained to focus more on
  these hard examples.
  """

  def __init__(self,
               metric_type: Optional[str] = None,
               axis: Optional[Sequence[int]] = (1, 2, 3)):
    """Initializes dice score loss object.

    Args:
      metric_type: An optional `str` specifying the type of the dice score to
        compute. Compute generalized or adaptive dice score if metric type is
        `generalized` or `adaptive`; otherwise compute original dice score.
      axis: An optional sequence of `int` specifying the axis to perform reduce
        ops for raw dice score.
    """
    self._dice_score = 0
    self._metric_type = metric_type
    self._axis = axis

  def __call__(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Computes and returns a loss based on 1 - dice score.

    Args:
      logits: A Tensor of the prediction.
      labels: A Tensor of the groundtruth label.

    Returns:
      The loss value of (1 - dice score).
    """
    labels = tf.cast(labels, logits.dtype)

    if labels.get_shape().ndims < 2 or logits.get_shape().ndims < 2:
      raise ValueError('The labels and logits must be at least rank 2.')

    epsilon = tf.keras.backend.epsilon()
    keep_label_axis = list(range(len(logits.shape) - 1))
    keep_batch_axis = list(range(1, len(logits.shape)))

    # Compute sample mask to filter out samples with both all-0's labels and
    # predictions because such samples should not contribute to mean dice score
    # in this batch.
    sample_mask = tf.logical_or(
        tf.cast(tf.reduce_sum(labels, axis=keep_batch_axis), dtype=tf.bool),
        tf.cast(tf.reduce_sum(logits, axis=keep_batch_axis), dtype=tf.bool))
    labels = tf.boolean_mask(labels, sample_mask)
    logits = tf.boolean_mask(logits, sample_mask)

    # If all samples are filtered out, return 0 as the loss so this batch does
    # not contribute.
    if labels.shape[0] == 0:
      return tf.convert_to_tensor(0.0)

    # Calculate intersections and unions per class.
    intersection = tf.reduce_sum(labels * logits, axis=keep_label_axis)
    union = tf.reduce_sum(labels + logits, axis=keep_label_axis)

    if self._metric_type == 'generalized':
      # Calculate the volume of groundtruth labels.
      w = tf.math.reciprocal(
          tf.square(tf.reduce_sum(labels, axis=keep_label_axis)) + epsilon)

      # Calculate the weighted dice score and normalizer.
      dice = 2 * tf.reduce_sum(w * intersection)
      normalizer = tf.reduce_sum(w * union)
      if normalizer == 0:
        return tf.convert_to_tensor(1.0)
      dice = tf.cast(dice, dtype=tf.float32)
      normalizer = tf.cast(normalizer, dtype=tf.float32)

      return 1 - tf.reduce_mean(dice / normalizer)
    elif self._metric_type == 'adaptive':
      dice = 2.0 * intersection / (union + epsilon)
      # Calculate weights based on Dice scores.
      weights = tf.exp(-1.0 * dice)

      # Multiply weights by corresponding scores and get sum.
      weighted_dice = tf.reduce_sum(weights * dice)

      # Calculate normalization factor.
      normalizer = tf.cast(tf.size(input=dice), dtype=tf.float32) * tf.exp(-1.0)
      if normalizer == 0:
        return tf.convert_to_tensor(1.0)
      weighted_dice = tf.cast(weighted_dice, dtype=tf.float32)
      return 1 - tf.reduce_mean(weighted_dice / normalizer)
    else:
      summation = tf.reduce_sum(
          labels, axis=self._axis) + tf.reduce_sum(
              logits, axis=self._axis)
      dice = (2 * tf.reduce_sum(labels * logits, axis=self._axis)) / (
          summation + epsilon)
      return 1 - tf.reduce_mean(dice)
