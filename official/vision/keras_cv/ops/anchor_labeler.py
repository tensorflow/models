# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Definition of anchor labeler, which assigns ground truth boxes to anchors."""

import tensorflow as tf


class AnchorLabeler:
  """Labeler for dense object detector."""

  def __init__(
      self,
      positive_class_weight=1.0,
      positive_regression_weight=1.0,
      negative_class_weight=1.0,
      negative_regression_weight=0.0,
      negative_class_label=-1,
      ignore_class_label=-2,
      negative_regression_label=0.,
      ignore_regression_label=0.):
    """Constructs Anchor Labeler.

    Args:
      positive_class_weight: classification weight to be associated to positive
        matched anchor. Defaults to 1.0.
      positive_regression_weight: regression weight to be associated to positive
        matched anchor. Defaults to 1.0.
      negative_class_weight: classification weight to be associated to negative
        matched anchor. Default to 1.0
      negative_regression_weight: classification weight to be associated to
        negative matched anchor. Default to 0.0.
      negative_class_label: An integer for classification label to be associated
        for negative matched anchor. Defaults to -1.
      ignore_class_label: An integer for classification label to be associated
        for ignored anchor. Defaults to -2.
      negative_regression_label: A float for regression label to be associated
        for negative matched anchor. Defaults to 0.
      ignore_regression_label: A float for regression label to be associated
        for ignored anchor. Defaults to 0.
    """
    self.positive_class_weight = positive_class_weight
    self.positive_regression_weight = positive_regression_weight
    self.negative_class_weight = negative_class_weight
    self.negative_regression_weight = negative_regression_weight
    self.negative_class_label = negative_class_label
    self.ignore_class_label = ignore_class_label
    self.negative_regression_label = negative_regression_label
    self.ignore_regression_label = ignore_regression_label

  def __call__(self, boxes, labels, matches):
    """Labels anchors with ground truth inputs.

    Args:
      boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      labels: An integer tensor with shape [N, 1] representing groundtruth
        classes.
      matches: An integer tensor with shape [N] representing match results, must
        be -1 for negative matched anchor, and -2 for ignored anchor.

    Returns:
      class_targets: A integer Tensor with shape [num_anchors].
      box_targets: A float Tensor with shape [num_anchors, 4].
      class_weights: A float Tensor with shape [num_anchors], that
        serves as masking / sample weight for classification loss. Its value
        is 1.0 for positive and negative matched anchors, and 0.0 for ignored
        anchors.
      box_weights: A float Tensor with shape [num_anchors], that
        serves as masking / sample weight for regression loss. Its value is
        1.0 for positive matched anchors, and 0.0 for negative and ignored
        anchors.
    """

    class_targets = self._gather_based_on_match(
        matches, tf.cast(labels, tf.int32),
        negative_value=tf.constant([self.negative_class_label], tf.int32),
        ignored_value=tf.constant([self.ignore_class_label], tf.int32))

    negative_reg_value = tf.constant(
        [self.negative_regression_label] * 4, dtype=tf.float32)
    ignore_reg_value = tf.constant(
        [self.ignore_regression_label] * 4, dtype=tf.float32)
    reg_targets = self._gather_based_on_match(
        matches, boxes, negative_reg_value, ignore_reg_value)

    num_gt_boxes = boxes.shape.as_list()[0] or tf.shape(boxes)[0]

    groundtruth_class_weights = self.positive_class_weight * tf.ones(
        [num_gt_boxes], dtype=tf.float32)
    class_weights = self._gather_based_on_match(
        matches, groundtruth_class_weights,
        negative_value=self.negative_class_weight,
        ignored_value=0.)

    groundtruth_reg_weights = self.positive_regression_weight * tf.ones(
        [num_gt_boxes], dtype=tf.float32)
    reg_weights = self._gather_based_on_match(
        matches, groundtruth_reg_weights,
        negative_value=self.negative_regression_weight, ignored_value=0.)

    return class_targets, reg_targets, class_weights, reg_weights

  def _gather_based_on_match(
      self, matches, inputs, negative_value, ignored_value):
    """Gathers elements from `input_tensor` based on match results.

    For columns that are matched to a row, gathered_tensor[col] is set to
    input_tensor[match[col]]. For columns that are unmatched,
    gathered_tensor[col] is set to negative_value. Finally, for columns that
    are ignored gathered_tensor[col] is set to ignored_value.

    Note that the input_tensor.shape[1:] must match with unmatched_value.shape
    and ignored_value.shape

    Args:
      matches: A integer tensor with shape [N] representing the
        matching results of anchors. (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.
      inputs: Tensor to gather values from.
      negative_value: Constant tensor value for unmatched columns.
      ignored_value: Constant tensor value for ignored columns.

    Returns:
      gathered_tensor: A tensor containing values gathered from input_tensor.
        The shape of the gathered tensor is [match.shape[0]] +
        input_tensor.shape[1:].
    """
    inputs = tf.concat(
        [tf.stack([ignored_value, negative_value]), inputs], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    gathered_tensor = tf.gather(inputs, gather_indices)
    return gathered_tensor
