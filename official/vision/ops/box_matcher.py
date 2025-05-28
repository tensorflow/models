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

"""Box matcher implementation."""

from typing import List, Tuple

import tensorflow as tf, tf_keras


class BoxMatcher:
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  positive_threshold (upper threshold) and negative_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored, for example:
  (1) thresholds=[negative_threshold, positive_threshold], and
      indicators=[negative_value, ignore_value, positive_value]: The similarity
      metrics below negative_threshold will be assigned with negative_value,
      the metrics between negative_threshold and positive_threshold will be
      assigned ignore_value, and the metrics above positive_threshold will be
      assigned positive_value.
  (2) thresholds=[negative_threshold, positive_threshold], and
      indicators=[ignore_value, negative_value, positive_value]: The similarity
      metric below negative_threshold will be assigned with ignore_value,
      the metrics between negative_threshold and positive_threshold will be
      assigned negative_value, and the metrics above positive_threshold will be
      assigned positive_value.
  """

  def __init__(self,
               thresholds: List[float],
               indicators: List[int],
               force_match_for_each_col: bool = False):
    """Construct BoxMatcher.

    Args:
      thresholds: A list of thresholds to classify the matches into different
        types (e.g. positive or negative or ignored match). The list needs to be
        sorted, and will be prepended with -Inf and appended with +Inf.
      indicators: A list of values representing match types (e.g. positive or
        negative or ignored match). len(`indicators`) must equal to
        len(`thresholds`) + 1.
      force_match_for_each_col: If True, ensures that each column is matched to
        at least one row (which is not guaranteed otherwise if the
        positive_threshold is high). Defaults to False. If True, all force
        matched row will be assigned to `indicators[-1]`.

    Raises:
      ValueError: If `threshold` not sorted,
        or len(indicators) != len(threshold) + 1
    """
    if not all([lo <= hi for (lo, hi) in zip(thresholds[:-1], thresholds[1:])]):
      raise ValueError('`threshold` must be sorted, got {}'.format(thresholds))
    self.indicators = indicators
    if len(indicators) != len(thresholds) + 1:
      raise ValueError('len(`indicators`) must be len(`thresholds`) + 1, got '
                       'indicators {}, thresholds {}'.format(
                           indicators, thresholds))
    thresholds = thresholds[:]
    thresholds.insert(0, -float('inf'))
    thresholds.append(float('inf'))
    self.thresholds = thresholds
    self._force_match_for_each_col = force_match_for_each_col

  def __call__(self,
               similarity_matrix: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: A float tensor of shape [num_rows, num_cols] or
        [batch_size, num_rows, num_cols] representing any similarity metric.

    Returns:
      matched_columns: An integer tensor of shape [num_rows] or [batch_size,
        num_rows] storing the index of the matched column for each row.
      match_indicators: An integer tensor of shape [num_rows] or [batch_size,
        num_rows] storing the match type indicator (e.g. positive or negative or
        ignored match).
    """
    squeeze_result = False
    if len(similarity_matrix.shape) == 2:
      squeeze_result = True
      similarity_matrix = tf.expand_dims(similarity_matrix, axis=0)

    static_shape = similarity_matrix.shape.as_list()
    num_rows = static_shape[1] or tf.shape(similarity_matrix)[1]
    batch_size = static_shape[0] or tf.shape(similarity_matrix)[0]

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the rows do not match to any columns.

      Returns:
        matched_columns: An integer tensor of shape [num_rows] or [batch_size,
          num_rows] storing the index of the matched column for each row.
        match_indicators: An integer tensor of shape [num_rows] or [batch_size,
          num_rows] storing the match type indicator (e.g. positive or negative
          or ignored match).
      """
      with tf.name_scope('empty_gt_boxes'):
        matched_columns = tf.zeros([batch_size, num_rows], dtype=tf.int32)
        match_indicators = -tf.ones([batch_size, num_rows], dtype=tf.int32)
        return matched_columns, match_indicators

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matched_columns: An integer tensor of shape [num_rows] or [batch_size,
          num_rows] storing the index of the matched column for each row.
        match_indicators: An integer tensor of shape [num_rows] or [batch_size,
          num_rows] storing the match type indicator (e.g. positive or negative
          or ignored match).
      """
      with tf.name_scope('non_empty_gt_boxes'):
        matched_columns = tf.argmax(
            similarity_matrix, axis=-1, output_type=tf.int32)

        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_vals = tf.reduce_max(similarity_matrix, axis=-1)
        match_indicators = tf.zeros([batch_size, num_rows], tf.int32)

        match_dtype = matched_vals.dtype
        for (ind, low, high) in zip(self.indicators, self.thresholds[:-1],
                                    self.thresholds[1:]):
          low_threshold = tf.cast(low, match_dtype)
          high_threshold = tf.cast(high, match_dtype)
          mask = tf.logical_and(
              tf.greater_equal(matched_vals, low_threshold),
              tf.less(matched_vals, high_threshold))
          match_indicators = self._set_values_using_indicator(
              match_indicators, mask, ind)

        if self._force_match_for_each_col:
          # [batch_size, num_cols], for each column (groundtruth_box), find the
          # best matching row (anchor).
          matching_rows = tf.argmax(
              input=similarity_matrix, axis=1, output_type=tf.int32)
          # [batch_size, num_cols, num_rows], a transposed 0-1 mapping matrix M,
          # where M[j, i] = 1 means column j is matched to row i.
          column_to_row_match_mapping = tf.one_hot(
              matching_rows, depth=num_rows)
          # [batch_size, num_rows], for each row (anchor), find the matched
          # column (groundtruth_box).
          force_matched_columns = tf.argmax(
              input=column_to_row_match_mapping, axis=1, output_type=tf.int32)
          # [batch_size, num_rows]
          force_matched_column_mask = tf.cast(
              tf.reduce_max(column_to_row_match_mapping, axis=1), tf.bool)
          # [batch_size, num_rows]
          matched_columns = tf.where(force_matched_column_mask,
                                     force_matched_columns, matched_columns)
          match_indicators = tf.where(
              force_matched_column_mask, self.indicators[-1] *
              tf.ones([batch_size, num_rows], dtype=tf.int32), match_indicators)

        return matched_columns, match_indicators

    num_gt_boxes = similarity_matrix.shape.as_list()[-1] or tf.shape(
        similarity_matrix)[-1]
    matched_columns, match_indicators = tf.cond(
        pred=tf.greater(num_gt_boxes, 0),
        true_fn=_match_when_rows_are_non_empty,
        false_fn=_match_when_rows_are_empty)

    if squeeze_result:
      matched_columns = tf.squeeze(matched_columns, axis=0)
      match_indicators = tf.squeeze(match_indicators, axis=0)

    return matched_columns, match_indicators

  def _set_values_using_indicator(self, x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)
