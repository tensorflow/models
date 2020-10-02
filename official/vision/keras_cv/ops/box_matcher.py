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

"""Box matcher implementation."""


import tensorflow as tf


class BoxMatcher:
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  positive_threshold (upper threshold) and negative_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored:
  (1) similarity >= positive_threshold: Highest similarity. Matched/Positive!
  (2) positive_threshold > similarity >= negative_threshold: Medium similarity.
        This is Ignored.
  (3) negative_threshold > similarity: Lowest similarity for Negative Match.
  For ignored matches this class sets the values in the Match object to -2.
  """

  def __init__(
      self,
      positive_threshold,
      negative_threshold=None,
      force_match_for_each_col=False,
      negative_lower_than_ignore=True,
      positive_value=1,
      negative_value=-1,
      ignore_value=-2):
    """Construct BoxMatcher.

    Args:
      positive_threshold: Threshold for positive matches. Positive if
        sim >= positive_threshold, where sim is the maximum value of the
        similarity matrix for a given column. Set to None for no threshold.
      negative_threshold: Threshold for negative matches. Negative if
        sim < negative_threshold or
        positive_threshold > sim >= negative_threshold.
        Defaults to positive_threshold when set to None.
      force_match_for_each_col: If True, ensures that each column is matched to
        at least one row (which is not guaranteed otherwise if the
        positive_threshold is high). Defaults to False.
      negative_lower_than_ignore: If True, the threshold is
        positive|ignore|negative, else positive|negative|ignore. Defaults to
        True.
      positive_value: An integer to fill for positive match labels.
      negative_value: An integer to fill for negative match labels.
      ignore_value: An integer to fill for ignored match labels.

    Raises:
      ValueError: If negative_threshold > positive_threshold.
    """
    self._positive_threshold = positive_threshold
    if negative_threshold is None:
      self._negative_threshold = positive_threshold
    else:
      if negative_threshold > positive_threshold:
        raise ValueError('negative_threshold needs to be smaller or equal'
                         'to positive_threshold')
      self._negative_threshold = negative_threshold

    self._positive_value = positive_value
    self._negative_value = negative_value
    self._ignore_value = ignore_value
    self._force_match_for_each_col = force_match_for_each_col
    self._negative_lower_than_ignore = negative_lower_than_ignore

  def __call__(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: A float tensor of shape [N, M] representing any
        similarity metric.

    Returns:
      A integer tensor of shape [N] with corresponding match indices for each
      of M columns, for positive match, the match result will be the
      corresponding row index, for negative match, the match will be
      `negative_value`, for ignored match, the match result will be
      `ignore_value`.
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
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      with tf.name_scope('empty_gt_boxes'):
        matches = tf.zeros([batch_size, num_rows], dtype=tf.int32)
        match_labels = self._negative_value * tf.ones(
            [batch_size, num_rows], dtype=tf.int32)
        return matches, match_labels

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      with tf.name_scope('non_empty_gt_boxes'):
        matches = tf.argmax(similarity_matrix, axis=-1, output_type=tf.int32)

        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_vals = tf.reduce_max(similarity_matrix, axis=-1)
        matched_labels = self._positive_value * tf.ones(
            [batch_size, num_rows], tf.int32)

        positive_threshold = tf.cast(
            self._positive_threshold, matched_vals.dtype)
        negative_threshold = tf.cast(
            self._negative_threshold, matched_vals.dtype)
        below_negative_threshold = tf.greater(negative_threshold, matched_vals)
        between_thresholds = tf.logical_and(
            tf.greater_equal(matched_vals, negative_threshold),
            tf.greater(positive_threshold, matched_vals))

        if self._negative_lower_than_ignore:
          matched_labels = self._set_values_using_indicator(
              matched_labels, below_negative_threshold, self._negative_value)
          matched_labels = self._set_values_using_indicator(
              matched_labels, between_thresholds, self._ignore_value)
        else:
          matched_labels = self._set_values_using_indicator(
              matched_labels, below_negative_threshold, self._ignore_value)
          matched_labels = self._set_values_using_indicator(
              matched_labels, between_thresholds, self._negative_value)

        if self._force_match_for_each_col:
          # [batch_size, M], for each col (groundtruth_box), find the best
          # matching row (anchor).
          force_match_column_ids = tf.argmax(
              input=similarity_matrix, axis=1, output_type=tf.int32)
          # [batch_size, M, N]
          force_match_column_indicators = tf.one_hot(
              force_match_column_ids, depth=num_rows)
          # [batch_size, N], for each row (anchor), find the largest column
          # index for groundtruth box
          force_match_row_ids = tf.argmax(
              input=force_match_column_indicators, axis=1, output_type=tf.int32)
          # [batch_size, N]
          force_match_column_mask = tf.cast(
              tf.reduce_max(force_match_column_indicators, axis=1),
              tf.bool)
          # [batch_size, N]
          final_matches = tf.where(force_match_column_mask, force_match_row_ids,
                                   matches)
          final_matched_labels = tf.where(
              force_match_column_mask,
              self._positive_value * tf.ones(
                  [batch_size, num_rows], dtype=tf.int32),
              matched_labels)
          return final_matches, final_matched_labels
        else:
          return matches, matched_labels

    num_gt_boxes = similarity_matrix.shape.as_list()[-1] or tf.shape(
        similarity_matrix)[-1]
    result_match, result_match_labels = tf.cond(
        pred=tf.greater(num_gt_boxes, 0),
        true_fn=_match_when_rows_are_non_empty,
        false_fn=_match_when_rows_are_empty)

    if squeeze_result:
      result_match = tf.squeeze(result_match, axis=0)
      result_match_labels = tf.squeeze(result_match_labels, axis=0)

    return result_match, result_match_labels

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
