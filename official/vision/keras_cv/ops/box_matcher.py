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
      force_match_for_each_row=False,
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
      force_match_for_each_row: If True, ensures that each row is matched to
        at least one column (which is not guaranteed otherwise if the
        positive_threshold is high). Defaults to False.
      negative_value: An integer to fill for negative matches.
      ignore_value: An integer to fill for ignored matches.

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

    self._negative_value = negative_value
    self._ignore_value = ignore_value
    self._force_match_for_each_row = force_match_for_each_row

  def __call__(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: A float tensor of shape [N, M] representing any
        similarity metric.

    Returns:
      A integer tensor with corresponding match indices for each of M columns,
      for positive match, the match result will be the corresponding row index,
      for negative match, the match will be `negative_value`, for ignored match,
      the match result will be `ignore_value`.
    """

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      static_shape = similarity_matrix.shape.as_list()
      num_cols = static_shape[1] or tf.shape(similarity_matrix)[1]
      return -1 * tf.ones([num_cols], dtype=tf.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      matches = tf.argmax(input=similarity_matrix, axis=0, output_type=tf.int32)

      # Deal with matched and unmatched threshold
      if self._positive_threshold is not None:
        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_vals = tf.reduce_max(similarity_matrix, axis=0)
        below_negative_threshold = tf.greater(self._negative_threshold,
                                              matched_vals)
        between_thresholds = tf.logical_and(
            tf.greater_equal(matched_vals, self._negative_threshold),
            tf.greater(self._positive_threshold, matched_vals))

        matches = self._set_values_using_indicator(matches,
                                                   below_negative_threshold,
                                                   self._negative_value)
        matches = self._set_values_using_indicator(matches,
                                                   between_thresholds,
                                                   self._ignore_value)

      if self._force_match_for_each_row:
        num_gt_boxes = similarity_matrix.shape.as_list()[1] or tf.shape(
            similarity_matrix)[1]
        force_match_column_ids = tf.argmax(
            input=similarity_matrix, axis=1, output_type=tf.int32)
        force_match_column_indicators = tf.one_hot(
            force_match_column_ids, depth=num_gt_boxes)
        force_match_row_ids = tf.argmax(
            input=force_match_column_indicators, axis=0, output_type=tf.int32)
        force_match_column_mask = tf.cast(
            tf.reduce_max(force_match_column_indicators, axis=0),
            tf.bool)
        final_matches = tf.where(force_match_column_mask, force_match_row_ids,
                                 matches)
        return final_matches
      else:
        return matches

    num_gt_boxes = similarity_matrix.shape.as_list()[0] or tf.shape(
        similarity_matrix)[0]
    return tf.cond(
        pred=tf.greater(num_gt_boxes, 0),
        true_fn=_match_when_rows_are_non_empty,
        false_fn=_match_when_rows_are_empty)

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
