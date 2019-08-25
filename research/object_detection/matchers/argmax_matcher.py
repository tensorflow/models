# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Argmax matcher implementation.

This class takes a similarity matrix and matches columns to rows based on the
maximum value per column. One can specify matched_thresholds and
to prevent columns from matching to rows (generally resulting in a negative
training example) and unmatched_theshold to ignore the match (generally
resulting in neither a positive or negative training example).

This matcher is used in Fast(er)-RCNN.

Note: matchers are used in TargetAssigners. There is a create_target_assigner
factory function for popular implementations.
"""
import tensorflow as tf

from object_detection.core import matcher
from object_detection.utils import shape_utils


class ArgMaxMatcher(matcher.Matcher):
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  matched_threshold (upper threshold) and unmatched_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored:
  (1) similarity >= matched_threshold: Highest similarity. Matched/Positive!
  (2) matched_threshold > similarity >= unmatched_threshold: Medium similarity.
          Depending on negatives_lower_than_unmatched, this is either
          Unmatched/Negative OR Ignore.
  (3) unmatched_threshold > similarity: Lowest similarity. Depending on flag
          negatives_lower_than_unmatched, either Unmatched/Negative OR Ignore.
  For ignored matches this class sets the values in the Match object to -2.
  """

  def __init__(self,
               matched_threshold,
               unmatched_threshold=None,
               negatives_lower_than_unmatched=True,
               force_match_for_each_row=False,
               use_matmul_gather=False):
    """Construct ArgMaxMatcher.

    Args:
      matched_threshold: Threshold for positive matches. Positive if
        sim >= matched_threshold, where sim is the maximum value of the
        similarity matrix for a given column. Set to None for no threshold.
      unmatched_threshold: Threshold for negative matches. Negative if
        sim < unmatched_threshold. Defaults to matched_threshold
        when set to None.
      negatives_lower_than_unmatched: Boolean which defaults to True. If True
        then negative matches are the ones below the unmatched_threshold,
        whereas ignored matches are in between the matched and umatched
        threshold. If False, then negative matches are in between the matched
        and unmatched threshold, and everything lower than unmatched is ignored.
      force_match_for_each_row: If True, ensures that each row is matched to
        at least one column (which is not guaranteed otherwise if the
        matched_threshold is high). Defaults to False. See
        argmax_matcher_test.testMatcherForceMatch() for an example.
      use_matmul_gather: Force constructed match objects to use matrix
        multiplication based gather instead of standard tf.gather.
        (Default: False).

    Raises:
      ValueError: if unmatched_threshold is set but matched_threshold is not set
        or if unmatched_threshold > matched_threshold.
    """
    super(ArgMaxMatcher, self).__init__(use_matmul_gather=use_matmul_gather)
    if (matched_threshold is None) and (unmatched_threshold is not None):
      raise ValueError('Need to also define matched_threshold when'
                       'unmatched_threshold is defined')
    self._matched_threshold = matched_threshold
    if unmatched_threshold is None:
      self._unmatched_threshold = matched_threshold
    else:
      if unmatched_threshold > matched_threshold:
        raise ValueError('unmatched_threshold needs to be smaller or equal'
                         'to matched_threshold')
      self._unmatched_threshold = unmatched_threshold
    if not negatives_lower_than_unmatched:
      if self._unmatched_threshold == self._matched_threshold:
        raise ValueError('When negatives are in between matched and '
                         'unmatched thresholds, these cannot be of equal '
                         'value. matched: {}, unmatched: {}'.format(
                             self._matched_threshold,
                             self._unmatched_threshold))
    self._force_match_for_each_row = force_match_for_each_row
    self._negatives_lower_than_unmatched = negatives_lower_than_unmatched

  def _match(self, similarity_matrix, valid_rows):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: tensor of shape [N, M] representing any similarity
        metric.
      valid_rows: a boolean tensor of shape [N] indicating valid rows.

    Returns:
      Match object with corresponding matches for each of M columns.
    """

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
          similarity_matrix)
      return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)

      # Deal with matched and unmatched threshold
      if self._matched_threshold is not None:
        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_vals = tf.reduce_max(similarity_matrix, 0)
        below_unmatched_threshold = tf.greater(self._unmatched_threshold,
                                               matched_vals)
        between_thresholds = tf.logical_and(
            tf.greater_equal(matched_vals, self._unmatched_threshold),
            tf.greater(self._matched_threshold, matched_vals))

        if self._negatives_lower_than_unmatched:
          matches = self._set_values_using_indicator(matches,
                                                     below_unmatched_threshold,
                                                     -1)
          matches = self._set_values_using_indicator(matches,
                                                     between_thresholds,
                                                     -2)
        else:
          matches = self._set_values_using_indicator(matches,
                                                     below_unmatched_threshold,
                                                     -2)
          matches = self._set_values_using_indicator(matches,
                                                     between_thresholds,
                                                     -1)

      if self._force_match_for_each_row:
        similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
            similarity_matrix)
        force_match_column_ids = tf.argmax(similarity_matrix, 1,
                                           output_type=tf.int32)
        force_match_column_indicators = (
            tf.one_hot(
                force_match_column_ids, depth=similarity_matrix_shape[1]) *
            tf.cast(tf.expand_dims(valid_rows, axis=-1), dtype=tf.float32))
        force_match_row_ids = tf.argmax(force_match_column_indicators, 0,
                                        output_type=tf.int32)
        force_match_column_mask = tf.cast(
            tf.reduce_max(force_match_column_indicators, 0), tf.bool)
        final_matches = tf.where(force_match_column_mask,
                                 force_match_row_ids, matches)
        return final_matches
      else:
        return matches

    if similarity_matrix.shape.is_fully_defined():
      if shape_utils.get_dim_as_int(similarity_matrix.shape[0]) == 0:
        return _match_when_rows_are_empty()
      else:
        return _match_when_rows_are_non_empty()
    else:
      return tf.cond(
          tf.greater(tf.shape(similarity_matrix)[0], 0),
          _match_when_rows_are_non_empty, _match_when_rows_are_empty)

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
