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

"""Matcher interface and Match class.

This module defines the Matcher interface and the Match object. The job of the
matcher is to match row and column indices based on the similarity matrix and
other optional parameters. Each column is matched to at most one row. There
are three possibilities for the matching:

1) match: A column matches a row.
2) no_match: A column does not match any row.
3) ignore: A column that is neither 'match' nor no_match.

The ignore case is regularly encountered in object detection: when an anchor has
a relatively small overlap with a ground-truth box, one neither wants to
consider this box a positive example (match) nor a negative example (no match).

The Match class is used to store the match results and it provides simple apis
to query the results.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class Match(object):
  """Class to store results from the matcher.

  This class is used to store the results from the matcher. It provides
  convenient methods to query the matching results.
  """

  def __init__(self, match_results):
    """Constructs a Match object.

    Args:
      match_results: Integer tensor of shape [N] with (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.

    Raises:
      ValueError: if match_results does not have rank 1 or is not an
        integer int32 scalar tensor
    """
    if match_results.shape.ndims != 1:
      raise ValueError('match_results should have rank 1')
    if match_results.dtype != tf.int32:
      raise ValueError('match_results should be an int32 or int64 scalar '
                       'tensor')
    self._match_results = match_results

  @property
  def match_results(self):
    """The accessor for match results.

    Returns:
      the tensor which encodes the match results.
    """
    return self._match_results

  def matched_column_indices(self):
    """Returns column indices that match to some row.

    The indices returned by this op are always sorted in increasing order.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return self._reshape_and_cast(tf.where(tf.greater(self._match_results, -1)))

  def matched_column_indicator(self):
    """Returns column indices that are matched.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return tf.greater_equal(self._match_results, 0)

  def num_matched_columns(self):
    """Returns number (int32 scalar tensor) of matched columns."""
    return tf.size(self.matched_column_indices())

  def unmatched_column_indices(self):
    """Returns column indices that do not match any row.

    The indices returned by this op are always sorted in increasing order.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return self._reshape_and_cast(tf.where(tf.equal(self._match_results, -1)))

  def unmatched_column_indicator(self):
    """Returns column indices that are unmatched.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return tf.equal(self._match_results, -1)

  def num_unmatched_columns(self):
    """Returns number (int32 scalar tensor) of unmatched columns."""
    return tf.size(self.unmatched_column_indices())

  def ignored_column_indices(self):
    """Returns column indices that are ignored (neither Matched nor Unmatched).

    The indices returned by this op are always sorted in increasing order.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return self._reshape_and_cast(tf.where(self.ignored_column_indicator()))

  def ignored_column_indicator(self):
    """Returns boolean column indicator where True means the colum is ignored.

    Returns:
      column_indicator: boolean vector which is True for all ignored column
      indices.
    """
    return tf.equal(self._match_results, -2)

  def num_ignored_columns(self):
    """Returns number (int32 scalar tensor) of matched columns."""
    return tf.size(self.ignored_column_indices())

  def unmatched_or_ignored_column_indices(self):
    """Returns column indices that are unmatched or ignored.

    The indices returned by this op are always sorted in increasing order.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return self._reshape_and_cast(tf.where(tf.greater(0, self._match_results)))

  def matched_row_indices(self):
    """Returns row indices that match some column.

    The indices returned by this op are ordered so as to be in correspondence
    with the output of matched_column_indicator().  For example if
    self.matched_column_indicator() is [0,2], and self.matched_row_indices() is
    [7, 3], then we know that column 0 was matched to row 7 and column 2 was
    matched to row 3.

    Returns:
      row_indices: int32 tensor of shape [K] with row indices.
    """
    return self._reshape_and_cast(
        tf.gather(self._match_results, self.matched_column_indices()))

  def _reshape_and_cast(self, t):
    return tf.cast(tf.reshape(t, [-1]), tf.int32)


class Matcher(object):
  """Abstract base class for matcher.
  """
  __metaclass__ = ABCMeta

  def match(self, similarity_matrix, scope=None, **params):
    """Computes matches among row and column indices and returns the result.

    Computes matches among the row and column indices based on the similarity
    matrix and optional arguments.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher value means more similar.
      scope: Op scope name. Defaults to 'Match' if None.
      **params: Additional keyword arguments for specific implementations of
        the Matcher.

    Returns:
      A Match object with the results of matching.
    """
    with tf.name_scope(scope, 'Match', [similarity_matrix, params]) as scope:
      return Match(self._match(similarity_matrix, **params))

  @abstractmethod
  def _match(self, similarity_matrix, **params):
    """Method to be overriden by implementations.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher value means more similar.
      **params: Additional keyword arguments for specific implementations of
        the Matcher.

    Returns:
      match_results: Integer tensor of shape [M]: match_results[i]>=0 means
        that column i is matched to row match_results[i], match_results[i]=-1
        means that the column is not matched. match_results[i]=-2 means that
        the column is ignored (usually this happens when there is a very weak
        match which one neither wants as positive nor negative example).
    """
    pass
