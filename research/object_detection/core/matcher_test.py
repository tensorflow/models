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

"""Tests for object_detection.core.matcher."""
import numpy as np
import tensorflow as tf

from object_detection.core import matcher


class AnchorMatcherTest(tf.test.TestCase):

  def test_get_correct_matched_columnIndices(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indices = [0, 1, 3, 5]
    matched_column_indices = match.matched_column_indices()
    self.assertEquals(matched_column_indices.dtype, tf.int32)
    with self.test_session() as sess:
      matched_column_indices = sess.run(matched_column_indices)
      self.assertAllEqual(matched_column_indices, expected_column_indices)

  def test_get_correct_counts(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    exp_num_matched_columns = 4
    exp_num_unmatched_columns = 2
    exp_num_ignored_columns = 1
    num_matched_columns = match.num_matched_columns()
    num_unmatched_columns = match.num_unmatched_columns()
    num_ignored_columns = match.num_ignored_columns()
    self.assertEquals(num_matched_columns.dtype, tf.int32)
    self.assertEquals(num_unmatched_columns.dtype, tf.int32)
    self.assertEquals(num_ignored_columns.dtype, tf.int32)
    with self.test_session() as sess:
      (num_matched_columns_out, num_unmatched_columns_out,
       num_ignored_columns_out) = sess.run(
           [num_matched_columns, num_unmatched_columns, num_ignored_columns])
      self.assertAllEqual(num_matched_columns_out, exp_num_matched_columns)
      self.assertAllEqual(num_unmatched_columns_out, exp_num_unmatched_columns)
      self.assertAllEqual(num_ignored_columns_out, exp_num_ignored_columns)

  def testGetCorrectUnmatchedColumnIndices(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indices = [2, 4]
    unmatched_column_indices = match.unmatched_column_indices()
    self.assertEquals(unmatched_column_indices.dtype, tf.int32)
    with self.test_session() as sess:
      unmatched_column_indices = sess.run(unmatched_column_indices)
      self.assertAllEqual(unmatched_column_indices, expected_column_indices)

  def testGetCorrectMatchedRowIndices(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_row_indices = [3, 1, 0, 5]
    matched_row_indices = match.matched_row_indices()
    self.assertEquals(matched_row_indices.dtype, tf.int32)
    with self.test_session() as sess:
      matched_row_inds = sess.run(matched_row_indices)
      self.assertAllEqual(matched_row_inds, expected_row_indices)

  def test_get_correct_ignored_column_indices(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indices = [6]
    ignored_column_indices = match.ignored_column_indices()
    self.assertEquals(ignored_column_indices.dtype, tf.int32)
    with self.test_session() as sess:
      ignored_column_indices = sess.run(ignored_column_indices)
      self.assertAllEqual(ignored_column_indices, expected_column_indices)

  def test_get_correct_matched_column_indicator(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indicator = [True, True, False, True, False, True, False]
    matched_column_indicator = match.matched_column_indicator()
    self.assertEquals(matched_column_indicator.dtype, tf.bool)
    with self.test_session() as sess:
      matched_column_indicator = sess.run(matched_column_indicator)
      self.assertAllEqual(matched_column_indicator, expected_column_indicator)

  def test_get_correct_unmatched_column_indicator(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indicator = [False, False, True, False, True, False, False]
    unmatched_column_indicator = match.unmatched_column_indicator()
    self.assertEquals(unmatched_column_indicator.dtype, tf.bool)
    with self.test_session() as sess:
      unmatched_column_indicator = sess.run(unmatched_column_indicator)
      self.assertAllEqual(unmatched_column_indicator, expected_column_indicator)

  def test_get_correct_ignored_column_indicator(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indicator = [False, False, False, False, False, False, True]
    ignored_column_indicator = match.ignored_column_indicator()
    self.assertEquals(ignored_column_indicator.dtype, tf.bool)
    with self.test_session() as sess:
      ignored_column_indicator = sess.run(ignored_column_indicator)
      self.assertAllEqual(ignored_column_indicator, expected_column_indicator)

  def test_get_correct_unmatched_ignored_column_indices(self):
    match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
    match = matcher.Match(match_results)
    expected_column_indices = [2, 4, 6]
    unmatched_ignored_column_indices = (match.
                                        unmatched_or_ignored_column_indices())
    self.assertEquals(unmatched_ignored_column_indices.dtype, tf.int32)
    with self.test_session() as sess:
      unmatched_ignored_column_indices = sess.run(
          unmatched_ignored_column_indices)
      self.assertAllEqual(unmatched_ignored_column_indices,
                          expected_column_indices)

  def test_all_columns_accounted_for(self):
    # Note: deliberately setting to small number so not always
    # all possibilities appear (matched, unmatched, ignored)
    num_matches = 10
    match_results = tf.random_uniform(
        [num_matches], minval=-2, maxval=5, dtype=tf.int32)
    match = matcher.Match(match_results)
    matched_column_indices = match.matched_column_indices()
    unmatched_column_indices = match.unmatched_column_indices()
    ignored_column_indices = match.ignored_column_indices()
    with self.test_session() as sess:
      matched, unmatched, ignored = sess.run([
          matched_column_indices, unmatched_column_indices,
          ignored_column_indices
      ])
      all_indices = np.hstack((matched, unmatched, ignored))
      all_indices_sorted = np.sort(all_indices)
      self.assertAllEqual(all_indices_sorted,
                          np.arange(num_matches, dtype=np.int32))


if __name__ == '__main__':
  tf.test.main()
