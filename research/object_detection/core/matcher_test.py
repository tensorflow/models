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
import tensorflow.compat.v1 as tf

from object_detection.core import matcher
from object_detection.utils import test_case


class MatchTest(test_case.TestCase):

  def test_get_correct_matched_columnIndices(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      matched_column_indices = match.matched_column_indices()
      return matched_column_indices
    expected_column_indices = [0, 1, 3, 5]
    matched_column_indices = self.execute(graph_fn, [])
    self.assertAllEqual(matched_column_indices, expected_column_indices)

  def test_get_correct_counts(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 1, -2])
      match = matcher.Match(match_results)
      num_matched_columns = match.num_matched_columns()
      num_unmatched_columns = match.num_unmatched_columns()
      num_ignored_columns = match.num_ignored_columns()
      num_matched_rows = match.num_matched_rows()
      return [num_matched_columns, num_unmatched_columns, num_ignored_columns,
              num_matched_rows]
    (num_matched_columns_out, num_unmatched_columns_out,
     num_ignored_columns_out,
     num_matched_rows_out) = self.execute_cpu(graph_fn, [])
    exp_num_matched_columns = 4
    exp_num_unmatched_columns = 2
    exp_num_ignored_columns = 1
    exp_num_matched_rows = 3
    self.assertAllEqual(num_matched_columns_out, exp_num_matched_columns)
    self.assertAllEqual(num_unmatched_columns_out, exp_num_unmatched_columns)
    self.assertAllEqual(num_ignored_columns_out, exp_num_ignored_columns)
    self.assertAllEqual(num_matched_rows_out, exp_num_matched_rows)

  def testGetCorrectUnmatchedColumnIndices(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      unmatched_column_indices = match.unmatched_column_indices()
      return unmatched_column_indices
    unmatched_column_indices = self.execute(graph_fn, [])
    expected_column_indices = [2, 4]
    self.assertAllEqual(unmatched_column_indices, expected_column_indices)

  def testGetCorrectMatchedRowIndices(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      matched_row_indices = match.matched_row_indices()
      return matched_row_indices
    matched_row_indices = self.execute(graph_fn, [])
    expected_row_indices = [3, 1, 0, 5]
    self.assertAllEqual(matched_row_indices, expected_row_indices)

  def test_get_correct_ignored_column_indices(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      ignored_column_indices = match.ignored_column_indices()
      return ignored_column_indices
    ignored_column_indices = self.execute(graph_fn, [])
    expected_column_indices = [6]
    self.assertAllEqual(ignored_column_indices, expected_column_indices)

  def test_get_correct_matched_column_indicator(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      matched_column_indicator = match.matched_column_indicator()
      return matched_column_indicator
    expected_column_indicator = [True, True, False, True, False, True, False]
    matched_column_indicator = self.execute(graph_fn, [])
    self.assertAllEqual(matched_column_indicator, expected_column_indicator)

  def test_get_correct_unmatched_column_indicator(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      unmatched_column_indicator = match.unmatched_column_indicator()
      return unmatched_column_indicator
    expected_column_indicator = [False, False, True, False, True, False, False]
    unmatched_column_indicator = self.execute(graph_fn, [])
    self.assertAllEqual(unmatched_column_indicator, expected_column_indicator)

  def test_get_correct_ignored_column_indicator(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      ignored_column_indicator = match.ignored_column_indicator()
      return ignored_column_indicator
    expected_column_indicator = [False, False, False, False, False, False, True]
    ignored_column_indicator = self.execute(graph_fn, [])
    self.assertAllEqual(ignored_column_indicator, expected_column_indicator)

  def test_get_correct_unmatched_ignored_column_indices(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      match = matcher.Match(match_results)
      unmatched_ignored_column_indices = (match.
                                          unmatched_or_ignored_column_indices())
      return unmatched_ignored_column_indices
    expected_column_indices = [2, 4, 6]
    unmatched_ignored_column_indices = self.execute(graph_fn, [])
    self.assertAllEqual(unmatched_ignored_column_indices,
                        expected_column_indices)

  def test_all_columns_accounted_for(self):
    # Note: deliberately setting to small number so not always
    # all possibilities appear (matched, unmatched, ignored)
    def graph_fn():
      match_results = tf.random_uniform(
          [num_matches], minval=-2, maxval=5, dtype=tf.int32)
      match = matcher.Match(match_results)
      matched_column_indices = match.matched_column_indices()
      unmatched_column_indices = match.unmatched_column_indices()
      ignored_column_indices = match.ignored_column_indices()
      return (matched_column_indices, unmatched_column_indices,
              ignored_column_indices)
    num_matches = 10
    matched, unmatched, ignored = self.execute(graph_fn, [])
    all_indices = np.hstack((matched, unmatched, ignored))
    all_indices_sorted = np.sort(all_indices)
    self.assertAllEqual(all_indices_sorted,
                        np.arange(num_matches, dtype=np.int32))

  def test_scalar_gather_based_on_match(self):
    def graph_fn():
      match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
      input_tensor = tf.constant([0, 1, 2, 3, 4, 5, 6, 7], dtype=tf.float32)
      match = matcher.Match(match_results)
      gathered_tensor = match.gather_based_on_match(input_tensor,
                                                    unmatched_value=100.,
                                                    ignored_value=200.)
      return gathered_tensor
    expected_gathered_tensor = [3, 1, 100, 0, 100, 5, 200]
    gathered_tensor_out = self.execute(graph_fn, [])
    self.assertAllEqual(expected_gathered_tensor, gathered_tensor_out)

  def test_multidimensional_gather_based_on_match(self):
    def graph_fn():
      match_results = tf.constant([1, -1, -2])
      input_tensor = tf.constant([[0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]],
                                 dtype=tf.float32)
      match = matcher.Match(match_results)
      gathered_tensor = match.gather_based_on_match(input_tensor,
                                                    unmatched_value=tf.zeros(4),
                                                    ignored_value=tf.zeros(4))
      return gathered_tensor
    expected_gathered_tensor = [[0, 0, 0.5, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]]
    gathered_tensor_out = self.execute(graph_fn, [])
    self.assertAllEqual(expected_gathered_tensor, gathered_tensor_out)

  def test_multidimensional_gather_based_on_match_with_matmul_gather_op(self):
    def graph_fn():
      match_results = tf.constant([1, -1, -2])
      input_tensor = tf.constant([[0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]],
                                 dtype=tf.float32)
      match = matcher.Match(match_results, use_matmul_gather=True)
      gathered_tensor = match.gather_based_on_match(input_tensor,
                                                    unmatched_value=tf.zeros(4),
                                                    ignored_value=tf.zeros(4))
      return gathered_tensor
    expected_gathered_tensor = [[0, 0, 0.5, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]]
    gathered_tensor_out = self.execute(graph_fn, [])
    self.assertAllEqual(expected_gathered_tensor, gathered_tensor_out)

if __name__ == '__main__':
  tf.test.main()
