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

"""Tests for object_detection.core.bipartite_matcher."""

import numpy as np
import tensorflow as tf

from object_detection.matchers import bipartite_matcher
from object_detection.utils import test_case


class GreedyBipartiteMatcherTest(test_case.TestCase):

  def test_get_expected_matches_when_all_rows_are_valid(self):
    similarity_matrix = np.array([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]],
                                 dtype=np.float32)
    valid_rows = np.ones([2], dtype=np.bool)
    expected_match_results = [-1, 1, 0]
    def graph_fn(similarity_matrix, valid_rows):
      matcher = bipartite_matcher.GreedyBipartiteMatcher()
      match = matcher.match(similarity_matrix, valid_rows=valid_rows)
      return match._match_results
    match_results_out = self.execute(graph_fn, [similarity_matrix, valid_rows])
    self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_all_rows_be_default(self):
    similarity_matrix = np.array([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]],
                                 dtype=np.float32)
    expected_match_results = [-1, 1, 0]
    def graph_fn(similarity_matrix):
      matcher = bipartite_matcher.GreedyBipartiteMatcher()
      match = matcher.match(similarity_matrix)
      return match._match_results
    match_results_out = self.execute(graph_fn, [similarity_matrix])
    self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_no_matches_with_zero_valid_rows(self):
    similarity_matrix = np.array([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]],
                                 dtype=np.float32)
    valid_rows = np.zeros([2], dtype=np.bool)
    expected_match_results = [-1, -1, -1]
    def graph_fn(similarity_matrix, valid_rows):
      matcher = bipartite_matcher.GreedyBipartiteMatcher()
      match = matcher.match(similarity_matrix, valid_rows=valid_rows)
      return match._match_results
    match_results_out = self.execute(graph_fn, [similarity_matrix, valid_rows])
    self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_only_one_valid_row(self):
    similarity_matrix = np.array([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]],
                                 dtype=np.float32)
    valid_rows = np.array([True, False], dtype=np.bool)
    expected_match_results = [-1, -1, 0]
    def graph_fn(similarity_matrix, valid_rows):
      matcher = bipartite_matcher.GreedyBipartiteMatcher()
      match = matcher.match(similarity_matrix, valid_rows=valid_rows)
      return match._match_results
    match_results_out = self.execute(graph_fn, [similarity_matrix, valid_rows])
    self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_only_one_valid_row_at_bottom(self):
    similarity_matrix = np.array([[0.15, 0.2, 0.3], [0.50, 0.1, 0.8]],
                                 dtype=np.float32)
    valid_rows = np.array([False, True], dtype=np.bool)
    expected_match_results = [-1, -1, 0]
    def graph_fn(similarity_matrix, valid_rows):
      matcher = bipartite_matcher.GreedyBipartiteMatcher()
      match = matcher.match(similarity_matrix, valid_rows=valid_rows)
      return match._match_results
    match_results_out = self.execute(graph_fn, [similarity_matrix, valid_rows])
    self.assertAllEqual(match_results_out, expected_match_results)


if __name__ == '__main__':
  tf.test.main()
