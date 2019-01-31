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

import tensorflow as tf

from object_detection.matchers import bipartite_matcher


class GreedyBipartiteMatcherTest(tf.test.TestCase):

  def test_get_expected_matches_when_all_rows_are_valid(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    valid_rows = tf.ones([2], dtype=tf.bool)
    expected_match_results = [-1, 1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, valid_rows=valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_all_rows_be_default(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    expected_match_results = [-1, 1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_no_matches_with_zero_valid_rows(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    valid_rows = tf.zeros([2], dtype=tf.bool)
    expected_match_results = [-1, -1, -1]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_only_one_valid_row(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    valid_rows = tf.constant([True, False], dtype=tf.bool)
    expected_match_results = [-1, -1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_only_one_valid_row_at_bottom(self):
    similarity_matrix = tf.constant([[0.15, 0.2, 0.3], [0.50, 0.1, 0.8]])
    valid_rows = tf.constant([False, True], dtype=tf.bool)
    expected_match_results = [-1, -1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)


if __name__ == '__main__':
  tf.test.main()
