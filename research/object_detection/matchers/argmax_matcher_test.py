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

"""Tests for object_detection.matchers.argmax_matcher."""

import numpy as np
import tensorflow as tf

from object_detection.matchers import argmax_matcher


class ArgMaxMatcherTest(tf.test.TestCase):

  def test_return_correct_matches_with_default_thresholds(self):
    similarity = np.array([[1., 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]])

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=None)
    expected_matched_rows = np.array([2, 0, 1, 0, 1])

    sim = tf.constant(similarity)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    matched_rows = match.matched_row_indices()
    unmatched_cols = match.unmatched_column_indices()

    with self.test_session() as sess:
      res_matched_cols = sess.run(matched_cols)
      res_matched_rows = sess.run(matched_rows)
      res_unmatched_cols = sess.run(unmatched_cols)

    self.assertAllEqual(res_matched_rows, expected_matched_rows)
    self.assertAllEqual(res_matched_cols, np.arange(similarity.shape[1]))
    self.assertEmpty(res_unmatched_cols)

  def test_return_correct_matches_with_empty_rows(self):

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=None)
    sim = 0.2*tf.ones([0, 5])
    match = matcher.match(sim)
    unmatched_cols = match.unmatched_column_indices()

    with self.test_session() as sess:
      res_unmatched_cols = sess.run(unmatched_cols)
      self.assertAllEqual(res_unmatched_cols, np.arange(5))

  def test_return_correct_matches_with_matched_threshold(self):
    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.int32)

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3)
    expected_matched_cols = np.array([0, 3, 4])
    expected_matched_rows = np.array([2, 0, 1])
    expected_unmatched_cols = np.array([1, 2])

    sim = tf.constant(similarity)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    matched_rows = match.matched_row_indices()
    unmatched_cols = match.unmatched_column_indices()

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      res_matched_cols = sess.run(matched_cols)
      res_matched_rows = sess.run(matched_rows)
      res_unmatched_cols = sess.run(unmatched_cols)

    self.assertAllEqual(res_matched_rows, expected_matched_rows)
    self.assertAllEqual(res_matched_cols, expected_matched_cols)
    self.assertAllEqual(res_unmatched_cols, expected_unmatched_cols)

  def test_return_correct_matches_with_matched_and_unmatched_threshold(self):
    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.int32)

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3,
                                           unmatched_threshold=2)
    expected_matched_cols = np.array([0, 3, 4])
    expected_matched_rows = np.array([2, 0, 1])
    expected_unmatched_cols = np.array([1])  # col 2 has too high maximum val

    sim = tf.constant(similarity)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    matched_rows = match.matched_row_indices()
    unmatched_cols = match.unmatched_column_indices()

    with self.test_session() as sess:
      res_matched_cols = sess.run(matched_cols)
      res_matched_rows = sess.run(matched_rows)
      res_unmatched_cols = sess.run(unmatched_cols)

    self.assertAllEqual(res_matched_rows, expected_matched_rows)
    self.assertAllEqual(res_matched_cols, expected_matched_cols)
    self.assertAllEqual(res_unmatched_cols, expected_unmatched_cols)

  def test_return_correct_matches_negatives_lower_than_unmatched_false(self):
    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.int32)

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3,
                                           unmatched_threshold=2,
                                           negatives_lower_than_unmatched=False)
    expected_matched_cols = np.array([0, 3, 4])
    expected_matched_rows = np.array([2, 0, 1])
    expected_unmatched_cols = np.array([2])  # col 1 has too low maximum val

    sim = tf.constant(similarity)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    matched_rows = match.matched_row_indices()
    unmatched_cols = match.unmatched_column_indices()

    with self.test_session() as sess:
      res_matched_cols = sess.run(matched_cols)
      res_matched_rows = sess.run(matched_rows)
      res_unmatched_cols = sess.run(unmatched_cols)

    self.assertAllEqual(res_matched_rows, expected_matched_rows)
    self.assertAllEqual(res_matched_cols, expected_matched_cols)
    self.assertAllEqual(res_unmatched_cols, expected_unmatched_cols)

  def test_return_correct_matches_unmatched_row_not_using_force_match(self):
    similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [3, 0, -1, 2, 0]], dtype=np.int32)

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3,
                                           unmatched_threshold=2)
    expected_matched_cols = np.array([0, 3])
    expected_matched_rows = np.array([2, 0])
    expected_unmatched_cols = np.array([1, 2, 4])

    sim = tf.constant(similarity)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    matched_rows = match.matched_row_indices()
    unmatched_cols = match.unmatched_column_indices()

    with self.test_session() as sess:
      res_matched_cols = sess.run(matched_cols)
      res_matched_rows = sess.run(matched_rows)
      res_unmatched_cols = sess.run(unmatched_cols)

    self.assertAllEqual(res_matched_rows, expected_matched_rows)
    self.assertAllEqual(res_matched_cols, expected_matched_cols)
    self.assertAllEqual(res_unmatched_cols, expected_unmatched_cols)

  def test_return_correct_matches_unmatched_row_while_using_force_match(self):
    similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [3, 0, -1, 2, 0]], dtype=np.int32)

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3,
                                           unmatched_threshold=2,
                                           force_match_for_each_row=True)
    expected_matched_cols = np.array([0, 1, 3])
    expected_matched_rows = np.array([2, 1, 0])
    expected_unmatched_cols = np.array([2, 4])  # col 2 has too high max val

    sim = tf.constant(similarity)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    matched_rows = match.matched_row_indices()
    unmatched_cols = match.unmatched_column_indices()

    with self.test_session() as sess:
      res_matched_cols = sess.run(matched_cols)
      res_matched_rows = sess.run(matched_rows)
      res_unmatched_cols = sess.run(unmatched_cols)

    self.assertAllEqual(res_matched_rows, expected_matched_rows)
    self.assertAllEqual(res_matched_cols, expected_matched_cols)
    self.assertAllEqual(res_unmatched_cols, expected_unmatched_cols)

  def test_valid_arguments_corner_case(self):
    argmax_matcher.ArgMaxMatcher(matched_threshold=1,
                                 unmatched_threshold=1)

  def test_invalid_arguments_corner_case_negatives_lower_than_thres_false(self):
    with self.assertRaises(ValueError):
      argmax_matcher.ArgMaxMatcher(matched_threshold=1,
                                   unmatched_threshold=1,
                                   negatives_lower_than_unmatched=False)

  def test_invalid_arguments_no_matched_threshold(self):
    with self.assertRaises(ValueError):
      argmax_matcher.ArgMaxMatcher(matched_threshold=None,
                                   unmatched_threshold=4)

  def test_invalid_arguments_unmatched_thres_larger_than_matched_thres(self):
    with self.assertRaises(ValueError):
      argmax_matcher.ArgMaxMatcher(matched_threshold=1,
                                   unmatched_threshold=2)

  def test_set_values_using_indicator(self):
    input_a = np.array([3, 4, 5, 1, 4, 3, 2])
    expected_b = np.array([3, 0, 0, 1, 0, 3, 2])  # Set a>3 to 0
    expected_c = np.array(
        [3., 4., 5., -1., 4., 3., -1.])  # Set a<3 to -1. Float32
    idxb_ = input_a > 3
    idxc_ = input_a < 3

    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=None)

    a = tf.constant(input_a)
    idxb = tf.constant(idxb_)
    idxc = tf.constant(idxc_)
    b = matcher._set_values_using_indicator(a, idxb, 0)
    c = matcher._set_values_using_indicator(tf.cast(a, tf.float32), idxc, -1)
    with self.test_session() as sess:
      res_b = sess.run(b)
      res_c = sess.run(c)
      self.assertAllEqual(res_b, expected_b)
      self.assertAllEqual(res_c, expected_c)


if __name__ == '__main__':
  tf.test.main()
