# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for box_matcher.py."""

import tensorflow as tf, tf_keras

from official.vision.ops import box_matcher


class BoxMatcherTest(tf.test.TestCase):

  def test_box_matcher_unbatched(self):
    sim_matrix = tf.constant(
        [[0.04, 0, 0, 0],
         [0, 0, 1., 0]],
        dtype=tf.float32)

    fg_threshold = 0.5
    bg_thresh_hi = 0.2
    bg_thresh_lo = 0.0

    matcher = box_matcher.BoxMatcher(
        thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
        indicators=[-3, -2, -1, 1])
    match_indices, match_indicators = matcher(sim_matrix)
    positive_matches = tf.greater_equal(match_indicators, 0)
    negative_matches = tf.equal(match_indicators, -2)

    self.assertAllEqual(
        positive_matches.numpy(), [False, True])
    self.assertAllEqual(
        negative_matches.numpy(), [True, False])
    self.assertAllEqual(
        match_indices.numpy(), [0, 2])
    self.assertAllEqual(
        match_indicators.numpy(), [-2, 1])

  def test_box_matcher_batched(self):
    sim_matrix = tf.constant(
        [[[0.04, 0, 0, 0],
          [0, 0, 1., 0]]],
        dtype=tf.float32)

    fg_threshold = 0.5
    bg_thresh_hi = 0.2
    bg_thresh_lo = 0.0

    matcher = box_matcher.BoxMatcher(
        thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
        indicators=[-3, -2, -1, 1])
    match_indices, match_indicators = matcher(sim_matrix)
    positive_matches = tf.greater_equal(match_indicators, 0)
    negative_matches = tf.equal(match_indicators, -2)

    self.assertAllEqual(
        positive_matches.numpy(), [[False, True]])
    self.assertAllEqual(
        negative_matches.numpy(), [[True, False]])
    self.assertAllEqual(
        match_indices.numpy(), [[0, 2]])
    self.assertAllEqual(
        match_indicators.numpy(), [[-2, 1]])


if __name__ == '__main__':
  tf.test.main()
