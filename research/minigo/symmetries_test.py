# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for symmetries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import numpy as np
import symmetries
import utils_test

tf.logging.set_verbosity(tf.logging.ERROR)


class TestSymmetryOperations(utils_test.MiniGoUnitTest):

  def setUp(self):
    np.random.seed(1)
    self.feat = np.random.random(
        [utils_test.BOARD_SIZE, utils_test.BOARD_SIZE, 3])
    self.pi = np.random.random([utils_test.BOARD_SIZE ** 2 + 1])
    super().setUp()

  def test_inversions(self):
    for s in symmetries.SYMMETRIES:
      with self.subTest(symmetry=s):
        self.assertEqualNPArray(
            self.feat, symmetries.apply_symmetry_feat(
                s, symmetries.apply_symmetry_feat(
                    symmetries.invert_symmetry(s), self.feat)))
        self.assertEqualNPArray(
            self.feat, symmetries.apply_symmetry_feat(
                symmetries.invert_symmetry(s), symmetries.apply_symmetry_feat(
                    s, self.feat)))

        self.assertEqualNPArray(
            self.pi, symmetries.apply_symmetry_pi(
                utils_test.BOARD_SIZE, s, symmetries.apply_symmetry_pi(
                    utils_test.BOARD_SIZE, symmetries.invert_symmetry(s),
                    self.pi)))
        self.assertEqualNPArray(
            self.pi, symmetries.apply_symmetry_pi(
                utils_test.BOARD_SIZE, symmetries.invert_symmetry(s),
                symmetries.apply_symmetry_pi(
                    utils_test.BOARD_SIZE, s, self.pi)))

  def test_compositions(self):
    test_cases = [
        ('rot90', 'rot90', 'rot180'),
        ('rot90', 'rot180', 'rot270'),
        ('identity', 'rot90', 'rot90'),
        ('fliprot90', 'rot90', 'fliprot180'),
        ('rot90', 'rot270', 'identity'),
    ]
    for s1, s2, composed in test_cases:
      with self.subTest(s1=s1, s2=s2, composed=composed):
        self.assertEqualNPArray(symmetries.apply_symmetry_feat(
            composed, self.feat), symmetries.apply_symmetry_feat(
                s2, symmetries.apply_symmetry_feat(s1, self.feat)))
        self.assertEqualNPArray(
            symmetries.apply_symmetry_pi(
                utils_test.BOARD_SIZE, composed, self.pi),
            symmetries.apply_symmetry_pi(
                utils_test.BOARD_SIZE, s2,
                symmetries.apply_symmetry_pi(
                    utils_test.BOARD_SIZE, s1, self.pi)))

  def test_uniqueness(self):
    all_symmetries_f = [
        symmetries.apply_symmetry_feat(
            s, self.feat) for s in symmetries.SYMMETRIES
    ]
    all_symmetries_pi = [
        symmetries.apply_symmetry_pi(
            utils_test.BOARD_SIZE, s, self.pi) for s in symmetries.SYMMETRIES
    ]
    for f1, f2 in itertools.combinations(all_symmetries_f, 2):
      self.assertNotEqualNPArray(f1, f2)
    for pi1, pi2 in itertools.combinations(all_symmetries_pi, 2):
      self.assertNotEqualNPArray(pi1, pi2)

  def test_proper_move_transform(self):
    # Check that the reinterpretation of 362 = 19*19 + 1 during symmetry
    # application is consistent with coords.from_flat
    move_array = np.arange(utils_test.BOARD_SIZE ** 2 + 1)
    coord_array = np.zeros([utils_test.BOARD_SIZE, utils_test.BOARD_SIZE])
    for c in range(utils_test.BOARD_SIZE ** 2):
      coord_array[coords.from_flat(utils_test.BOARD_SIZE, c)] = c
    for s in symmetries.SYMMETRIES:
      with self.subTest(symmetry=s):
        transformed_moves = symmetries.apply_symmetry_pi(
            utils_test.BOARD_SIZE, s, move_array)
        transformed_board = symmetries.apply_symmetry_feat(s, coord_array)
        for new_coord, old_coord in enumerate(transformed_moves[:-1]):
          self.assertEqual(
              old_coord,
              transformed_board[
                  coords.from_flat(utils_test.BOARD_SIZE, new_coord)])


if __name__ == '__main__':
  tf.test.main()
