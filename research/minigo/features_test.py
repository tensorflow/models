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
"""Tests for features."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

import features
import go
import numpy as np
import utils_test

tf.logging.set_verbosity(tf.logging.ERROR)

EMPTY_ROW = '.' * utils_test.BOARD_SIZE + '\n'
TEST_BOARD = utils_test.load_board('''
.X.....OO
X........
XXXXXXXXX
''' + EMPTY_ROW * 6)

TEST_POSITION = go.Position(
    utils_test.BOARD_SIZE,
    board=TEST_BOARD,
    n=3,
    komi=6.5,
    caps=(1, 2),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8)),
            go.PlayerMove(go.BLACK, (1, 0))),
    to_play=go.BLACK,
)

TEST_BOARD2 = utils_test.load_board('''
.XOXXOO..
XO.OXOX..
XXO..X...
''' + EMPTY_ROW * 6)

TEST_POSITION2 = go.Position(
    utils_test.BOARD_SIZE,
    board=TEST_BOARD2,
    n=0,
    komi=6.5,
    caps=(0, 0),
    ko=None,
    recent=tuple(),
    to_play=go.BLACK,
)


TEST_POSITION3 = go.Position(utils_test.BOARD_SIZE)
for coord in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)):
  TEST_POSITION3.play_move(coord, mutate=True)
# resulting position should look like this:
# X.XO.....
# .X.......
# .........


class TestFeatureExtraction(utils_test.MiniGoUnitTest):

  def test_stone_features(self):
    f = features.stone_features(utils_test.BOARD_SIZE, TEST_POSITION3)
    self.assertEqual(TEST_POSITION3.to_play, go.WHITE)
    self.assertEqual(f.shape, (9, 9, 16))
    self.assertEqualNPArray(f[:, :, 0], utils_test.load_board('''
      ...X.....
      .........''' + EMPTY_ROW * 7))

    self.assertEqualNPArray(f[:, :, 1], utils_test.load_board('''
      X.X......
      .X.......''' + EMPTY_ROW * 7))

    self.assertEqualNPArray(f[:, :, 2], utils_test.load_board('''
      .X.X.....
      .........''' + EMPTY_ROW * 7))

    self.assertEqualNPArray(f[:, :, 3], utils_test.load_board('''
      X.X......
      .........''' + EMPTY_ROW * 7))

    self.assertEqualNPArray(f[:, :, 4], utils_test.load_board('''
      .X.......
      .........''' + EMPTY_ROW * 7))

    self.assertEqualNPArray(f[:, :, 5], utils_test.load_board('''
      X.X......
      .........''' + EMPTY_ROW * 7))

    for i in range(10, 16):
      self.assertEqualNPArray(
          f[:, :, i], np.zeros([utils_test.BOARD_SIZE, utils_test.BOARD_SIZE]))


if __name__ == '__main__':
  tf.test.main()
