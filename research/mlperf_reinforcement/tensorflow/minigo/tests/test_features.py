# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import numpy as np

import features
import go
from tests import test_utils

EMPTY_ROW = '.' * go.N + '\n'
TEST_BOARD = test_utils.load_board('''
.X.....OO
X........
XXXXXXXXX
''' + EMPTY_ROW * 6)

TEST_POSITION = go.Position(
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

TEST_BOARD2 = test_utils.load_board('''
.XOXXOO..
XO.OXOX..
XXO..X...
''' + EMPTY_ROW * 6)

TEST_POSITION2 = go.Position(
    board=TEST_BOARD2,
    n=0,
    komi=6.5,
    caps=(0, 0),
    ko=None,
    recent=tuple(),
    to_play=go.BLACK,
)


TEST_POSITION3 = go.Position()
for coord in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)):
    TEST_POSITION3.play_move(coord, mutate=True)
# resulting position should look like this:
# X.XO.....
# .X.......
# .........


class TestFeatureExtraction(test_utils.MiniGoUnitTest):
    def test_stone_features(self):
        f = features.stone_features(TEST_POSITION3)
        self.assertEqual(TEST_POSITION3.to_play, go.WHITE)
        self.assertEqual(f.shape, (9, 9, 16))
        self.assertEqualNPArray(f[:, :, 0], test_utils.load_board('''
            ...X.....
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 1], test_utils.load_board('''
            X.X......
            .X.......''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 2], test_utils.load_board('''
            .X.X.....
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 3], test_utils.load_board('''
            X.X......
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 4], test_utils.load_board('''
            .X.......
            .........''' + EMPTY_ROW * 7))

        self.assertEqualNPArray(f[:, :, 5], test_utils.load_board('''
            X.X......
            .........''' + EMPTY_ROW * 7))

        for i in range(10, 16):
            self.assertEqualNPArray(f[:, :, i], np.zeros([go.N, go.N]))

    def test_stone_color_feature(self):
        f = features.stone_color_feature(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, 3))
        # plane 0 is B
        self.assertEqual(f[0, 1, 0], 1)
        self.assertEqual(f[0, 1, 1], 0)
        # plane 1 is W
        self.assertEqual(f[0, 8, 1], 1)
        self.assertEqual(f[0, 8, 0], 0)
        # plane 2 is empty
        self.assertEqual(f[0, 5, 2], 1)
        self.assertEqual(f[0, 5, 1], 0)

    def test_liberty_feature(self):
        f = features.liberty_feature(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, features.liberty_feature.planes))

        self.assertEqual(f[0, 0, 0], 0)
        # the stone at 0, 1 has 3 liberties.
        self.assertEqual(f[0, 1, 2], 1)
        self.assertEqual(f[0, 1, 4], 0)
        # the group at 0, 7 has 3 liberties
        self.assertEqual(f[0, 7, 2], 1)
        self.assertEqual(f[0, 8, 2], 1)
        # the group at 1, 0 has 18 liberties
        self.assertEqual(f[1, 0, 7], 1)

    def test_recent_moves_feature(self):
        f = features.recent_move_feature(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, features.recent_move_feature.planes))
        # most recent move at (1, 0)
        self.assertEqual(f[1, 0, 0], 1)
        self.assertEqual(f[1, 0, 3], 0)
        # second most recent move at (0, 8)
        self.assertEqual(f[0, 8, 1], 1)
        self.assertEqual(f[0, 8, 0], 0)
        # third most recent move at (0, 1)
        self.assertEqual(f[0, 1, 2], 1)
        # no more older moves
        self.assertEqualNPArray(f[:, :, 3], np.zeros([9, 9]))
        self.assertEqualNPArray(
            f[:, :, features.recent_move_feature.planes - 1], np.zeros([9, 9]))

    def test_would_capture_feature(self):
        f = features.would_capture_feature(TEST_POSITION2)
        self.assertEqual(
            f.shape, (9, 9, features.would_capture_feature.planes))
        # move at (1, 2) would capture 2 stones
        self.assertEqual(f[1, 2, 1], 1)
        # move at (0, 0) should not capture stones because it's B's move.
        self.assertEqual(f[0, 0, 0], 0)
        # move at (0, 7) would capture 3 stones
        self.assertEqual(f[0, 7, 2], 1)
        self.assertEqual(f[0, 7, 1], 0)
