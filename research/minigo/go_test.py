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
"""Tests for go."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import go
from go import Position, PlayerMove, LibertyTracker, WHITE, BLACK
import numpy as np
import sgf_wrapper
import utils_test

EMPTY_ROW = '.' * utils_test.BOARD_SIZE + '\n'
TEST_BOARD = utils_test.load_board('''
.X.....OO
X........
''' + EMPTY_ROW * 7)

NO_HANDICAP_SGF = '''(;CA[UTF-8]SZ[9]PB[Murakawa Daisuke]PW[Iyama Yuta]KM[6.5]
                   HA[0]RE[W+1.5]GM[1];B[fd];W[cf];B[eg];W[dd];B[dc];W[cc];
                   B[de];W[cd];B[ed];W[he];B[ce];W[be];B[df];W[bf];B[hd];
                   W[ge];B[gd];W[gg];B[db];W[cb];B[cg];W[bg];B[gh];W[fh];
                   B[hh];W[fg];B[eh];W[ei];B[di];W[fi];B[hg];W[dh];B[ch];
                   W[ci];B[bh];W[ff];B[fe];W[hf];B[id];W[bi];B[ah];W[ef];
                   B[dg];W[ee];B[di];W[ig];B[ai];W[ih];B[fb];W[hi];B[ag];
                   W[ab];B[bd];W[bc];B[ae];W[ad];B[af];W[bd];B[ca];W[ba];
                   B[da];W[ie])'''


def coords_from_kgs_set(string):
  def _from_kgs(kgsc):
    return coords.from_kgs(utils_test.BOARD_SIZE, kgsc)
  return frozenset(map(_from_kgs, string.split()))


class TestBasicFunctions(utils_test.MiniGoUnitTest):

  def test_load_board(self):
    self.assertEqualNPArray(utils_test.EMPTY_BOARD, np.zeros(
        [utils_test.BOARD_SIZE, utils_test.BOARD_SIZE]))
    self.assertEqualNPArray(
        utils_test.EMPTY_BOARD, utils_test.load_board(
            '. \n' * utils_test.BOARD_SIZE ** 2))

  def test_neighbors(self):
    corner = coords.from_kgs(utils_test.BOARD_SIZE, 'A1')
    neighbors = [
        utils_test.EMPTY_BOARD[c] for c in utils_test.NEIGHBORS[corner]]
    self.assertEqual(len(neighbors), 2)

    side = coords.from_kgs(utils_test.BOARD_SIZE, 'A2')
    side_neighbors = [
        utils_test.EMPTY_BOARD[c] for c in utils_test.NEIGHBORS[side]]
    self.assertEqual(len(side_neighbors), 3)

  def test_is_koish(self):
    self.assertEqual(go.is_koish(
        utils_test.BOARD_SIZE, TEST_BOARD, coords.from_kgs(
            utils_test.BOARD_SIZE, 'A9')), BLACK)
    self.assertEqual(go.is_koish(
        utils_test.BOARD_SIZE, TEST_BOARD, coords.from_kgs(
            utils_test.BOARD_SIZE, 'B8')), None)
    self.assertEqual(go.is_koish(
        utils_test.BOARD_SIZE, TEST_BOARD, coords.from_kgs(
            utils_test.BOARD_SIZE, 'B9')), None)
    self.assertEqual(
        go.is_koish(utils_test.BOARD_SIZE, TEST_BOARD, coords.from_kgs(
            utils_test.BOARD_SIZE, 'E5')), None)

  def test_is_eyeish(self):
    board = utils_test.load_board('''
      .XX...XXX
      X.X...X.X
      XX.....X.
      ........X
      XXXX.....
      OOOX....O
      X.OXX.OO.
      .XO.X.O.O
      XXO.X.OO.
    ''')
    B_eyes = coords_from_kgs_set('A2 A9 B8 J7 H8')
    W_eyes = coords_from_kgs_set('H2 J1 J3')
    not_eyes = coords_from_kgs_set('B3 E5')
    for be in B_eyes:
      self.assertEqual(go.is_eyeish(
          utils_test.BOARD_SIZE, board, be), BLACK, str(be))
    for we in W_eyes:
      self.assertEqual(go.is_eyeish(
          utils_test.BOARD_SIZE, board, we), WHITE, str(we))
    for ne in not_eyes:
      self.assertEqual(go.is_eyeish(
          utils_test.BOARD_SIZE, board, ne), None, str(ne))


class TestLibertyTracker(utils_test.MiniGoUnitTest):

  def test_lib_tracker_init(self):
    board = utils_test.load_board('X........' + EMPTY_ROW * 8)

    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    self.assertEqual(len(lib_tracker.groups), 1)
    self.assertNotEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'A9')], go.MISSING_GROUP_ID)
    self.assertEqual(lib_tracker.liberty_cache[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')], 2)
    sole_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')]]
    self.assertEqual(sole_group.stones, coords_from_kgs_set('A9'))
    self.assertEqual(sole_group.liberties, coords_from_kgs_set('B9 A8'))
    self.assertEqual(sole_group.color, BLACK)

  def test_place_stone(self):
    board = utils_test.load_board('X........' + EMPTY_ROW * 8)
    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    lib_tracker.add_stone(BLACK, coords.from_kgs(utils_test.BOARD_SIZE, 'B9'))
    self.assertEqual(len(lib_tracker.groups), 1)
    self.assertNotEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'A9')], go.MISSING_GROUP_ID)
    self.assertEqual(lib_tracker.liberty_cache[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')], 3)
    self.assertEqual(lib_tracker.liberty_cache[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B9')], 3)
    sole_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')]]
    self.assertEqual(sole_group.stones, coords_from_kgs_set('A9 B9'))
    self.assertEqual(sole_group.liberties,
                     coords_from_kgs_set('C9 A8 B8'))
    self.assertEqual(sole_group.color, BLACK)

  def test_place_stone_opposite_color(self):
    board = utils_test.load_board('X........' + EMPTY_ROW * 8)
    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    lib_tracker.add_stone(WHITE, coords.from_kgs(utils_test.BOARD_SIZE, 'B9'))
    self.assertEqual(len(lib_tracker.groups), 2)
    self.assertNotEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'A9')], go.MISSING_GROUP_ID)
    self.assertNotEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'B9')], go.MISSING_GROUP_ID)
    self.assertEqual(lib_tracker.liberty_cache[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')], 1)
    self.assertEqual(lib_tracker.liberty_cache[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B9')], 2)
    black_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')]]
    white_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B9')]]
    self.assertEqual(black_group.stones, coords_from_kgs_set('A9'))
    self.assertEqual(black_group.liberties, coords_from_kgs_set('A8'))
    self.assertEqual(black_group.color, BLACK)
    self.assertEqual(white_group.stones, coords_from_kgs_set('B9'))
    self.assertEqual(white_group.liberties, coords_from_kgs_set('C9 B8'))
    self.assertEqual(white_group.color, WHITE)

  def test_merge_multiple_groups(self):
    board = utils_test.load_board('''
      .X.......
      X.X......
      .X.......
    ''' + EMPTY_ROW * 6)
    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    lib_tracker.add_stone(BLACK, coords.from_kgs(utils_test.BOARD_SIZE, 'B8'))
    self.assertEqual(len(lib_tracker.groups), 1)
    self.assertNotEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'B8')], go.MISSING_GROUP_ID)
    sole_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B8')]]
    self.assertEqual(sole_group.stones,
                     coords_from_kgs_set('B9 A8 B8 C8 B7'))
    self.assertEqual(sole_group.liberties,
                     coords_from_kgs_set('A9 C9 D8 A7 C7 B6'))
    self.assertEqual(sole_group.color, BLACK)

    liberty_cache = lib_tracker.liberty_cache
    for stone in sole_group.stones:
      self.assertEqual(liberty_cache[stone], 6, str(stone))

  def test_capture_stone(self):
    board = utils_test.load_board('''
      .X.......
      XO.......
      .X.......
    ''' + EMPTY_ROW * 6)
    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    captured = lib_tracker.add_stone(BLACK, coords.from_kgs(
        utils_test.BOARD_SIZE, 'C8'))
    self.assertEqual(len(lib_tracker.groups), 4)
    self.assertEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'B8')], go.MISSING_GROUP_ID)
    self.assertEqual(captured, coords_from_kgs_set('B8'))

  def test_capture_many(self):
    board = utils_test.load_board('''
      .XX......
      XOO......
      .XX......
    ''' + EMPTY_ROW * 6)
    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    captured = lib_tracker.add_stone(BLACK, coords.from_kgs(
        utils_test.BOARD_SIZE, 'D8'))
    self.assertEqual(len(lib_tracker.groups), 4)
    self.assertEqual(
        lib_tracker.group_index[coords.from_kgs(
            utils_test.BOARD_SIZE, 'B8')], go.MISSING_GROUP_ID)
    self.assertEqual(captured, coords_from_kgs_set('B8 C8'))

    left_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A8')]]
    self.assertEqual(left_group.stones, coords_from_kgs_set('A8'))
    self.assertEqual(left_group.liberties,
                     coords_from_kgs_set('A9 B8 A7'))

    right_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'D8')]]
    self.assertEqual(right_group.stones, coords_from_kgs_set('D8'))
    self.assertEqual(right_group.liberties,
                     coords_from_kgs_set('D9 C8 E8 D7'))

    top_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B9')]]
    self.assertEqual(top_group.stones, coords_from_kgs_set('B9 C9'))
    self.assertEqual(top_group.liberties,
                     coords_from_kgs_set('A9 D9 B8 C8'))

    bottom_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B7')]]
    self.assertEqual(bottom_group.stones, coords_from_kgs_set('B7 C7'))
    self.assertEqual(bottom_group.liberties,
                     coords_from_kgs_set('B8 C8 A7 D7 B6 C6'))

    liberty_cache = lib_tracker.liberty_cache
    for stone in top_group.stones:
      self.assertEqual(liberty_cache[stone], 4, str(stone))
    for stone in left_group.stones:
      self.assertEqual(liberty_cache[stone], 3, str(stone))
    for stone in right_group.stones:
      self.assertEqual(liberty_cache[stone], 4, str(stone))
    for stone in bottom_group.stones:
      self.assertEqual(liberty_cache[stone], 6, str(stone))
    for stone in captured:
      self.assertEqual(liberty_cache[stone], 0, str(stone))

  def test_capture_multiple_groups(self):
    board = utils_test.load_board('''
      .OX......
      OXX......
      XX.......
    ''' + EMPTY_ROW * 6)
    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    captured = lib_tracker.add_stone(BLACK, coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9'))
    self.assertEqual(len(lib_tracker.groups), 2)
    self.assertEqual(captured, coords_from_kgs_set('B9 A8'))

    corner_stone = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')]]
    self.assertEqual(corner_stone.stones, coords_from_kgs_set('A9'))
    self.assertEqual(corner_stone.liberties, coords_from_kgs_set('B9 A8'))

    surrounding_stones = lib_tracker.groups[lib_tracker.group_index[
        coords.from_kgs(utils_test.BOARD_SIZE, 'C9')]]
    self.assertEqual(surrounding_stones.stones,
                     coords_from_kgs_set('C9 B8 C8 A7 B7'))
    self.assertEqual(surrounding_stones.liberties,
                     coords_from_kgs_set('B9 D9 A8 D8 C7 A6 B6'))

    liberty_cache = lib_tracker.liberty_cache
    for stone in corner_stone.stones:
      self.assertEqual(liberty_cache[stone], 2, str(stone))
    for stone in surrounding_stones.stones:
      self.assertEqual(liberty_cache[stone], 7, str(stone))

  def test_same_friendly_group_neighboring_twice(self):
    board = utils_test.load_board('''
      XX.......
      X........
    ''' + EMPTY_ROW * 7)

    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    captured = lib_tracker.add_stone(BLACK, coords.from_kgs(
        utils_test.BOARD_SIZE, 'B8'))
    self.assertEqual(len(lib_tracker.groups), 1)
    sole_group_id = lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')]
    sole_group = lib_tracker.groups[sole_group_id]
    self.assertEqual(sole_group.stones,
                     coords_from_kgs_set('A9 B9 A8 B8'))
    self.assertEqual(sole_group.liberties,
                     coords_from_kgs_set('C9 C8 A7 B7'))
    self.assertEqual(captured, set())

  def test_same_opponent_group_neighboring_twice(self):
    board = utils_test.load_board('''
      XX.......
      X........
    ''' + EMPTY_ROW * 7)

    lib_tracker = LibertyTracker.from_board(utils_test.BOARD_SIZE, board)
    captured = lib_tracker.add_stone(WHITE, coords.from_kgs(
        utils_test.BOARD_SIZE, 'B8'))
    self.assertEqual(len(lib_tracker.groups), 2)
    black_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9')]]
    self.assertEqual(black_group.stones, coords_from_kgs_set('A9 B9 A8'))
    self.assertEqual(black_group.liberties, coords_from_kgs_set('C9 A7'))

    white_group = lib_tracker.groups[lib_tracker.group_index[coords.from_kgs(
        utils_test.BOARD_SIZE, 'B8')]]
    self.assertEqual(white_group.stones, coords_from_kgs_set('B8'))
    self.assertEqual(white_group.liberties, coords_from_kgs_set('C8 B7'))

    self.assertEqual(captured, set())


class TestPosition(utils_test.MiniGoUnitTest):

  def test_passing(self):
    start_position = Position(
        utils_test.BOARD_SIZE,
        board=TEST_BOARD,
        n=0,
        komi=6.5,
        caps=(1, 2),
        ko=coords.from_kgs(utils_test.BOARD_SIZE, 'A1'),
        recent=tuple(),
        to_play=BLACK,
    )
    expected_position = Position(
        utils_test.BOARD_SIZE,
        board=TEST_BOARD,
        n=1,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=(PlayerMove(BLACK, None),),
        to_play=WHITE,
    )
    pass_position = start_position.pass_move()
    self.assertEqualPositions(pass_position, expected_position)

  def test_flipturn(self):
    start_position = Position(
        utils_test.BOARD_SIZE,
        board=TEST_BOARD,
        n=0,
        komi=6.5,
        caps=(1, 2),
        ko=coords.from_kgs(utils_test.BOARD_SIZE, 'A1'),
        recent=tuple(),
        to_play=BLACK,
    )
    expected_position = Position(
        utils_test.BOARD_SIZE,
        board=TEST_BOARD,
        n=0,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=tuple(),
        to_play=WHITE,
    )
    flip_position = start_position.flip_playerturn()
    self.assertEqualPositions(flip_position, expected_position)

  def test_is_move_suicidal(self):
    board = utils_test.load_board('''
      ...O.O...
      ....O....
      XO.....O.
      OXO...OXO
      O.XO.OX.O
      OXO...OOX
      XO.......
      ......XXO
      .....XOO.
    ''')
    position = Position(
        utils_test.BOARD_SIZE,
        board=board,
        to_play=BLACK,
    )
    suicidal_moves = coords_from_kgs_set('E9 H5')
    nonsuicidal_moves = coords_from_kgs_set('B5 J1 A9')
    for move in suicidal_moves:
      # sanity check my coordinate input
      assert position.board[move] == go.EMPTY
      self.assertTrue(position.is_move_suicidal(move), str(move))
    for move in nonsuicidal_moves:
      # sanity check my coordinate input
      assert position.board[move] == go.EMPTY
      self.assertFalse(position.is_move_suicidal(move), str(move))

  def test_legal_moves(self):
    board = utils_test.load_board('''
      .O.O.XOX.
      O..OOOOOX
      ......O.O
      OO.....OX
      XO.....X.
      .O.......
      OX.....OO
      XX...OOOX
      .....O.X.
    ''')
    position = Position(utils_test.BOARD_SIZE, board=board, to_play=BLACK)
    illegal_moves = coords_from_kgs_set('A9 E9 J9')
    legal_moves = coords_from_kgs_set('A4 G1 J1 H7') | {None}
    for move in illegal_moves:
      with self.subTest(type='illegal', move=move):
        self.assertFalse(position.is_move_legal(move))
    for move in legal_moves:
      with self.subTest(type='legal', move=move):
        self.assertTrue(position.is_move_legal(move))
    # check that the bulk legal test agrees with move-by-move illegal test.
    bulk_legality = position.all_legal_moves()
    for i, bulk_legal in enumerate(bulk_legality):
      with self.subTest(type='bulk', move=coords.from_flat(
          utils_test.BOARD_SIZE, i)):
        self.assertEqual(
            bulk_legal, position.is_move_legal(
                coords.from_flat(utils_test.BOARD_SIZE, i)))

    # flip the colors and check that everything is still (il)legal
    position = Position(utils_test.BOARD_SIZE, board=-board, to_play=WHITE)
    for move in illegal_moves:
      with self.subTest(type='illegal', move=move):
        self.assertFalse(position.is_move_legal(move))
    for move in legal_moves:
      with self.subTest(type='legal', move=move):
        self.assertTrue(position.is_move_legal(move))
    bulk_legality = position.all_legal_moves()
    for i, bulk_legal in enumerate(bulk_legality):
      with self.subTest(type='bulk', move=coords.from_flat(
          utils_test.BOARD_SIZE, i)):
        self.assertEqual(
            bulk_legal, position.is_move_legal(coords.from_flat(
                utils_test.BOARD_SIZE, i)))

  def test_move(self):
    start_position = Position(
        utils_test.BOARD_SIZE,
        board=TEST_BOARD,
        n=0,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=tuple(),
        to_play=BLACK,
    )
    expected_board = utils_test.load_board('''
      .XX....OO
      X........
    ''' + EMPTY_ROW * 7)
    expected_position = Position(
        utils_test.BOARD_SIZE,
        board=expected_board,
        n=1,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=(PlayerMove(BLACK, coords.from_kgs(
            utils_test.BOARD_SIZE, 'C9')),),
        to_play=WHITE,
    )
    actual_position = start_position.play_move(coords.from_kgs(
        utils_test.BOARD_SIZE, 'C9'))
    self.assertEqualPositions(actual_position, expected_position)

    expected_board2 = utils_test.load_board('''
      .XX....OO
      X.......O
    ''' + EMPTY_ROW * 7)
    expected_position2 = Position(
        utils_test.BOARD_SIZE,
        board=expected_board2,
        n=2,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=(
            PlayerMove(BLACK, coords.from_kgs(utils_test.BOARD_SIZE, 'C9')),
            PlayerMove(WHITE, coords.from_kgs(utils_test.BOARD_SIZE, 'J8')),),
        to_play=BLACK,
    )
    actual_position2 = actual_position.play_move(coords.from_kgs(
        utils_test.BOARD_SIZE, 'J8'))
    self.assertEqualPositions(actual_position2, expected_position2)

  def test_move_with_capture(self):
    start_board = utils_test.load_board(
        EMPTY_ROW * 5 + '''
        XXXX.....
        XOOX.....
        O.OX.....
        OOXX.....
        ''')
    start_position = Position(
        utils_test.BOARD_SIZE,
        board=start_board,
        n=0,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=tuple(),
        to_play=BLACK,
        )
    expected_board = utils_test.load_board(
        EMPTY_ROW * 5 + '''
        XXXX.....
        X..X.....
        .X.X.....
        ..XX.....
        ''')
    expected_position = Position(
        utils_test.BOARD_SIZE,
        board=expected_board,
        n=1,
        komi=6.5,
        caps=(7, 2),
        ko=None,
        recent=(PlayerMove(BLACK, coords.from_kgs(
            utils_test.BOARD_SIZE, 'B2')),),
        to_play=WHITE,)
    actual_position = start_position.play_move(coords.from_kgs(
        utils_test.BOARD_SIZE, 'B2'))
    self.assertEqualPositions(actual_position, expected_position)

  def test_ko_move(self):
    start_board = utils_test.load_board('''
      .OX......
      OX.......
    ''' + EMPTY_ROW * 7)
    start_position = Position(
        utils_test.BOARD_SIZE,
        board=start_board,
        n=0,
        komi=6.5,
        caps=(1, 2),
        ko=None,
        recent=tuple(),
        to_play=BLACK,
        )
    expected_board = utils_test.load_board('''
      X.X......
      OX.......
    ''' + EMPTY_ROW * 7)
    expected_position = Position(
        utils_test.BOARD_SIZE,
        board=expected_board,
        n=1,
        komi=6.5,
        caps=(2, 2),
        ko=coords.from_kgs(utils_test.BOARD_SIZE, 'B9'),
        recent=(PlayerMove(BLACK, coords.from_kgs(
            utils_test.BOARD_SIZE, 'A9')),),
        to_play=WHITE,
        )
    actual_position = start_position.play_move(coords.from_kgs(
        utils_test.BOARD_SIZE, 'A9'))

    self.assertEqualPositions(actual_position, expected_position)

    # Check that retaking ko is illegal until two intervening moves
    with self.assertRaises(go.IllegalMove):
      actual_position.play_move(coords.from_kgs(utils_test.BOARD_SIZE, 'B9'))
    pass_twice = actual_position.pass_move().pass_move()
    ko_delayed_retake = pass_twice.play_move(coords.from_kgs(
        utils_test.BOARD_SIZE, 'B9'))
    expected_position = Position(
        utils_test.BOARD_SIZE,
        board=start_board,
        n=4,
        komi=6.5,
        caps=(2, 3),
        ko=coords.from_kgs(utils_test.BOARD_SIZE, 'A9'),
        recent=(
            PlayerMove(BLACK, coords.from_kgs(utils_test.BOARD_SIZE, 'A9')),
            PlayerMove(WHITE, None),
            PlayerMove(BLACK, None),
            PlayerMove(WHITE, coords.from_kgs(utils_test.BOARD_SIZE, 'B9')),),
        to_play=BLACK)
    self.assertEqualPositions(ko_delayed_retake, expected_position)

  def test_is_game_over(self):
    root = go.Position(utils_test.BOARD_SIZE)
    self.assertFalse(root.is_game_over())
    first_pass = root.play_move(None)
    self.assertFalse(first_pass.is_game_over())
    second_pass = first_pass.play_move(None)
    self.assertTrue(second_pass.is_game_over())

  def test_scoring(self):
    board = utils_test.load_board('''
        .XX......
        OOXX.....
        OOOX...X.
        OXX......
        OOXXXXXX.
        OOOXOXOXX
        .O.OOXOOX
        .O.O.OOXX
        ......OOO
      ''')
    position = Position(
        utils_test.BOARD_SIZE,
        board=board,
        n=54,
        komi=6.5,
        caps=(2, 5),
        ko=None,
        recent=tuple(),
        to_play=BLACK,
    )
    expected_score = 1.5
    self.assertEqual(position.score(), expected_score)

    board = utils_test.load_board('''
        XXX......
        OOXX.....
        OOOX...X.
        OXX......
        OOXXXXXX.
        OOOXOXOXX
        .O.OOXOOX
        .O.O.OOXX
        ......OOO
      ''')
    position = Position(
        utils_test.BOARD_SIZE,
        board=board,
        n=55,
        komi=6.5,
        caps=(2, 5),
        ko=None,
        recent=tuple(),
        to_play=WHITE,
    )
    expected_score = 2.5
    self.assertEqual(position.score(), expected_score)

  def test_replay_position(self):
    sgf_positions = list(sgf_wrapper.replay_sgf(
        utils_test.BOARD_SIZE, NO_HANDICAP_SGF))
    initial = sgf_positions[0]
    self.assertEqual(initial.result, go.WHITE)

    final = sgf_positions[-1].position.play_move(
        sgf_positions[-1].next_move)

    # sanity check to ensure we're working with the right position
    final_board = utils_test.load_board('''
        .OXX.....
        O.OX.X...
        .OOX.....
        OOOOXXXXX
        XOXXOXOOO
        XOOXOO.O.
        XOXXXOOXO
        XXX.XOXXO
        X..XOO.O.
      ''')
    expected_final_position = go.Position(
        utils_test.BOARD_SIZE,
        final_board,
        n=62,
        komi=6.5,
        caps=(3, 2),
        ko=None,
        recent=tuple(),
        to_play=go.BLACK
    )
    self.assertEqualPositions(expected_final_position, final)
    self.assertEqual(final.n, len(final.recent))

    replayed_positions = list(go.replay_position(
        utils_test.BOARD_SIZE, final, 1))
    for sgf_pos, replay_pos in zip(sgf_positions, replayed_positions):
      self.assertEqualPositions(sgf_pos.position, replay_pos.position)


if __name__ == '__main__':
  tf.test.main()
