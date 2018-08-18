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
"""Tests for sgf_wrapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import go
from sgf_wrapper import replay_sgf, translate_sgf_move, make_sgf
import utils_test

JAPANESE_HANDICAP_SGF = '''(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]
SZ[9]HA[2]RE[Void]KM[5.50]PW[test_white]PB[test_black]AB[gc][cg];W[ee];B[dg])'''

CHINESE_HANDICAP_SGF = '''(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[9]
HA[2]RE[Void]KM[5.50]PW[test_white]PB[test_black]RE[B+39.50];B[gc];B[cg];W[ee];
B[gg];W[eg];B[ge];W[ce];B[ec];W[cc];B[dd];W[de];B[cd];W[bd];B[bc];W[bb];B[be];
W[ac];B[bf];W[dh];B[ch];W[ci];B[bi];W[di];B[ah];W[gh];B[hh];W[fh];B[hg];W[gi];
B[fg];W[dg];B[ei];W[cf];B[ef];W[ff];B[fe];W[bg];B[bh];W[af];B[ag];W[ae];B[ad];
W[ae];B[ed];W[db];B[df];W[eb];B[fb];W[ea];B[fa])'''

NO_HANDICAP_SGF = '''(;CA[UTF-8]SZ[9]PB[Murakawa Daisuke]PW[Iyama Yuta]KM[6.5]
HA[0]RE[W+1.5]GM[1];B[fd];W[cf];B[eg];W[dd];B[dc];W[cc];B[de];W[cd];B[ed];W[he];
B[ce];W[be];B[df];W[bf];B[hd];W[ge];B[gd];W[gg];B[db];W[cb];B[cg];W[bg];B[gh];
W[fh];B[hh];W[fg];B[eh];W[ei];B[di];W[fi];B[hg];W[dh];B[ch];W[ci];B[bh];W[ff];
B[fe];W[hf];B[id];W[bi];B[ah];W[ef];B[dg];W[ee];B[di];W[ig];B[ai];W[ih];B[fb];
W[hi];B[ag];W[ab];B[bd];W[bc];B[ae];W[ad];B[af];W[bd];B[ca];W[ba];B[da];W[ie])
'''

tf.logging.set_verbosity(tf.logging.ERROR)


class TestSgfGeneration(utils_test.MiniGoUnitTest):

  def test_translate_sgf_move(self):
    self.assertEqual(
        ';B[db]',
        translate_sgf_move(go.PlayerMove(go.BLACK, (1, 3)), None))
    self.assertEqual(
        ';W[aa]',
        translate_sgf_move(go.PlayerMove(go.WHITE, (0, 0)), None))
    self.assertEqual(
        ';W[]',
        translate_sgf_move(go.PlayerMove(go.WHITE, None), None))
    self.assertEqual(
        ';B[db]C[comment]',
        translate_sgf_move(go.PlayerMove(go.BLACK, (1, 3)), 'comment'))

  def test_make_sgf(self):
    all_pwcs = list(replay_sgf(utils_test.BOARD_SIZE, NO_HANDICAP_SGF))
    second_last_position, last_move, _ = all_pwcs[-1]
    last_position = second_last_position.play_move(last_move)

    back_to_sgf = make_sgf(
        utils_test.BOARD_SIZE,
        last_position.recent,
        last_position.score(),
        komi=last_position.komi,
    )
    reconstructed_positions = list(replay_sgf(
        utils_test.BOARD_SIZE, back_to_sgf))
    second_last_position2, last_move2, _ = reconstructed_positions[-1]
    last_position2 = second_last_position2.play_move(last_move2)

    self.assertEqualPositions(last_position, last_position2)


class TestSgfWrapper(utils_test.MiniGoUnitTest):

  def test_sgf_props(self):
    sgf_replayer = replay_sgf(utils_test.BOARD_SIZE, CHINESE_HANDICAP_SGF)
    initial = next(sgf_replayer)
    self.assertEqual(initial.result, go.BLACK)
    self.assertEqual(initial.position.komi, 5.5)

  def test_japanese_handicap_handling(self):
    intermediate_board = utils_test.load_board('''
      .........
      .........
      ......X..
      .........
      ....O....
      .........
      ..X......
      .........
      .........
    ''')
    intermediate_position = go.Position(
        utils_test.BOARD_SIZE,
        intermediate_board,
        n=1,
        komi=5.5,
        caps=(0, 0),
        recent=(go.PlayerMove(go.WHITE, coords.from_kgs(
            utils_test.BOARD_SIZE, 'E5')),),
        to_play=go.BLACK,
    )
    final_board = utils_test.load_board('''
      .........
      .........
      ......X..
      .........
      ....O....
      .........
      ..XX.....
      .........
      .........
    ''')
    final_position = go.Position(
        utils_test.BOARD_SIZE,
        final_board,
        n=2,
        komi=5.5,
        caps=(0, 0),
        recent=(
            go.PlayerMove(go.WHITE, coords.from_kgs(
                utils_test.BOARD_SIZE, 'E5')),
            go.PlayerMove(go.BLACK, coords.from_kgs(
                utils_test.BOARD_SIZE, 'D3')),),
        to_play=go.WHITE,
    )
    positions_w_context = list(replay_sgf(
        utils_test.BOARD_SIZE, JAPANESE_HANDICAP_SGF))
    self.assertEqualPositions(
        intermediate_position, positions_w_context[1].position)
    final_replayed_position = positions_w_context[-1].position.play_move(
        positions_w_context[-1].next_move)
    self.assertEqualPositions(final_position, final_replayed_position)

  def test_chinese_handicap_handling(self):
    intermediate_board = utils_test.load_board('''
      .........
      .........
      ......X..
      .........
      .........
      .........
      .........
      .........
      .........
    ''')
    intermediate_position = go.Position(
        utils_test.BOARD_SIZE,
        intermediate_board,
        n=1,
        komi=5.5,
        caps=(0, 0),
        recent=(go.PlayerMove(go.BLACK, coords.from_kgs(
            utils_test.BOARD_SIZE, 'G7')),),
        to_play=go.BLACK,
    )
    final_board = utils_test.load_board('''
      ....OX...
      .O.OOX...
      O.O.X.X..
      .OXXX....
      OX...XX..
      .X.XXO...
      X.XOOXXX.
      XXXO.OOX.
      .XOOX.O..
    ''')
    final_position = go.Position(
        utils_test.BOARD_SIZE,
        final_board,
        n=50,
        komi=5.5,
        caps=(7, 2),
        ko=None,
        recent=(
            go.PlayerMove(
                go.WHITE, coords.from_kgs(utils_test.BOARD_SIZE, 'E9')),
            go.PlayerMove(
                go.BLACK, coords.from_kgs(utils_test.BOARD_SIZE, 'F9')),),
        to_play=go.WHITE
    )
    positions_w_context = list(replay_sgf(
        utils_test.BOARD_SIZE, CHINESE_HANDICAP_SGF))
    self.assertEqualPositions(
        intermediate_position, positions_w_context[1].position)
    self.assertEqual(
        positions_w_context[1].next_move, coords.from_kgs(
            utils_test.BOARD_SIZE, 'C3'))
    final_replayed_position = positions_w_context[-1].position.play_move(
        positions_w_context[-1].next_move)
    self.assertEqualPositions(final_position, final_replayed_position)


if __name__ == '__main__':
  tf.test.main()
