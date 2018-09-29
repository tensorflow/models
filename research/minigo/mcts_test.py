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
"""Tests for mcts."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import go
from mcts import MCTSNode
import numpy as np
import utils_test

tf.logging.set_verbosity(tf.logging.ERROR)

ALMOST_DONE_BOARD = utils_test.load_board('''
  .XO.XO.OO
  X.XXOOOO.
  XXXXXOOOO
  XXXXXOOOO
  .XXXXOOO.
  XXXXXOOOO
  .XXXXOOO.
  XXXXXOOOO
  XXXXOOOOO
''')

TEST_POSITION = go.Position(
    utils_test.BOARD_SIZE,
    board=ALMOST_DONE_BOARD,
    n=105,
    komi=2.5,
    caps=(1, 4),
    ko=None,
    recent=(go.PlayerMove(go.BLACK, (0, 1)),
            go.PlayerMove(go.WHITE, (0, 8))),
    to_play=go.BLACK
)

SEND_TWO_RETURN_ONE = go.Position(
    utils_test.BOARD_SIZE,
    board=ALMOST_DONE_BOARD,
    n=75,
    komi=0.5,
    caps=(0, 0),
    ko=None,
    recent=(
        go.PlayerMove(go.BLACK, (0, 1)),
        go.PlayerMove(go.WHITE, (0, 8)),
        go.PlayerMove(go.BLACK, (1, 0))),
    to_play=go.WHITE
)

MAX_DEPTH = (utils_test.BOARD_SIZE ** 2) * 1.4


class TestMctsNodes(utils_test.MiniGoUnitTest):

  def test_action_flipping(self):
    np.random.seed(1)
    probs = np.array([.02] * (
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1))
    probs += np.random.random(
        [utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1]) * 0.001
    black_root = MCTSNode(
        utils_test.BOARD_SIZE, go.Position(utils_test.BOARD_SIZE))
    white_root = MCTSNode(utils_test.BOARD_SIZE, go.Position(
        utils_test.BOARD_SIZE, to_play=go.WHITE))
    black_root.select_leaf().incorporate_results(probs, 0, black_root)
    white_root.select_leaf().incorporate_results(probs, 0, white_root)
    # No matter who is to play, when we know nothing else, the priors
    # should be respected, and the same move should be picked
    black_leaf = black_root.select_leaf()
    white_leaf = white_root.select_leaf()
    self.assertEqual(black_leaf.fmove, white_leaf.fmove)
    self.assertEqualNPArray(
        black_root.child_action_score, white_root.child_action_score)

  def test_select_leaf(self):
    flattened = coords.to_flat(utils_test.BOARD_SIZE, coords.from_kgs(
        utils_test.BOARD_SIZE, 'D9'))
    probs = np.array([.02] * (
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1))
    probs[flattened] = 0.4
    root = MCTSNode(utils_test.BOARD_SIZE, SEND_TWO_RETURN_ONE)
    root.select_leaf().incorporate_results(probs, 0, root)

    self.assertEqual(root.position.to_play, go.WHITE)
    self.assertEqual(root.select_leaf(), root.children[flattened])

  def test_backup_incorporate_results(self):
    probs = np.array([.02] * (
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1))
    root = MCTSNode(utils_test.BOARD_SIZE, SEND_TWO_RETURN_ONE)
    root.select_leaf().incorporate_results(probs, 0, root)

    leaf = root.select_leaf()
    leaf.incorporate_results(probs, -1, root)  # white wins!

    # Root was visited twice: first at the root, then at this child.
    self.assertEqual(root.N, 2)
    # Root has 0 as a prior and two visits with value 0, -1
    self.assertAlmostEqual(root.Q, -1/3)  # average of 0, 0, -1
    # Leaf should have one visit
    self.assertEqual(root.child_N[leaf.fmove], 1)
    self.assertEqual(leaf.N, 1)
    # And that leaf's value had its parent's Q (0) as a prior, so the Q
    # should now be the average of 0, -1
    self.assertAlmostEqual(root.child_Q[leaf.fmove], -0.5)
    self.assertAlmostEqual(leaf.Q, -0.5)

    # We're assuming that select_leaf() returns a leaf like:
    #   root
    #     \
    #     leaf
    #       \
    #       leaf2
    # which happens in this test because root is W to play and leaf was a W win.
    self.assertEqual(root.position.to_play, go.WHITE)
    leaf2 = root.select_leaf()
    leaf2.incorporate_results(probs, -0.2, root)  # another white semi-win
    self.assertEqual(root.N, 3)
    # average of 0, 0, -1, -0.2
    self.assertAlmostEqual(root.Q, -0.3)

    self.assertEqual(leaf.N, 2)
    self.assertEqual(leaf2.N, 1)
    # average of 0, -1, -0.2
    self.assertAlmostEqual(leaf.Q, root.child_Q[leaf.fmove])
    self.assertAlmostEqual(leaf.Q, -0.4)
    # average of -1, -0.2
    self.assertAlmostEqual(leaf.child_Q[leaf2.fmove], -0.6)
    self.assertAlmostEqual(leaf2.Q, -0.6)

  def test_do_not_explore_past_finish(self):
    probs = np.array([0.02] * (
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1), dtype=np.float32)
    root = MCTSNode(utils_test.BOARD_SIZE, go.Position(utils_test.BOARD_SIZE))
    root.select_leaf().incorporate_results(probs, 0, root)
    first_pass = root.maybe_add_child(
        coords.to_flat(utils_test.BOARD_SIZE, None))
    first_pass.incorporate_results(probs, 0, root)
    second_pass = first_pass.maybe_add_child(
        coords.to_flat(utils_test.BOARD_SIZE, None))
    with self.assertRaises(AssertionError):
      second_pass.incorporate_results(probs, 0, root)
    node_to_explore = second_pass.select_leaf()
    # should just stop exploring at the end position.
    self.assertEqual(node_to_explore, second_pass)

  def test_add_child(self):
    root = MCTSNode(utils_test.BOARD_SIZE, go.Position(utils_test.BOARD_SIZE))
    child = root.maybe_add_child(17)
    self.assertIn(17, root.children)
    self.assertEqual(child.parent, root)
    self.assertEqual(child.fmove, 17)

  def test_add_child_idempotency(self):
    root = MCTSNode(utils_test.BOARD_SIZE, go.Position(utils_test.BOARD_SIZE))
    child = root.maybe_add_child(17)
    current_children = copy.copy(root.children)
    child2 = root.maybe_add_child(17)
    self.assertEqual(child, child2)
    self.assertEqual(current_children, root.children)

  def test_never_select_illegal_moves(self):
    probs = np.array([0.02] * (
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1))
    # let's say the NN were to accidentally put a high weight on an illegal move
    probs[1] = 0.99
    root = MCTSNode(utils_test.BOARD_SIZE, SEND_TWO_RETURN_ONE)
    root.incorporate_results(probs, 0, root)
    # and let's say the root were visited a lot of times, which pumps up the
    # action score for unvisited moves...
    root.N = 100000
    root.child_N[root.position.all_legal_moves()] = 10000
    # this should not throw an error...
    leaf = root.select_leaf()
    # the returned leaf should not be the illegal move
    self.assertNotEqual(leaf.fmove, 1)

    # and even after injecting noise, we should still not select an illegal move
    for _ in range(10):
      root.inject_noise()
      leaf = root.select_leaf()
      self.assertNotEqual(leaf.fmove, 1)

  def test_dont_pick_unexpanded_child(self):
    probs = np.array([0.001] * (
        utils_test.BOARD_SIZE * utils_test.BOARD_SIZE + 1))
    # make one move really likely so that tree search goes down that path twice
    # even with a virtual loss
    probs[17] = 0.999
    root = MCTSNode(utils_test.BOARD_SIZE, go.Position(utils_test.BOARD_SIZE))
    root.incorporate_results(probs, 0, root)
    leaf1 = root.select_leaf()
    self.assertEqual(leaf1.fmove, 17)
    leaf1.add_virtual_loss(up_to=root)
    # the second select_leaf pick should return the same thing, since the child
    # hasn't yet been sent to neural net for eval + result incorporation
    leaf2 = root.select_leaf()
    self.assertIs(leaf1, leaf2)


if __name__ == '__main__':
  tf.test.main()
