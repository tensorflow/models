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
"""Monte Carlo Tree Search implementation.

All terminology here (Q, U, N, p_UCT) uses the same notation as in the
AlphaGo (AG) paper, and more details can be found in the paper. Here is a brief
description:
  Q: the action value of a position
  U: the search control strategy
  N: the visit counts of a state
  p_UCT: the PUCT algorithm for action selection
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import coords
import numpy as np

# Exploration constant
c_PUCT = 1.38  # pylint: disable=invalid-name


# Dirichlet noise, as a function of board_size
def D_NOISE_ALPHA(board_size):  # pylint: disable=invalid-name
  return 0.03 * 361 / (board_size ** 2)


class DummyNode(object):
  """A fake node of a MCTS search tree.

  This node is intended to be a placeholder for the root node, which would
  otherwise have no parent node. If all nodes have parents, code becomes
  simpler.
  """

  # pylint: disable=invalid-name
  def __init__(self, board_size):
    self.board_size = board_size
    self.parent = None
    self.child_N = collections.defaultdict(float)
    self.child_W = collections.defaultdict(float)


class MCTSNode(object):
  """A node of a MCTS search tree.

  A node knows how to compute the action scores of all of its children,
  so that a decision can be made about which move to explore next. Upon
  selecting a move, the children dictionary is updated with a new node.

  position: A go.Position instance
  fmove: A move (coordinate) that led to this position, a a flattened coord
    (raw number between 0-N^2, with None a pass)
  parent: A parent MCTSNode.
  """
  # pylint: disable=invalid-name

  def __init__(self, board_size, position, fmove=None, parent=None):
    if parent is None:
      parent = DummyNode(board_size)
    self.board_size = board_size
    self.parent = parent
    self.fmove = fmove  # move that led to this position, as flattened coords
    self.position = position
    self.is_expanded = False
    self.losses_applied = 0  # number of virtual losses on this node
    # using child_() allows vectorized computation of action score.
    self.illegal_moves = 1000 * (1 - self.position.all_legal_moves())
    self.child_N = np.zeros([board_size * board_size + 1], dtype=np.float32)
    self.child_W = np.zeros([board_size * board_size + 1], dtype=np.float32)
    # save a copy of the original prior before it gets mutated by d-noise.
    self.original_prior = np.zeros([board_size * board_size + 1],
                                   dtype=np.float32)
    self.child_prior = np.zeros([board_size * board_size + 1], dtype=np.float32)
    self.children = {}  # map of flattened moves to resulting MCTSNode

  def __repr__(self):
    return '<MCTSNode move={}, N={}, to_play={}>'.format(
        self.position.recent[-1:], self.N, self.position.to_play)

  @property
  def child_action_score(self):
    return (self.child_Q * self.position.to_play
            + self.child_U - self.illegal_moves)

  @property
  def child_Q(self):
    return self.child_W / (1 + self.child_N)

  @property
  def child_U(self):
    return (c_PUCT * math.sqrt(1 + self.N) *
            self.child_prior / (1 + self.child_N))

  @property
  def Q(self):
    return self.W / (1 + self.N)

  @property
  def N(self):
    return self.parent.child_N[self.fmove]

  @N.setter
  def N(self, value):
    self.parent.child_N[self.fmove] = value

  @property
  def W(self):
    return self.parent.child_W[self.fmove]

  @W.setter
  def W(self, value):
    self.parent.child_W[self.fmove] = value

  @property
  def Q_perspective(self):
    """Return value of position, from perspective of player to play."""
    return self.Q * self.position.to_play

  def select_leaf(self):
    current = self
    pass_move = self.board_size * self.board_size
    while True:
      current.N += 1
      # if a node has never been evaluated, we have no basis to select a child.
      if not current.is_expanded:
        break
      # HACK: if last move was a pass, always investigate double-pass first
      # to avoid situations where we auto-lose by passing too early.
      if (current.position.recent
          and current.position.recent[-1].move is None
          and current.child_N[pass_move] == 0):
        current = current.maybe_add_child(pass_move)
        continue

      best_move = np.argmax(current.child_action_score)
      current = current.maybe_add_child(best_move)
    return current

  def maybe_add_child(self, fcoord):
    """Add child node for fcoord if it doesn't already exist, and returns it."""
    if fcoord not in self.children:
      new_position = self.position.play_move(
          coords.from_flat(self.board_size, fcoord))
      self.children[fcoord] = MCTSNode(
          self.board_size, new_position, fmove=fcoord, parent=self)
    return self.children[fcoord]

  def add_virtual_loss(self, up_to):
    """Propagate a virtual loss up to the root node.

    Args:
      up_to: The node to propagate until. (Keep track of this! You'll
        need it to reverse the virtual loss later.)
    """
    self.losses_applied += 1
    # This is a "win" for the current node; hence a loss for its parent node
    # who will be deciding whether to investigate this node again.
    loss = self.position.to_play
    self.W += loss
    if self.parent is None or self is up_to:
      return
    self.parent.add_virtual_loss(up_to)

  def revert_virtual_loss(self, up_to):
    self.losses_applied -= 1
    revert = -self.position.to_play
    self.W += revert
    if self.parent is None or self is up_to:
      return
    self.parent.revert_virtual_loss(up_to)

  def revert_visits(self, up_to):
    """Revert visit increments."""
    # Sometimes, repeated calls to select_leaf return the same node.
    # This is rare and we're okay with the wasted computation to evaluate
    # the position multiple times by the dual_net. But select_leaf has the
    # side effect of incrementing visit counts. Since we want the value to
    # only count once for the repeatedly selected node, we also have to
    # revert the incremented visit counts.

    self.N -= 1
    if self.parent is None or self is up_to:
      return
    self.parent.revert_visits(up_to)

  def incorporate_results(self, move_probabilities, value, up_to):
    assert move_probabilities.shape == (self.board_size * self.board_size + 1,)
    # A finished game should not be going through this code path - should
    # directly call backup_value() on the result of the game.
    assert not self.position.is_game_over()
    if self.is_expanded:
      self.revert_visits(up_to=up_to)
      return
    self.is_expanded = True
    self.original_prior = self.child_prior = move_probabilities
    # initialize child Q as current node's value, to prevent dynamics where
    # if B is winning, then B will only ever explore 1 move, because the Q
    # estimation will be so much larger than the 0 of the other moves.
    #
    # Conversely, if W is winning, then B will explore all 362 moves before
    # continuing to explore the most favorable move. This is a waste of search.
    #
    # The value seeded here acts as a prior, and gets averaged into
    # Q calculations.
    self.child_W = np.ones([self.board_size * self.board_size + 1],
                           dtype=np.float32) * value
    self.backup_value(value, up_to=up_to)

  def backup_value(self, value, up_to):
    """Propagates a value estimation up to the root node.

    Args:
      value: the value to be propagated (1 = black wins, -1 = white wins)
      up_to: the node to propagate until.
    """
    self.W += value
    if self.parent is None or self is up_to:
      return
    self.parent.backup_value(value, up_to)

  def is_done(self):
    # True if the last two moves were Pass or if the position is at a move
    # greater than the max depth.

    max_depth = (self.board_size ** 2) * 1.4  # 505 moves for 19x19, 113 for 9x9
    return self.position.is_game_over() or self.position.n >= max_depth

  def inject_noise(self):
    dirch = np.random.dirichlet([D_NOISE_ALPHA(self.board_size)] * (
        (self.board_size * self.board_size) + 1))
    self.child_prior = self.child_prior * 0.75 + dirch * 0.25

  def children_as_pi(self, squash=False):
    """Returns the child visit counts as a probability distribution, pi."""
    # If squash is true, exponentiate the probabilities by a temperature
    # slightly larger than unity to encourage diversity in early play and
    # hopefully to move away from 3-3s

    probs = self.child_N
    if squash:
      probs **= .95
    return probs / np.sum(probs)

  def most_visited_path(self):
    node = self
    output = []
    while node.children:
      next_kid = np.argmax(node.child_N)
      node = node.children.get(next_kid)
      if node is None:
        output.append('GAME END')
        break
      output.append('{} ({}) ==> '.format(
          coords.to_kgs(
              self.board_size,
              coords.from_flat(self.board_size, node.fmove)), node.N))
    output.append('Q: {:.5f}\n'.format(node.Q))
    return ''.join(output)

  def mvp_gg(self):
    """ Returns most visited path in go-gui VAR format e.g. 'b r3 w c17..."""
    node = self
    output = []
    while node.children and max(node.child_N) > 1:
      next_kid = np.argmax(node.child_N)
      node = node.children[next_kid]
      output.append('{}'.format(coords.to_kgs(
          self.board_size, coords.from_flat(self.board_size, node.fmove))))
    return ' '.join(output)

  def describe(self):
    sort_order = list(range(self.board_size * self.board_size + 1))
    sort_order.sort(key=lambda i: (
        self.child_N[i], self.child_action_score[i]), reverse=True)
    soft_n = self.child_N / sum(self.child_N)
    p_delta = soft_n - self.child_prior
    p_rel = p_delta / self.child_prior
    # Dump out some statistics
    output = []
    output.append('{q:.4f}\n'.format(q=self.Q))
    output.append(self.most_visited_path())
    output.append(
        '''move:  action      Q      U      P    P-Dir    N  soft-N
        p-delta  p-rel\n''')
    output.append(
        '\n'.join([
            '''{!s:6}: {: .3f}, {: .3f}, {:.3f}, {:.3f}, {:.3f}, {:4d} {:.4f}
            {: .5f} {: .2f}'''.format(
                coords.to_kgs(self.board_size, coords.from_flat(
                    self.board_size, key)),
                self.child_action_score[key],
                self.child_Q[key],
                self.child_U[key],
                self.child_prior[key],
                self.original_prior[key],
                int(self.child_N[key]),
                soft_n[key],
                p_delta[key],
                p_rel[key])
            for key in sort_order][:15]))
    return ''.join(output)
