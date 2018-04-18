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
"""Evaluation of playing games between two neural nets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import go
from gtp_wrapper import MCTSPlayer
import sgf_wrapper


def play_match(params, black_net, white_net, games, readouts,
               sgf_dir, verbosity):
  """Plays matches between two neural nets.

  One net that wins by a margin of 55% will be the winner.

  Args:
    params: An object of hyperparameters.
    black_net: Instance of the DualNetRunner class to play as black.
    white_net: Instance of the DualNetRunner class to play as white.
    games: Number of games to play. We play all the games at the same time.
    readouts: Number of readouts to perform for each step in each game.
    sgf_dir: Directory to write the sgf results.
    verbosity: Verbosity to show evaluation process.

  Returns:
    'B' is the winner is black_net, otherwise 'W'.
  """
  # For n games, we create lists of n black and n white players
  black = MCTSPlayer(
      params.board_size, black_net, verbosity=verbosity, two_player_mode=True,
      num_parallel=params.simultaneous_leaves)
  white = MCTSPlayer(
      params.board_size, white_net, verbosity=verbosity, two_player_mode=True,
      num_parallel=params.simultaneous_leaves)

  black_name = os.path.basename(black_net.save_file)
  white_name = os.path.basename(white_net.save_file)

  black_win_counts = 0
  white_win_counts = 0

  for i in range(games):
    num_move = 0  # The move number of the current game

    black.initialize_game()
    white.initialize_game()

    while True:
      start = time.time()
      active = white if num_move % 2 else black
      inactive = black if num_move % 2 else white

      current_readouts = active.root.N
      while active.root.N < current_readouts + readouts:
        active.tree_search()

      # print some stats on the search
      if verbosity >= 3:
        print(active.root.position)

      # First, check the roots for hopeless games.
      if active.should_resign():  # Force resign
        active.set_result(-active.root.position.to_play, was_resign=True)
        inactive.set_result(
            active.root.position.to_play, was_resign=True)

      if active.is_done():
        fname = '{:d}-{:s}-vs-{:s}-{:d}.sgf'.format(
            int(time.time()), white_name, black_name, i)
        with open(os.path.join(sgf_dir, fname), 'w') as f:
          sgfstr = sgf_wrapper.make_sgf(
              params.board_size, active.position.recent, active.result_string,
              black_name=black_name, white_name=white_name)
          f.write(sgfstr)
        print('Finished game', i, active.result_string)
        if active.result_string is not None:
          if active.result_string[0] == 'B':
            black_win_counts += 1
          elif active.result_string[0] == 'W':
            white_win_counts += 1

        break

      move = active.pick_move()
      active.play_move(move)
      inactive.play_move(move)

      dur = time.time() - start
      num_move += 1

      if (verbosity > 1) or (verbosity == 1 and num_move % 10 == 9):
        timeper = (dur / readouts) * 100.0
        print(active.root.position)
        print('{:d}: {:d} readouts, {:.3f} s/100. ({:.2f} sec)'.format(
            num_move, readouts, timeper, dur))

  if (black_win_counts - white_win_counts) > params.eval_win_rate * games:
    return go.BLACK_NAME
  else:
    return go.WHITE_NAME
