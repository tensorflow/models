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

import go
import time
import numpy as np
import random
import sys

import coords
import go
from gtp_wrapper import MCTSPlayer

SIMULTANEOUS_LEAVES = 8


def play(network, readouts, resign_threshold, verbosity=0):
    ''' Plays out a self-play match, returning
    - the final position
    - the n x 362 tensor of floats representing the mcts search probabilities
    - the n-ary tensor of floats representing the original value-net estimate
    where n is the number of moves in the game'''
    player = MCTSPlayer(network,
                        resign_threshold=resign_threshold,
                        verbosity=verbosity,
                        num_parallel=SIMULTANEOUS_LEAVES)
    global_n = 0

    # Disable resign in 5% of games
    if random.random() < 0.05:
        player.resign_threshold = -1.0

    player.initialize_game()

    # Must run this once at the start, so that noise injection actually
    # affects the first move of the game.
    first_node = player.root.select_leaf()
    prob, val = network.run(first_node.position)
    first_node.incorporate_results(prob, val, first_node)

    while True:
        start = time.time()
        player.root.inject_noise()
        current_readouts = player.root.N
        # we want to do "X additional readouts", rather than "up to X readouts".
        while player.root.N < current_readouts + readouts:
            player.tree_search()

        if (verbosity >= 3):
            print(player.root.position)
            print(player.root.describe())

        if player.should_resign():
            player.set_result(-1 * player.root.position.to_play,
                              was_resign=True)
            break
        move = player.pick_move()
        player.play_move(move)
        if player.root.is_done():
            player.set_result(player.root.position.result(), was_resign=False)
            break

        if (verbosity >= 2) or (verbosity >= 1 and player.root.position.n % 10 == 9):
            print("Q: {:.5f}".format(player.root.Q))
            dur = time.time() - start
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (
                player.root.position.n, readouts, dur / readouts * 100.0, dur), flush=True)
        if verbosity >= 3:
            print("Played >>",
                  coords.to_kgs(coords.from_flat(player.root.fmove)))

    if verbosity >= 2:
        print("%s: %.3f" % (player.result_string, player.root.Q), file=sys.stderr)
        print(player.root.position,
              player.root.position.score(), file=sys.stderr)

    return player
