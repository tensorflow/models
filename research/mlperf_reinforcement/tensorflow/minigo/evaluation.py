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

"""Evalation plays games between two neural nets."""

import os
import sys
import time
import sgf_wrapper
import subprocess
import multiprocessing
import random
import numpy
import tensorflow as tf

from gtp_wrapper import MCTSPlayer
import goparams

from utils import logged_timer as timer
import dual_net

SIMULTANEOUS_LEAVES = 8

def play_match_many_instance_both(prev_model, cur_model, games, readouts, sgf_dir, verbosity):
    # pfProcs: games prev_model go first
    # cfProcs: games cur_model go first
    pfProcs = []
    cfProcs = []
    winners = []
    # passing 'randomness' to spawned instances
    seed = int(sys.argv[1])
    iteration = int(sys.argv[2])
    seed = hash(hash(seed)+iteration)
    for i in range(games):
        cmd = 'GOPARAMS={} OMP_NUM_THREADS=1 KMP_HW_SUBSET={} KMP_AFFINITY=granularity=fine,proclist=[{}],explicit python3 evaluation.py {} {} {} {} {} {} {}'.format(os.environ['GOPARAMS'], os.environ['KMP_HW_SUBSET'], i%multiprocessing.cpu_count(), seed+i, prev_model, cur_model, games, readouts, sgf_dir, verbosity)
        pfProcs.append(subprocess.Popen(cmd, shell=True))
        cmd = 'GOPARAMS={} OMP_NUM_THREADS=1 KMP_HW_SUBSET={} KMP_AFFINITY=granularity=fine,proclist=[{}],explicit python3 evaluation.py {} {} {} {} {} {} {}'.format(os.environ['GOPARAMS'], os.environ['KMP_HW_SUBSET'], (i+games)%multiprocessing.cpu_count(), seed+i, cur_model, prev_model, games, readouts, sgf_dir, verbosity)
        cfProcs.append(subprocess.Popen(cmd, shell=True))
    # winPF: number of games cur_model win in games prev_model go first
    # winCF: number of games cur_model win in games cur_model go first
    winPF = 0
    winCF = 0
    for proc in pfProcs:
        retVal = proc.wait()
        if retVal == 10:
            winners.append('W')
            winPF += 1
        elif retVal == 20:
            winners.append('B')
        else:
            print("unexpected exit value", retVal)
    for proc in cfProcs:
        retVal = proc.wait()
        if retVal == 10:
            winners.append('B')
        elif retVal == 20:
            winners.append('W')
            winCF += 1
        else:
            print("unexpected exit value", retVal)
    print ("1v1 result: {}/{}, {}/{}".format(winPF, games, winCF, games))
    return winners

def play_match_one(black_model, white_model, i, readouts, sgf_dir, verbosity):
    """Plays matches between two neural nets.

    black_net: Instance of minigo.DualNetwork, a wrapper around a tensorflow
        convolutional network.
    white_net: Instance of the minigo.DualNetwork.
    games: number of games to play. We play all the games at the same time.
    sgf_dir: directory to write the sgf results.
    readouts: number of readouts to perform for each step in each game.
    return true if black wins
    """

    with timer("Loading weights"):
        black_net = dual_net.DualNetwork(black_model, selfplay=True, inference=True)
        white_net = dual_net.DualNetwork(white_model, selfplay=True, inference=True)

    # For n games, we create lists of n black and n white players
    black = MCTSPlayer(
        black_net, verbosity=verbosity, two_player_mode=True, num_parallel=SIMULTANEOUS_LEAVES)
    white = MCTSPlayer(
        white_net, verbosity=verbosity, two_player_mode=True, num_parallel=SIMULTANEOUS_LEAVES)

    black_name = os.path.basename(black_net.save_file)
    white_name = os.path.basename(white_net.save_file)

    black_win = False

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
            active.set_result(-1 *
                              active.root.position.to_play, was_resign=True)
            inactive.set_result(
                active.root.position.to_play, was_resign=True)

        if active.is_done():
            fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()),
                                                        white_name, black_name, i)
            if active.result_string is None:
              # This is an exceptionally  rare corner case where we don't get a winner.
              # Our temporary solution is to just drop this game.
              break
            if (active.result_string[0] == 'W'):
                black_win = False
            else:
                black_win = True
            if num_move > 0:
                with open(os.path.join(sgf_dir, fname), 'w') as _file:
                    sgfstr = sgf_wrapper.make_sgf(active.position.recent,
                                                  active.result_string, black_name=black_name,
                                                  white_name=white_name)
                    _file.write(sgfstr)
            print("Finished game", i, active.result_string)
            break

        move = active.pick_move()
        # print('DBUG Picked move: ', move, active, num_move)
        active.play_move(move)
        inactive.play_move(move)

        dur = time.time() - start
        num_move += 1

        if (verbosity > 1) or (verbosity == 1 and num_move % 10 == 9):
            timeper = (dur / readouts) * 100.0
            print(active.root.position)
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (num_move,
                                                               readouts,
                                                               timeper,
                                                               dur))
    return black_win

def play_match(black_net, white_net, games, readouts, sgf_dir, verbosity):
    """Plays matches between two neural nets.

    black_net: Instance of minigo.DualNetwork, a wrapper around a tensorflow
        convolutional network.
    white_net: Instance of the minigo.DualNetwork.
    games: number of games to play. We play all the games at the same time.
    sgf_dir: directory to write the sgf results.
    readouts: number of readouts to perform for each step in each game.
    """

    # For n games, we create lists of n black and n white players
    black = MCTSPlayer(
        black_net, verbosity=verbosity, two_player_mode=True, num_parallel=SIMULTANEOUS_LEAVES)
    white = MCTSPlayer(
        white_net, verbosity=verbosity, two_player_mode=True, num_parallel=SIMULTANEOUS_LEAVES)

    black_name = os.path.basename(black_net.save_file)
    white_name = os.path.basename(white_net.save_file)

    winners = []
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
                active.set_result(-1 *
                                  active.root.position.to_play, was_resign=True)
                inactive.set_result(
                    active.root.position.to_play, was_resign=True)

            if active.is_done():
                fname = "{:d}-{:s}-vs-{:s}-{:d}.sgf".format(int(time.time()),
                                                            white_name, black_name, i)
                if active.result_string is None:
                  # This is an exceptionally  rare corner case where we don't get a winner.
                  # Our temporary solution is to just drop this game.
                  break
                winners.append(active.result_string[0])
                if num_move > 0:
                    with open(os.path.join(sgf_dir, fname), 'w') as _file:
                        sgfstr = sgf_wrapper.make_sgf(active.position.recent,
                                                      active.result_string, black_name=black_name,
                                                      white_name=white_name)
                        _file.write(sgfstr)
                print("Finished game", i, active.result_string)
                break

            move = active.pick_move()
            # print('DBUG Picked move: ', move, active, num_move)
            active.play_move(move)
            inactive.play_move(move)

            dur = time.time() - start
            num_move += 1

            if (verbosity > 1) or (verbosity == 1 and num_move % 10 == 9):
                timeper = (dur / readouts) * 100.0
                print(active.root.position)
                print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (num_move,
                                                                   readouts,
                                                                   timeper,
                                                                   dur))
    return winners

if __name__ == '__main__':
    seed = int(sys.argv[1])
    random.seed(seed)
    tf.set_random_seed(seed)
    numpy.random.seed(seed)
    black_win = play_match_one(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]))
    if (black_win):
        exit(20)
    else:
        exit(10)
