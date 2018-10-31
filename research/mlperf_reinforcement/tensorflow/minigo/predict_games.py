"""TODO(vbittorf): DO NOT SUBMIT without one-line documentation for predict_moves.

TODO(vbittorf): DO NOT SUBMIT without a detailed description of predict_moves.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import dual_net
import strategies
import sgf_wrapper
import evaluation
from gtp_wrapper import MCTSPlayer
import sys

import os
import glob

REPLAY_CACHE = {}

def get_models_from_argv():
  unglob = sys.argv[1:]
  models = []
  for m in unglob:
    models.extend(glob.glob(m))
  models = map(lambda s: '.'.join(s.split('.')[:-1]), models)
  models = list(sorted(list(set(models))))
  print(models)
  return models

def main():
  models = get_models_from_argv()


  sgf_files = [
    './benchmark_sgf/9x9_pro_YKSH.sgf',
    './benchmark_sgf/9x9_pro_IYMD.sgf',
    './benchmark_sgf/9x9_pro_YSIY.sgf',
    './benchmark_sgf/9x9_pro_IYHN.sgf',
  ]

  model_results = {}

  for model_path in models:
    model_results[model_path], total_pct = report_for_puzzles(model_path, sgf_files, 2, tries_per_move=1)
    report_model_results(model_results)


def report_model_results(model_results):
  for model in sorted(model_results):
    print(os.path.basename(model))
    res = model_results[model]
    tot = 0
    for puzzle in sorted(res):
      ratings = res[puzzle]
      rate = sum(ratings) * 1.0 / len(ratings)
      pct_correct = '{:.2f}%'.format(rate * 100)
      tot += rate
      print('\t'.join([os.path.basename(puzzle), pct_correct]))
    print('Total\t{:.2f}%'.format(tot * 100.0 / len(res)))
    return tot * 100.0 / len(res)


def report_for_puzzles(model_path, sgf_files, rounds, tries_per_move=1):
  results = {}
  tries = 0
  sum_ratings = 0
  network = dual_net.DualNetwork(model_path)
  for attempt in range(rounds):
    for filename in sgf_files:
      if filename not in results:
        results[filename] = []
      move_ratings = predict_move(filename, network, tries_per_move=tries_per_move)
      tries += len(move_ratings)
      sum_ratings += sum(move_ratings)
      results[filename].append(sum(move_ratings) / len(move_ratings))
      report_model_results({model_path: results})
  return results, sum_ratings * 1.0 / tries


def predict_9x9_puzzles(model_path, tries_per_puzzle):
    sgf_files = [
      'benchmark_sgf/9x9_prob_0002.sgf',
      'benchmark_sgf/9x9_prob_0004.sgf',
      'benchmark_sgf/9x9_prob_0005.sgf',
      'benchmark_sgf/9x9_easy_capture.sgf',
      'benchmark_sgf/9x9_easy_capture_rot90.sgf',
      'benchmark_sgf/9x9_easy_capture_rot180.sgf',
      'benchmark_sgf/9x9_easy_capture_rot270.sgf',
    ]
    result, total_pct = report_for_puzzles(model_path, sgf_files, tries_per_puzzle)
    report_model_results({model_path: result})
    return result, total_pct


def predict_position(position_w_context, player, readouts=1000):
    player.initialize_game(position_w_context.position)

    current_readouts = player.root.N
    while player.root.N < current_readouts + readouts:
       player.tree_search()

    move = player.pick_move()
    player.play_move(move)
    print(player.root.position)
    print(move, position_w_context.next_move)
    return move, position_w_context.next_move, move == position_w_context.next_move


def predict_move(filename, network, tries_per_move=1, readouts=1000):
  replay = []

  if filename not in REPLAY_CACHE:
    with open(filename) as f:
        text = f.read()
        for position_w_context in sgf_wrapper.replay_sgf(text):
          replay.append(position_w_context)
    REPLAY_CACHE[filename] = replay
  replay = REPLAY_CACHE[filename]


  black_net = network

  player = MCTSPlayer(
        black_net, verbosity=0, two_player_mode=True, num_parallel=4)

  tried = 0
  correct = 0
  move_ratings = []
  for position_w_context in replay:
      if position_w_context.next_move is None:
          continue

      num_correct = 0
      for i in range(tries_per_move):
        move, correct_move, is_correct = predict_position(position_w_context, player, readouts=readouts)
        if is_correct:
          num_correct += 1
      move_ratings.append(num_correct * 1.0 / tries_per_move)
      print('RATING: ', sum(move_ratings) / len(move_ratings))
  return move_ratings


if __name__ == '__main__':
  main()
