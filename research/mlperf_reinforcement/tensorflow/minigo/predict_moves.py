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

import os
import glob

def main():
  # model_path = '/usr/local/google/home/vbittorf/Documents/minigo/20hour1000game/models/000002-nassau'
  #model_path = '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000063-set-maggot'
  #model_path = '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000078-modern-cod'
  # model_path = '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000000-bootstrap'

  models = [
    # '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000000-bootstrap',
    # '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000114-star-bird',
    # '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000217-first-spider',
    # '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000325-secure-racer',
    # '/usr/local/google/home/vbittorf/Documents/minigo/gcp0/000091-proven-deer',
    # '/usr/local/google/home/vbittorf/Documents/minigo/gcp0/000050-deep-dingo',
    # '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000063-set-maggot',
    # '/usr/local/google/home/vbittorf/Documents/minigo/pro9x9/000496-polite-ray',


    #'/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000005-above-sole',
    #'/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000010-wanted-eel',
    #'/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000015-glad-magpie',
    #'/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000020-alive-cat',
    #'/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000025-robust-moray',


    #'/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/0*',
    #'/usr/local/google/home/vbittorf/results/minigo/9x9_normal_500sp_sandbox3/models/0*',
    '/usr/local/google/home/vbittorf/results/minigo/9x9_normal_500sp_sandbox1/models/0*',
  ]

  '''
  models = [
    '/usr/local/google/home/vbittorf/Documents/minigo/gcp3/000001-centurion',
    '/usr/local/google/home/vbittorf/Documents/minigo/gcp1/000009-lynx',
  ]
  '''


  '''
  sgf_files = [
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0001.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0002.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0003.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0004.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0005.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0008.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0009.sgf',
  '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0010.sgf',
  ]
  '''

  unglob_models = models

  models = []
  for model in unglob_models:
    globbed = glob.glob(model)
    tmp = []
    for g in globbed:
      tmp.append(g.split('.')[0])
    tmp = list(sorted(list(set(tmp))))
    models.extend(tmp)

  sgf_files = [
    '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_prob_0002.sgf',
    '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_prob_0004.sgf',
    '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_prob_0005.sgf',
    '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_easy_capture.sgf',
    #'/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_easy_capture_rot90.sgf',
    #'/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_easy_capture_rot180.sgf',
    #'/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/9x9_easy_capture_rot270.sgf',
  ]

  model_results = {}

  for model_path in models:
    model_results[model_path], total_pct = report_for_puzzles(model_path, sgf_files, 5)
    report_model_results(model_results)


def report_model_results(model_results):
  for model in sorted(model_results):
    print(os.path.basename(model))
    res = model_results[model]
    tot = 0
    for puzzle in sorted(res):
      moves = res[puzzle]
      move_str = ' '.join(map(lambda t: str(t[0]), moves))
      pct = len(list(filter(lambda m: m[2], moves))) * 1.0 / len(moves) * 100
      pct_correct = '{:.2f}%'.format(pct)
      tot += pct
      print('\t'.join([os.path.basename(puzzle), pct_correct, move_str]))
    print('Total\t' + str(tot / len(res)) + '%')


def report_for_puzzles(model_path, sgf_files, rounds):
  results = {}
  tries = 0
  correct = 0
  network = dual_net.DualNetwork(model_path)
  for attempt in range(rounds):
    for filename in sgf_files:
      if filename not in results:
        results[filename] = []
      move, target, is_right = predict_move(filename, network)
      tries += 1
      if is_right:
        correct += 1
      results[filename].append((move, target, is_right))
      report_model_results({model_path: results})
  return results, correct * 1.0 / tries


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


def predict_move(filename, network):
  # Strategies: def initialize_game(self, position=None):

  #filename = '/usr/local/google/home/vbittorf/projects/minigo/benchmark_sgf/prob_0001.sgf'
  replay = []

  with open(filename) as f:
      text = f.read()
      print(text)
      for position_w_context in sgf_wrapper.replay_sgf(text):
        replay.append(position_w_context)

  print(replay)

  # model_path = '/usr/local/google/home/vbittorf/Documents/minigo/rl_pipeline/models/000003-leopard'
  #model_path = '/usr/local/google/home/vbittorf/Documents/minigo/20hour1000game/models/000002-nassau'
  #white_net = dual_net.DualNetwork(model_path)
  #black_net = dual_net.DualNetwork(model_path)

  #print(evaluation.play_match(white_net, black_net, 1, 50, "/tmp/sgf", 0))


  black_net = network

  player = MCTSPlayer(
        black_net, verbosity=0, two_player_mode=True, num_parallel=4)

  readouts = 361 * 10
  tried = 0
  correct = 0
  for position_w_context in replay:
      if position_w_context.next_move is None:
          continue
      player.initialize_game(position_w_context.position)

      current_readouts = player.root.N
      while player.root.N < current_readouts + readouts:
         player.tree_search()


      move = player.pick_move()
      #if player.should_resign():  # Force resign
      #  move = 'R'
      #else:
      #  move = player.suggest_move(position_w_context.position)
      tried += 1
      if move == position_w_context.next_move:
        correct += 1
      player.play_move(move)
      print(player.root.position)
      print(move, position_w_context.next_move)
      return move, position_w_context.next_move, move == position_w_context.next_move
  print('Correct: ', correct * 1.0 / tried)






if __name__ == '__main__':
  main()
