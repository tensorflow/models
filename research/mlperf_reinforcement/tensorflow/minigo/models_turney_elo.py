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

"""Wrapper scripts to ensure that main.py commands are called correctly."""


import glob
import random
import sys
import os
import main


# K is how strongly a match should influence ELOs, 32 is ICC default
# K = 32.0
K = 64.0


SEEDS = '''
000002-wanted-gull      246     88
000002-pure-gnu 263     94
000006-big-dodo 337     97
000002-solid-lab        393     78
000010-clean-mutt       617     103
000013-noted-lark       681     96
000004-ideal-mako       717     88
000012-strong-koi       746     97
000007-proud-corgi      762     85
000016-driven-tarpon    780     98
000019-star-ibex        797     102
000023-main-burro       897     96
000027-first-ewe        1106    101
000496-polite-ray       1492    94
000495-prompt-marmot    1506    72
000496-polite-ray-upgrade       1523    107
'''

SEEDS = ''

SEED_ELOS = {}



def get_models_from_argv():
  unglob = sys.argv[2:]
  models = []
  for m in unglob:
    models.extend(glob.glob(m))
  models = map(lambda s: '.'.join(s.split('.')[:-1]), models)
  models = list(sorted(list(set(models))))
  print(models)
  return models


def new_elos(winner, loser):
  q_w = 10 ** (winner / 400)
  q_l = 10 ** (loser / 400)

  e_w = q_w / (q_w + q_l)
  e_l = q_l / (q_w + q_l)

  new_w = winner + K * (1 - e_w)
  new_l = loser + K * (0 - e_l)
  return new_w, new_l



def play_models_and_update(black, white, elos):
  output_dir = '/tmp/play_models2'
  os.system('mkdir ' + output_dir);
  white_win = main.evaluate(black, white, output_dir=output_dir, games=1, readouts=400)

  black_elo = elos[black]
  white_elo = elos[white]
  if white_win > 0:
    white_elo, black_elo = new_elos(white_elo, black_elo)
  else:
    black_elo, white_elo = new_elos(black_elo, white_elo)
  elos[black] = black_elo
  elos[white] = white_elo


def print_elos(elos, games, out_file):
  for model in sorted(elos, key=lambda k: elos[k]):
    print(
        '{}\t{}\t{}'.format(os.path.basename(model), elos[model], games[model]))
  print('{}\n'.format(sum(games.values())))

  with open(out_file, 'w') as f:
    for model in sorted(elos, key=lambda k: elos[k]):
      f.write(
          '{}\t{:.0f}\t{}\n'.format(os.path.basename(model), elos[model], games[model]))
    f.write('{}\n'.format(sum(games.values())))


def get_default_elo(model):
  print('Finding default for ', model)
  for ke in SEED_ELOS:
    print('checking!: ', ke)
    if model in ke or ke in model:
      print('Found default!: ', SEED_ELOS[ke])
      return SEED_ELOS[ke]
  return 800


if __name__ == '__main__':
  out_file = sys.argv[1]
  if os.path.exists(out_file):
    with open(out_file, 'r') as f:
      SEEDS = f.read()
    for line in SEEDS.split('\n'):
      stuff = line.strip().split()
      if len(stuff) < 2:
        continue
      name, elo = stuff[0], stuff[1]
      SEED_ELOS[name] = float(elo)

  models = get_models_from_argv()

  models = list(filter(lambda m: 'continue' not in m, models))
  print(models)

  elos = {}
  games = {}
  wins = {}

  while True:
    players = random.sample(models, 2)
    black = players[0]
    white = players[1]
    if black not in elos:
      elos[black] = get_default_elo(black)
      games[black] = 0
    if white not in elos:
      elos[white] = get_default_elo(white)
      games[white] = 0
    games[white] += 1
    games[black] += 1

    play_models_and_update(black, white, elos)
    print()
    print()
    print_elos(elos, games, out_file)
    print()


