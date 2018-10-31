#!/usr/bin/env python
"""TODO(vbittorf): DO NOT SUBMIT without one-line documentation for qmeas_analysis.

TODO(vbittorf): DO NOT SUBMIT without a detailed description of qmeas_analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import glob
import os
import time
import math

import sys

def str_timer(t):
  return '{}: {:.3f} / {} ({:.3f})'.format(t['name'], t['time'], t['run'], t['total'])

KEY = 'train'


def selfplay_stats(jsons):
  sp_times = []
  for j in jsons:
    if j['type'] != 'timer':
      continue
    if j['event'] != 'stop':
      continue
    if j['name'] != 'selfplay':
      continue
    sp_times.append(j['times']['selfplay']['time'])
  print('Average Selfplay Time: {:.2f}s'.format(sum(sp_times) / len(sp_times)))


def proporition_times(jsons):
  last = None
  for j in jsons:
    if j['type'] != 'timer':
      continue
    if j['event'] != 'stop':
      continue
    last = j
  if last is not None:
    tvals = {}
    for name in last['times']:
      timer = last['times'][name]
      tvals[name] = timer['total']
    print()
    print('Proportional Times')
    for name in reversed(sorted(tvals, key=lambda n: tvals[n])):
      secs = tvals[name]
      print('{:<20}{:7.1f}%{:>30.1f}'.format(name, secs / max(tvals.values()) * 100, secs))


def puzzle_quality(jsons, lowest_wall):
  print()
  print('Puzzle Quality')
  print('Main Time (h)\tPct Correct')
  for j in jsons:
    if j['type'] != 'value':
      continue
    if j['name'] != 'puzzle_total':
      continue
    print('{:f}\t{:.1f}%'.format((j['wall_time'] - lowest_wall) / 60 / 60, 100 * float(j['value'])))


def reject_times(jsons, lowest_wall):
  print()
  print('Reject Times')
  print('Main Time (h)\tRejected')
  for j in jsons:
    if j['type'] != 'value':
      continue
    if j['name'] != 'evaluate_choice':
      continue
    if j['value'] == 'new':
      continue
    print('{:f}\t{}%'.format((j['wall_time'] - lowest_wall), 1))


def eval_quality(jsons, lowest_wall):
  print()
  print('Eval Quality')
  print('Main Time (h)\tWin Pct')
  for j in jsons:
    if j['type'] != 'value':
      continue
    if j['name'] != 'evaluate_win_pct':
      continue
    print('{:f}\t{:.1f}%'.format((j['wall_time'] - lowest_wall) / 60 / 60, 100 * float(j['value'])))


def puzzle_results(jsons):
  print()
  print()
  for j in jsons:
    if j['type'] != 'value':
      continue
    if j['name'] != 'puzzle_result':
      continue
    res = eval(j['value'])
    print()
    print(j['times']['main']['time'])
    for puz in sorted(res):
      correct_count = len(list(filter(lambda t: t[2], res[puz])))
      print('{:>2.0f}%    {}'.format(correct_count * 100.0 / len(res[puz]), puz))


def get_lowest_wall_time(jsons):
  lowest_wall = None
  for j in jsons:
    if lowest_wall is None:
      lowest_wall = j['wall_time']
    if lowest_wall > j['wall_time']:
      lowest_wall = j['wall_time']
  return lowest_wall


def get_biggest_wall_time(jsons):
  lowest_wall = None
  for j in jsons:
    if lowest_wall is None:
      lowest_wall = j['wall_time']
    if lowest_wall < j['wall_time']:
      lowest_wall = j['wall_time']
  return lowest_wall


def improved_factor(jsons, lowest_wall):
  # "name": "eval_summary", "type": "value", "value": {"model": "000004-key-weasel", "win_pct": 0.51, "keep": false}
  print()
  print()
  fact = 1
  for j in jsons:
    if j['type'] != 'value':
      continue
    if j['name'] != 'eval_summary':
      continue
    res = j['value']
    if res['keep']:
      p_win = res['win_pct']
      fact *= (p_win / (1.0 - p_win))
      hours = (j['wall_time'] - lowest_wall) / 60 / 60
      print('  {:<20}    {:>5.1f}  (+{:3.1f}%)  {:3.0f}h'.format(res['model'], math.log(fact, 10), res['win_pct'] * 100, hours))


def main():
  dirname = sys.argv[1]
  jsons = []

  for filename in glob.glob(os.path.join(dirname, '*.json')):
    with open(filename) as f:
      for l in f:
        jsons.append(json.loads(l))

  jsons = list(sorted(jsons, key=lambda t: t['wall_time']))

  lowest_wall_time = get_lowest_wall_time(jsons)
  biggest_wall_time = get_biggest_wall_time(jsons)
  last_profiler = None
  #selfplay_stats(jsons)
  proporition_times(jsons)
  puzzle_quality(jsons, lowest_wall_time)
  eval_quality(jsons, lowest_wall_time)
  improved_factor(jsons, lowest_wall_time)
  # puzzle_results(jsons)

  print()
  print()
  print('Total Duration: {:.2f} hours'.format((biggest_wall_time - lowest_wall_time) / 60 / 60))
  print('Last update: {:.2f} hours ago'.format((time.time() - biggest_wall_time) / 60 / 60))


  '''
  for j in jsons:
    if j['type'] == 'profile':
      last_profiler = j['val']
    if j['type'] != 'timer':
      continue
    if j['event'] != 'stop':
      continue
    if j['name'] != KEY:
      continue
    for t in j['times']:
      if t['name'] == KEY:
        print(str_timer(t))
    for t in sorted(j['times'], key=lambda x: x['name']):
      if t['name'] != KEY:
        print('  ', str_timer(t))
  '''

  '''
  if last_profiler is not None:
    print()
    print()
    print(last_profiler)
    print()
  '''



if __name__ == '__main__':
  main()
