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
import argh
import argparse
import cloud_logging
import logging
import os
import main
import shipname
import sys
import time
import shutil

from utils import timer
from tensorflow import gfile
import logging

import qmeas
import glob


def get_models_from_argv():
  unglob = sys.argv[1:]
  models = []
  for m in unglob:
    models.extend(glob.glob(m))
  models = map(lambda s: '.'.join(s.split('.')[:-1]), models)
  models = list(sorted(list(set(models))))
  print(models)
  return models


if __name__ == '__main__':
  #qmeas.start()
  #qmeas.create_main_profiler()
  #white_model = sys.argv[1]
  #black_model = sys.argv[2]
  #print('whtie = ', white_model)
  #print('black = ', black_model)

  models = get_models_from_argv()

  curve = []

  for i in range(len(models)):
    output_dir = '/tmp/play_models'
    os.system('mkdir ' + output_dir);
    last_win = main.evaluate_evenly(models[i], models[-1], output_dir=output_dir, games=10)
    curve.append(1 - last_win)
    print('CURVE:')
    for i, c in enumerate(curve):
      print(i, c)

  for i, v in enumerate(curve):
    print('{}\t{}'.format(i, v))

  #qmeas.record_profiler()
  #qmeas.end()


