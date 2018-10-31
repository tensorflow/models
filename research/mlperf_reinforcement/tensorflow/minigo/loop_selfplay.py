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
import shipname
import sys
import time
import shutil
import dual_net
import preprocessing
import subprocess

import glob
from tensorflow import gfile

from utils import timer
import logging

import goparams
import predict_moves

import qmeas

SEED = None
ITERATION = None

# Pull in environment variables. Run `source ./cluster/common` to set these.
#BUCKET_NAME = os.environ['BUCKET_NAME']

#BASE_DIR = "gs://{}".format(BUCKET_NAME)
BASE_DIR = goparams.BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, 'models')
SELFPLAY_DIR = os.path.join(BASE_DIR, 'data/selfplay')
BURY_DIR = os.path.join(BASE_DIR, 'bury_models')
HOLDOUT_DIR = os.path.join(BASE_DIR, 'data/holdout')
SGF_DIR = os.path.join(BASE_DIR, 'sgf')
TRAINING_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'training_chunks')

ESTIMATOR_WORKING_DIR = os.path.join(BASE_DIR, 'estimator_working_dir')

# How many games before the selfplay workers will stop trying to play more.
MAX_GAMES_PER_GENERATION = goparams.MAX_GAMES_PER_GENERATION

# What percent of games to holdout from training per generation

HOLDOUT_PCT = goparams.HOLDOUT_PCT


def print_flags():
    flags = {
        #'BUCKET_NAME': BUCKET_NAME,
        'BASE_DIR': BASE_DIR,
        'MODELS_DIR': MODELS_DIR,
        'SELFPLAY_DIR': SELFPLAY_DIR,
        'HOLDOUT_DIR': HOLDOUT_DIR,
        'SGF_DIR': SGF_DIR,
        'TRAINING_CHUNK_DIR': TRAINING_CHUNK_DIR,
        'ESTIMATOR_WORKING_DIR': ESTIMATOR_WORKING_DIR,
    }
    print("Computed variables are:")
    print('\n'.join('--{}={}'.format(flag, value)
                    for flag, value in flags.items()))


def get_models():
    """Finds all models, returning a list of model number and names
    sorted increasing.

    Returns: [(13, 000013-modelname), (17, 000017-modelname), ...etc]
    """
    all_models = gfile.Glob(os.path.join(MODELS_DIR, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = sorted([
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames])
    return model_numbers_names


def get_latest_model():
    """Finds the latest model, returning its model number and name

    Returns: (17, 000017-modelname)
    """
    models = get_models()
    if len(models) == 0:
        models = [(0, '000000-bootstrap')]
    return models[-1]



def main_():
    """Run the reinforcement learning loop

    This tries to create a realistic way to run the reinforcement learning with
    all default parameters.
    """
    print('Starting self play loop.')

    qmeas.start_time('selfplay_wait')
    start_t = time.time()

    _, model_name = get_latest_model()

    num_workers = 0

    procs = [
    ]
    def count_live_procs():
      return len(list(filter(lambda proc: proc.poll() is None, procs)))
    def start_worker(num_workers):
      #procs.append(subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE))
      worker_seed = hash(hash(SEED) + ITERATION) + num_workers
      cmd = 'GOPARAMS={} python3 selfplay_worker.py {} {}'.format(os.environ['GOPARAMS'], BASE_DIR, worker_seed)
      procs.append(subprocess.Popen(cmd, shell=True))

    selfplay_dir = os.path.join(SELFPLAY_DIR, model_name)
    def count_games():
      # returns number of games in the selfplay directory
      if not os.path.exists(os.path.join(SELFPLAY_DIR, model_name)):
        # directory not existing implies no games have been played yet
        return 0
      return len(gfile.Glob(os.path.join(SELFPLAY_DIR, model_name, '*.zz')))


    for i in range(goparams.NUM_PARALLEL_SELFPLAY):
      print('Starting Worker...')
      num_workers += 1
      start_worker(num_workers)
      time.sleep(1)
    sys.stdout.flush()

    while count_games() < MAX_GAMES_PER_GENERATION:
        time.sleep(10)
        games = count_games()
        print('Found Games: {}'.format(games))
        print('selfplaying: {:.2f} games/hour'.format(games / ((time.time() - start_t) / 60 / 60) ))
        print('Worker Processes: {}'.format(count_live_procs()))
        sys.stdout.flush()


    print('Done with selfplay loop.')

    time.sleep(10)

    for proc in procs:
      proc.kill()

    # Sometimes the workers need extra help...
    time.sleep(5)
    os.system('pkill -f selfplay_worker.py')

    # Let things settle after we kill processes.
    time.sleep(10)

    # Because we use process level parallelism for selfpaying and we don't
    # sync or communicate between processes, there could be too many games
    # played (up to 1 extra game per worker process).
    # This is a rather brutish way to ensure we train on the correct number
    # of games...
    print('There are {} games in the selfplay directory at {}'.format(count_games(), selfplay_dir))
    sys.stdout.flush()
    while count_games() > MAX_GAMES_PER_GENERATION:
      print('Too many selfplay games ({}/{}) ... deleting one'.format(count_games(), MAX_GAMES_PER_GENERATION))
      # This will remove exactly one game file from the selfplay directory... or
      # so we hope :)
      sys.stdout.flush()
      os.system('ls {}/* -d | tail -n 1 | xargs rm'.format(selfplay_dir))
      # unclear if this sleep is necessary...
      time.sleep(1)
    print('After cleanup, there are {} games in the selfplay directory at {}'.format(count_games(), selfplay_dir))
    sys.stdout.flush()

    qmeas.stop_time('selfplay_wait')


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    qmeas.start(os.path.join(BASE_DIR, 'stats'))

    SEED = int(sys.argv[1])
    ITERATION = int(sys.argv[2])

    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    main_()
    qmeas.end()
