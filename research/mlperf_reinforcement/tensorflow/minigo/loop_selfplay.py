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
import os.path
import shipname
import sys
import time
import shutil
import dual_net
import preprocessing
import subprocess
import multiprocessing

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
SELFPLAY_BACKUP_DIR = os.path.join(BASE_DIR, 'data/selfplay_backup')
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
        'SELFPLAY_BACKUP_DIR': SELFPLAY_BACKUP_DIR,
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

    if sys.argv[3]=='worker' or sys.argv[3]=='driver':
        selfplay_dir = os.path.join(SELFPLAY_DIR, model_name)
    else:
        selfplay_dir = SELFPLAY_BACKUP_DIR

    def count_live_procs():
      return len(list(filter(lambda proc: proc.poll() is None, procs)))
    def start_worker(num_workers):
      worker_seed = hash(hash(SEED) + ITERATION) + num_workers
      cmd = 'GOPARAMS={} OMP_NUM_THREADS=1 KMP_HW_SUBSET={} KMP_AFFINITY=granularity=fine,proclist=[{}],explicit python3 selfplay_worker.py {} {} {}'.format(os.environ['GOPARAMS'], os.environ['KMP_HW_SUBSET'], num_workers%multiprocessing.cpu_count(), BASE_DIR, worker_seed, sys.argv[3])
      procs.append(subprocess.Popen(cmd, shell=True))

    def count_games():
      # returns number of games in the selfplay directory
      if not os.path.exists(selfplay_dir):
        # directory not existing implies no games have been played yet
        return 0
      return len(gfile.Glob(os.path.join(selfplay_dir, '*.zz')))

    # generate selfplay games until needed number of games reached
    if sys.argv[3]=='worker':
        for i in range(goparams.NUM_PARALLEL_SELFPLAY):
            print('Starting Worker...')
            start_worker(num_workers)
            time.sleep(0.1)
            num_workers += 1
        sys.stdout.flush()

        while count_games() < MAX_GAMES_PER_GENERATION and not os.path.isfile("PK_FLAG"):
            time.sleep(1)
            games = count_games()
            sys.stdout.flush()


        print('Done with selfplay loop.')


        for proc in procs:
            proc.kill()

        # Sometimes the workers need extra help...
        os.system('pkill -f selfplay_worker.py')

        sys.stdout.flush()

    # check generated games, remove exssesive games
    if sys.argv[3]=='driver':
        # Because we use process level parallelism for selfpaying and we don't
        # sync or communicate between processes, there could be too many games
        # played (up to 1 extra game per worker process).
        # This is a rather brutish way to ensure we train on the correct number
        # of games...
        print('There are {} games in the selfplay directory at {}'.format(count_games(), selfplay_dir))
        sys.stdout.flush()
        while count_games() > MAX_GAMES_PER_GENERATION:
            games = count_games()
            print('Too many selfplay games ({}/{}) ... deleting extra'.format(games, MAX_GAMES_PER_GENERATION))
            # This will remove exactly one game file from the selfplay directory... or
            # so we hope :)
            sys.stdout.flush()
            os.system('ls {}/* -d | tail -n {} | xargs rm '.format(selfplay_dir, games-MAX_GAMES_PER_GENERATION))
        print('After cleanup, there are {} games in the selfplay directory at {}'.format(count_games(), selfplay_dir))
        sys.stdout.flush()

    # generate backup games, in case the new model will be buried and we need more old games for training
    if sys.argv[3]=='backup':
        for i in range(goparams.NUM_PARALLEL_SELFPLAY):
            print('Starting Worker...')
            start_worker(num_workers)
            num_workers += 1
        sys.stdout.flush()

        while count_games() < MAX_GAMES_PER_GENERATION:
            time.sleep(1)
            games = count_games()
            sys.stdout.flush()


        print('Done with selfplay loop.')


        for proc in procs:
            proc.kill()

        # Sometimes the workers need extra help...
        os.system('pkill -f selfplay_worker.py')

        sys.stdout.flush()

    if sys.argv[3]=='clean_backup':
        print('cleaning up {}'.format(SELFPLAY_BACKUP_DIR))
        os.system('rm {}/*'.format(SELFPLAY_BACKUP_DIR))
    qmeas.stop_time('selfplay_wait')


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    qmeas.start(os.path.join(BASE_DIR, 'stats'))

    # make sure seed is small enough
    SEED = int(sys.argv[1])%65536
    # make sure ITERATION is small enough because we use nanosecond as ITERATION
    ITERATION = int(sys.argv[2])%65536

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
    if len(sys.argv)>3:
        main_()
    qmeas.end()
