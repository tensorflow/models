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
import dual_net
import preprocessing

import glob

from utils import timer
from tensorflow import gfile
import logging

import goparams
import predict_moves

import qmeas

from mlperf_compliance import mlperf_log

# Pull in environment variables. Run `source ./cluster/common` to set these.
#BUCKET_NAME = os.environ['BUCKET_NAME']

#BASE_DIR = "gs://{}".format(BUCKET_NAME)
BASE_DIR = goparams.BASE_DIR
if os.path.isdir(BASE_DIR): # if it already exists, delete it.
    shutil.rmtree(BASE_DIR, ignore_errors=True)
os.system('mkdir ' + BASE_DIR)

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


def bootstrap():
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join(MODELS_DIR, bootstrap_name)
    print("Bootstrapping with working dir {}\n Model 0 exported to {}".format(
        ESTIMATOR_WORKING_DIR, bootstrap_model_path))
    main.bootstrap(ESTIMATOR_WORKING_DIR, bootstrap_model_path)


def echo():
    pass  # Flags are echo'd in the ifmain block below.


def main_fn():
    """Run the reinforcement learning loop

    This tries to create a realistic way to run the reinforcement learning with
    all default parameters.
    """

    if goparams.DUMMY_MODEL:
        # monkeypatch the hyperparams so that we get a quickly executing network.
        dual_net.get_default_hyperparams = lambda **kwargs: {
            'k': 8, 'fc_width': 16, 'num_shared_layers': 1, 'l2_strength': 1e-4, 'momentum': 0.9}

        dual_net.TRAIN_BATCH_SIZE = 16
        dual_net.EXAMPLES_PER_GENERATION = 64

        #monkeypatch the shuffle buffer size so we don't spin forever shuffling up positions.
        preprocessing.SHUFFLE_BUFFER_SIZE = 1000

    print("Creating random initial weights...")
    bootstrap()



if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    qmeas.start(os.path.join(BASE_DIR, 'stats'))

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

    # mlperf logging for starting the entire run
    mlperf_log.minigo_print(key=mlperf_log.RUN_START)

    main_fn()
    qmeas.end()
