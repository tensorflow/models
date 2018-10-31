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

import argparse
import argh
import os.path
import collections
import random
import re
import shutil
import socket
import sys
import tempfile
import time
import cloud_logging
from tqdm import tqdm
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import gfile

import go
import dual_net
from gtp_wrapper import make_gtp_instance, MCTSPlayer
import preprocessing
import selfplay_mcts
from utils import logged_timer as timer
import evaluation
import sgf_wrapper
import utils

import qmeas
import goparams

# How many positions we should aggregate per 'chunk'.
EXAMPLES_PER_RECORD = goparams.EXAMPLES_PER_RECORD

# How many positions to draw from for our training window.
# AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
# WINDOW_SIZE = 125000000
#WINDOW_SIZE = 500000
WINDOW_SIZE = goparams.WINDOW_SIZE


def _ensure_dir_exists(directory):
    if directory.startswith('gs://'):
        return
    os.makedirs(directory, exist_ok=True)


def gtp(load_file: "The path to the network model files"=None,
        readouts: 'How many simulations to run per move'=10000,
        #readouts: 'How many simulations to run per move'=2000,
        cgos_mode: 'Whether to use CGOS time constraints'=False,
        verbose=1):
    engine = make_gtp_instance(load_file,
                               readouts_per_move=readouts,
                               verbosity=verbose,
                               cgos_mode=cgos_mode)
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()


def bootstrap(
        working_dir: 'tf.estimator working directory. If not set, defaults to a random tmp dir'=None,
        model_save_path: 'Where to export the first bootstrapped generation'=None):
    qmeas.start_time('bootstrap')
    if working_dir is None:
        with tempfile.TemporaryDirectory() as working_dir:
            _ensure_dir_exists(working_dir)
            _ensure_dir_exists(os.path.dirname(model_save_path))
            dual_net.bootstrap(working_dir)
            dual_net.export_model(working_dir, model_save_path)
    else:
        _ensure_dir_exists(working_dir)
        _ensure_dir_exists(os.path.dirname(model_save_path))
        dual_net.bootstrap(working_dir)
        dual_net.export_model(working_dir, model_save_path)
    qmeas.stop_time('bootstrap')


def train(
        working_dir: 'tf.estimator working directory.',
        chunk_dir: 'Directory where gathered training chunks are.',
        model_save_path: 'Where to export the completed generation.',
        generation_num: 'Which generation you are training.'=0):
    qmeas.start_time('train')
    tf_records = sorted(gfile.Glob(os.path.join(chunk_dir, '*.tfrecord.zz')))
    tf_records = tf_records[-1 * (WINDOW_SIZE // EXAMPLES_PER_RECORD):]

    print("Training from:", tf_records[0], "to", tf_records[-1])

    with timer("Training"):
        dual_net.train(working_dir, tf_records, generation_num)
        dual_net.export_model(working_dir, model_save_path)
    qmeas.stop_time('train')


def validate(
        working_dir: 'tf.estimator working directory',
        *tf_record_dirs: 'Directories where holdout data are',
        checkpoint_name: 'Which checkpoint to evaluate (None=latest)'=None,
        validate_name: 'Name for validation set (i.e., selfplay or human)'=None):
    qmeas.start_time('validate')
    tf_records = []
    with timer("Building lists of holdout files"):
        for record_dir in tf_record_dirs:
            tf_records.extend(gfile.Glob(os.path.join(record_dir, '*.zz')))

    first_record = os.path.basename(tf_records[0])
    last_record = os.path.basename(tf_records[-1])
    with timer("Validating from {} to {}".format(first_record, last_record)):
        dual_net.validate(
            working_dir, tf_records, checkpoint_name=checkpoint_name,
            name=validate_name)
    qmeas.stop_time('validate')


def evaluate(
        black_model: 'The path to the model to play black',
        white_model: 'The path to the model to play white',
        output_dir: 'Where to write the evaluation results'='sgf/evaluate',
        readouts: 'How many readouts to make per move.'=200,
        games: 'the number of games to play'=20,
        verbose: 'How verbose the players should be (see selfplay)' = 1):
    qmeas.start_time('evaluate')
    _ensure_dir_exists(output_dir)

    with timer("Loading weights"):
        black_net = dual_net.DualNetwork(black_model)
        white_net = dual_net.DualNetwork(white_model)

    winners = []
    with timer("%d games" % games):
        winners = evaluation.play_match(
            black_net, white_net, games, readouts, output_dir, verbose)
    qmeas.stop_time('evaluate')
    white_count = 0
    for win in winners:
      if 'W' in win or 'w' in win:
        white_count += 1
    return white_count * 1.0 / games

    # qmeas.report_profiler()

def evaluate_evenly(
        black_model: 'The path to the model to play black',
        white_model: 'The path to the model to play white',
        output_dir: 'Where to write the evaluation results'='sgf/evaluate',
        readouts: 'How many readouts to make per move.'=200,
        games: 'the number of games to play'=20,
        verbose: 'How verbose the players should be (see selfplay)' = 1):
  ''' Returns the white win rate; playes 'games' number of games on both sides. '''
  try:
    result = (evaluate(black_model, white_model, output_dir, readouts, games, verbose) + (1 - evaluate(white_model, black_model, output_dir, readouts, games, verbose)))/ 2.0
  except TypeError:
    # It is remotely possible that in weird twist of fate results in a type
    # error... Possibly due to weird corner cases in the evaluation...
    # Our fall back will be to try agian.
    result = (evaluate(black_model, white_model, output_dir, readouts, games, verbose) + (1 - evaluate(white_model, black_model, output_dir, readouts, games, verbose)))/ 2.0
    # should this really happen twice, the world really doesn't
    # want this to be successful... and we will raise the error.
    # If this is being run by the main loop harness, then the
    # effect of raising here will be to keep the newest model and go back to
    # selfplay.
  return result



def selfplay(
        load_file: "The path to the network model files",
        output_dir: "Where to write the games"="data/selfplay",
        holdout_dir: "Where to write the games"="data/holdout",
        output_sgf: "Where to write the sgfs"="sgf/",
        readouts: 'How many simulations to run per move'=100,
        verbose: '>=2 will print debug info, >=3 will print boards' = 1,
        resign_threshold: 'absolute value of threshold to resign at' = 0.95,
        holdout_pct: 'how many games to hold out for validation' = 0.05):
    qmeas.start_time('selfplay')
    clean_sgf = os.path.join(output_sgf, 'clean')
    full_sgf = os.path.join(output_sgf, 'full')
    _ensure_dir_exists(clean_sgf)
    _ensure_dir_exists(full_sgf)
    _ensure_dir_exists(output_dir)
    _ensure_dir_exists(holdout_dir)

    with timer("Loading weights from %s ... " % load_file):
        network = dual_net.DualNetwork(load_file)

    with timer("Playing game"):
        player = selfplay_mcts.play(
            network, readouts, resign_threshold, verbose)

    output_name = '{}-{}'.format(int(time.time() * 1000 * 1000), socket.gethostname())
    game_data = player.extract_data()
    with gfile.GFile(os.path.join(clean_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf(use_comments=False))
    with gfile.GFile(os.path.join(full_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf())

    tf_examples = preprocessing.make_dataset_from_selfplay(game_data)

    # Hold out 5% of games for evaluation.
    if random.random() < holdout_pct:
        fname = os.path.join(holdout_dir, "{}.tfrecord.zz".format(output_name))
    else:
        fname = os.path.join(output_dir, "{}.tfrecord.zz".format(output_name))

    preprocessing.write_tf_examples(fname, tf_examples)
    qmeas.stop_time('selfplay')


def selfplay_cache_model(
        network: "The path to the network model files",
        output_dir: "Where to write the games"="data/selfplay",
        holdout_dir: "Where to write the games"="data/holdout",
        output_sgf: "Where to write the sgfs"="sgf/",
        readouts: 'How many simulations to run per move'=100,
        verbose: '>=2 will print debug info, >=3 will print boards' = 1,
        resign_threshold: 'absolute value of threshold to resign at' = 0.95,
        holdout_pct: 'how many games to hold out for validation' = 0.05):
    qmeas.start_time('selfplay')
    clean_sgf = os.path.join(output_sgf, 'clean')
    full_sgf = os.path.join(output_sgf, 'full')
    _ensure_dir_exists(clean_sgf)
    _ensure_dir_exists(full_sgf)
    _ensure_dir_exists(output_dir)
    _ensure_dir_exists(holdout_dir)

    with timer("Playing game"):
        player = selfplay_mcts.play(
            network, readouts, resign_threshold, verbose)

    output_name = '{}-{}'.format(int(time.time() * 1000 * 1000), socket.gethostname())
    game_data = player.extract_data()
    with gfile.GFile(os.path.join(clean_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf(use_comments=False))
    with gfile.GFile(os.path.join(full_sgf, '{}.sgf'.format(output_name)), 'w') as f:
        f.write(player.to_sgf())

    tf_examples = preprocessing.make_dataset_from_selfplay(game_data)

    # Hold out 5% of games for evaluation.
    if random.random() < holdout_pct:
        fname = os.path.join(holdout_dir, "{}.tfrecord.zz".format(output_name))
    else:
        fname = os.path.join(output_dir, "{}.tfrecord.zz".format(output_name))

    preprocessing.write_tf_examples(fname, tf_examples)
    qmeas.stop_time('selfplay')



def gather(
        input_directory: 'where to look for games'='data/selfplay/',
        output_directory: 'where to put collected games'='data/training_chunks/',
        examples_per_record: 'how many tf.examples to gather in each chunk'=EXAMPLES_PER_RECORD):
    qmeas.start_time('gather')
    _ensure_dir_exists(output_directory)
    models = [model_dir.strip('/')
              for model_dir in sorted(gfile.ListDirectory(input_directory))[-50:]]
    with timer("Finding existing tfrecords..."):
        model_gamedata = {
            model: gfile.Glob(
                os.path.join(input_directory, model, '*.tfrecord.zz'))
            for model in models
        }
    print("Found %d models" % len(models))
    for model_name, record_files in sorted(model_gamedata.items()):
        print("    %s: %s files" % (model_name, len(record_files)))

    meta_file = os.path.join(output_directory, 'meta.txt')
    try:
        with gfile.GFile(meta_file, 'r') as f:
            already_processed = set(f.read().split())
    except tf.errors.NotFoundError:
        already_processed = set()

    num_already_processed = len(already_processed)

    for model_name, record_files in sorted(model_gamedata.items()):
        if set(record_files) <= already_processed:
            continue
        print("Gathering files for %s:" % model_name)
        for i, example_batch in enumerate(
                tqdm(preprocessing.shuffle_tf_examples(examples_per_record, record_files))):
            output_record = os.path.join(output_directory,
                                         '{}-{}.tfrecord.zz'.format(model_name, str(i)))
            preprocessing.write_tf_examples(
                output_record, example_batch, serialize=False)
        already_processed.update(record_files)

    print("Processed %s new files" %
          (len(already_processed) - num_already_processed))
    with gfile.GFile(meta_file, 'w') as f:
        f.write('\n'.join(sorted(already_processed)))
    qmeas.stop_time('gather')


parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, bootstrap, train,
                           selfplay, gather, evaluate, validate])

if __name__ == '__main__':
    cloud_logging.configure()
    argh.dispatch(parser)
