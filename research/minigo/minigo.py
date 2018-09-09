# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train MiniGo with several iterations of RL learning.

One iteration of RL learning consists of bootstrap, selfplay, gather and train:
  bootstrap: Initialize a random model
  selfplay: Play games with the latest model to produce data used for training
  gather: Group games played with the same model into larger files of tfexamples
  train: Train a new model with the selfplay results from the most recent
    N generations.
After training, validation can be performed on the holdout data.
Given two models, evaluation can be applied to choose a stronger model.
The training pipeline consists of multiple RL learning iterations to achieve
better models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import socket
import sys
import time

import tensorflow as tf  # pylint: disable=g-bad-import-order

import dualnet
import evaluation
import go
import model_params
import preprocessing
import selfplay_mcts
import utils

_TF_RECORD_SUFFIX = '.tfrecord.zz'


def _ensure_dir_exists(directory):
  """Check if directory exists. If not, create it.

  Args:
    directory: A given directory
  """
  if os.path.isdir(directory) is False:
    tf.gfile.MakeDirs(directory)


def bootstrap(estimator_model_dir, trained_models_dir, params):
  """Initialize the model with random weights.

  Args:
    estimator_model_dir: tf.estimator model directory.
    trained_models_dir: Dir to save the trained models. Here to export the first
      bootstrapped generation.
    params: A MiniGoParams instance of hyperparameters for the model.
  """
  bootstrap_name = utils.generate_model_name(0)
  _ensure_dir_exists(trained_models_dir)
  bootstrap_model_path = os.path.join(trained_models_dir, bootstrap_name)
  _ensure_dir_exists(estimator_model_dir)

  print('Bootstrapping with working dir {}\n Model 0 exported to {}'.format(
      estimator_model_dir, bootstrap_model_path))
  dualnet.bootstrap(estimator_model_dir, params)
  dualnet.export_model(estimator_model_dir, bootstrap_model_path)


def selfplay(selfplay_dirs, selfplay_model, params):
  """Perform selfplay with a specific model.

  Args:
    selfplay_dirs: A dict to specify the directories used in selfplay.
      selfplay_dirs = {
          'output_dir': output_dir,
          'holdout_dir': holdout_dir,
          'clean_sgf': clean_sgf,
          'full_sgf': full_sgf
      }
    selfplay_model: The actual Dualnet runner for selfplay.
    params: A MiniGoParams instance of hyperparameters for the model.
  """
  with utils.logged_timer('Playing game'):
    player = selfplay_mcts.play(
        params.board_size, selfplay_model, params.selfplay_readouts,
        params.selfplay_resign_threshold, params.simultaneous_leaves,
        params.selfplay_verbose)

  output_name = '{}-{}'.format(int(time.time()), socket.gethostname())

  def _write_sgf_data(dir_sgf, use_comments):
    with tf.gfile.GFile(
        os.path.join(dir_sgf, '{}.sgf'.format(output_name)), 'w') as f:
      f.write(player.to_sgf(use_comments=use_comments))

  _write_sgf_data(selfplay_dirs['clean_sgf'], use_comments=False)
  _write_sgf_data(selfplay_dirs['full_sgf'], use_comments=True)

  game_data = player.extract_data()
  tf_examples = preprocessing.make_dataset_from_selfplay(game_data, params)

  # Hold out 5% of games for evaluation.
  if random.random() < params.holdout_pct:
    fname = os.path.join(
        selfplay_dirs['holdout_dir'], output_name + _TF_RECORD_SUFFIX)
  else:
    fname = os.path.join(
        selfplay_dirs['output_dir'], output_name + _TF_RECORD_SUFFIX)

  preprocessing.write_tf_examples(fname, tf_examples)


def gather(selfplay_dir, training_chunk_dir, params):
  """Gather selfplay data into large training chunk.

  Args:
    selfplay_dir: Where to look for games. Set as 'base_dir/data/selfplay/'.
    training_chunk_dir: where to put collected games. Set as
      'base_dir/data/training_chunks/'.
    params: A MiniGoParams instance of hyperparameters for the model.
  """
  # Check the selfplay data from the most recent 50 models.
  _ensure_dir_exists(training_chunk_dir)
  sorted_model_dirs = sorted(tf.gfile.ListDirectory(selfplay_dir))
  models = [model_dir.strip('/')
            for model_dir in sorted_model_dirs[-params.gather_generation:]]

  with utils.logged_timer('Finding existing tfrecords...'):
    model_gamedata = {
        model: tf.gfile.Glob(
            os.path.join(selfplay_dir, model, '*'+_TF_RECORD_SUFFIX))
        for model in models
    }
  print('Found {} models'.format(len(models)))
  for model_name, record_files in sorted(model_gamedata.items()):
    print('    {}: {} files'.format(model_name, len(record_files)))

  meta_file = os.path.join(training_chunk_dir, 'meta.txt')
  try:
    with tf.gfile.GFile(meta_file, 'r') as f:
      already_processed = set(f.read().split())
  except tf.errors.NotFoundError:
    already_processed = set()

  num_already_processed = len(already_processed)

  for model_name, record_files in sorted(model_gamedata.items()):
    if set(record_files) <= already_processed:
      continue
    print('Gathering files from {}:'.format(model_name))
    tf_examples = preprocessing.shuffle_tf_examples(
        params.shuffle_buffer_size, params.examples_per_chunk, record_files)
    # tqdm to make the loops show a smart progress meter
    for i, example_batch in enumerate(tf_examples):
      output_record = os.path.join(
          training_chunk_dir,
          ('{}-{}'+_TF_RECORD_SUFFIX).format(model_name, str(i)))
      preprocessing.write_tf_examples(
          output_record, example_batch, serialize=False)
    already_processed.update(record_files)

  print('Processed {} new files'.format(
      len(already_processed) - num_already_processed))
  with tf.gfile.GFile(meta_file, 'w') as f:
    f.write('\n'.join(sorted(already_processed)))


def train(trained_models_dir, estimator_model_dir, training_chunk_dir,
          generation, params):
  """Train the latest model from gathered data.

  Args:
    trained_models_dir: Where to export the completed generation.
    estimator_model_dir: tf.estimator model directory.
    training_chunk_dir: Directory where gathered training chunks are.
    generation: Which generation you are training.
    params: A MiniGoParams instance of hyperparameters for the model.
  """
  new_model_name = utils.generate_model_name(generation)
  print('New model will be {}'.format(new_model_name))
  new_model = os.path.join(trained_models_dir, new_model_name)

  print('Training on gathered game data...')
  tf_records = sorted(
      tf.gfile.Glob(os.path.join(training_chunk_dir, '*'+_TF_RECORD_SUFFIX)))
  tf_records = tf_records[
      -(params.train_window_size // params.examples_per_chunk):]

  print('Training from: {} to {}'.format(tf_records[0], tf_records[-1]))
  with utils.logged_timer('Training'):
    dualnet.train(estimator_model_dir, tf_records, generation, params)
    dualnet.export_model(estimator_model_dir, new_model)


def validate(trained_models_dir, holdout_dir, estimator_model_dir, params):
  """Validate the latest model on the holdout dataset.

  Args:
    trained_models_dir: Directories where the completed generations/models are.
    holdout_dir: Directories where holdout data are.
    estimator_model_dir: tf.estimator model directory.
    params: A MiniGoParams instance of hyperparameters for the model.
  """
  model_num, _ = utils.get_latest_model(trained_models_dir)

  # Get the holdout game data
  nums_names = utils.get_models(trained_models_dir)

  # Model N was trained on games up through model N-1, so the validation set
  # should only be for models through N-1 as well, thus the (model_num) term.
  models = [num_name for num_name in nums_names if num_name[0] < model_num]

  # pair is a tuple of (model_num, model_name), like (13, 000013-modelname)
  holdout_dirs = [os.path.join(holdout_dir, pair[1])
                  for pair in models[-params.holdout_generation:]]
  tf_records = []
  with utils.logged_timer('Building lists of holdout files'):
    for record_dir in holdout_dirs:
      if os.path.exists(record_dir):  # make sure holdout dir exists
        tf_records.extend(
            tf.gfile.Glob(os.path.join(record_dir, '*'+_TF_RECORD_SUFFIX)))

  if not tf_records:
    print('No holdout dataset for validation! '
          'Please check your holdout directory: {}'.format(holdout_dir))
    return

  print('The length of tf_records is {}.'.format(len(tf_records)))
  first_tf_record = os.path.basename(tf_records[0])
  last_tf_record = os.path.basename(tf_records[-1])
  with utils.logged_timer('Validating from {} to {}'.format(
      first_tf_record, last_tf_record)):
    dualnet.validate(estimator_model_dir, tf_records, params)


def evaluate(black_model_name, black_net, white_model_name, white_net,
             evaluate_dir, params):
  """Evaluate with two models.

  With two DualNetRunners to play as black and white in a Go match. Two models
  play several games, and the model that wins by a margin of 55% will be the
  winner.

  Args:
    black_model_name: The name of the model playing black.
    black_net: The DualNetRunner model for black
    white_model_name: The name of the model playing white.
    white_net: The DualNetRunner model for white.
    evaluate_dir: Where to write the evaluation results. Set as
      'base_dir/sgf/evaluate/'.
    params: A MiniGoParams instance of hyperparameters for the model.

  Returns:
    The model name of the winner.

  Raises:
      ValueError: if neither `WHITE` or `BLACK` is returned.
  """
  with utils.logged_timer('{} games'.format(params.eval_games)):
    winner = evaluation.play_match(
        params, black_net, white_net, params.eval_games,
        params.eval_readouts, evaluate_dir, params.eval_verbose)

  if winner != go.WHITE_NAME and winner != go.BLACK_NAME:
    raise ValueError('Winner should be either White or Black!')

  return black_model_name if winner == go.BLACK_NAME else white_model_name


def _set_params(flags):
  """Set hyperparameters from board size.

  Args:
    flags: Flags from Argparser.

  Returns:
  An MiniGoParams instance of hyperparameters.
  """
  params = model_params.MiniGoParams()
  k = utils.round_power_of_two(flags.board_size ** 2 / 3)
  params.num_filters = k  # Number of filters in the convolution layer
  params.fc_width = 2 * k  # Width of each fully connected layer
  params.num_shared_layers = flags.board_size  # Number of shared trunk layers
  params.board_size = flags.board_size  # Board size

  # How many positions can fit on a graphics card. 256 for 9s, 16 or 32 for 19s.
  if flags.batch_size is None:
    if flags.board_size == 9:
      params.batch_size = 256
    else:
      params.batch_size = 32
  else:
    params.batch_size = flags.batch_size

  return params


def _prepare_selfplay(
    model_name, trained_models_dir, selfplay_dir, holdout_dir, sgf_dir, params):
  """Set directories and load the network for selfplay.

  Args:
    model_name: The name of the model for self-play
    trained_models_dir: Directories where the completed generations/models are.
    selfplay_dir: Where to write the games. Set as 'base_dir/data/selfplay/'.
    holdout_dir: Where to write the holdout data. Set as
      'base_dir/data/holdout/'.
    sgf_dir: Where to write the sgf (Smart Game Format) files. Set as
      'base_dir/sgf/'.
    params: A MiniGoParams instance of hyperparameters for the model.

  Returns:
    The directories and network model for selfplay.
  """
  # Set paths for the model with 'model_name'
  model_path = os.path.join(trained_models_dir, model_name)
  output_dir = os.path.join(selfplay_dir, model_name)
  holdout_dir = os.path.join(holdout_dir, model_name)
  # clean_sgf is to write sgf file without comments.
  # full_sgf is to write sgf file with comments.
  clean_sgf = os.path.join(sgf_dir, model_name, 'clean')
  full_sgf = os.path.join(sgf_dir, model_name, 'full')

  _ensure_dir_exists(output_dir)
  _ensure_dir_exists(holdout_dir)
  _ensure_dir_exists(clean_sgf)
  _ensure_dir_exists(full_sgf)
  selfplay_dirs = {
      'output_dir': output_dir,
      'holdout_dir': holdout_dir,
      'clean_sgf': clean_sgf,
      'full_sgf': full_sgf
  }
  # cache the network model for self-play
  with utils.logged_timer('Loading weights from {} ... '.format(model_path)):
    network = dualnet.DualNetRunner(model_path, params)
  return selfplay_dirs, network


def run_selfplay(selfplay_model, selfplay_games, dirs, params):
  """Run selfplay to generate training data.

  Args:
    selfplay_model: The model name for selfplay.
    selfplay_games: The number of selfplay games.
    dirs: A MiniGoDirectory instance of directories used in each step.
    params: A MiniGoParams instance of hyperparameters for the model.
  """
  selfplay_dirs, network = _prepare_selfplay(
      selfplay_model, dirs.trained_models_dir, dirs.selfplay_dir,
      dirs.holdout_dir, dirs.sgf_dir, params)

  print('Self-play with model: {}'.format(selfplay_model))
  for _ in range(selfplay_games):
    selfplay(selfplay_dirs, network, params)


def main(_):
  """Run the reinforcement learning loop."""
  tf.logging.set_verbosity(tf.logging.INFO)

  params = _set_params(FLAGS)

  # A dummy model for debug/testing purpose with fewer games and iterations
  if FLAGS.test:
    params = model_params.DummyMiniGoParams()
    base_dir = FLAGS.base_dir + str(FLAGS.board_size) + '_size_dummy/'
  else:
    # Set directories for models and datasets
    base_dir = FLAGS.base_dir + str(FLAGS.board_size) + '_size/'

  dirs = utils.MiniGoDirectory(base_dir)

  # Run selfplay only if user specifies the argument.
  if FLAGS.selfplay:
    selfplay_model_name = FLAGS.selfplay_model_name or utils.get_latest_model(
        dirs.trained_models_dir)[1]
    max_games = FLAGS.selfplay_max_games or params.max_games_per_generation
    run_selfplay(selfplay_model_name, max_games, dirs, params)
    return

  # Run the RL pipeline
  # if no models have been trained, start from bootstrap model

  if not os.path.isdir(dirs.trained_models_dir):
    print('No trained model exists! Starting from Bootstrap...')
    print('Creating random initial weights...')
    bootstrap(dirs.estimator_model_dir, dirs.trained_models_dir, params)
  else:
    print('A MiniGo base directory has been found! ')
    print('Start from the last checkpoint...')

  _, best_model_so_far = utils.get_latest_model(dirs.trained_models_dir)
  for rl_iter in range(params.max_iters_per_pipeline):
    print('RL_iteration: {}'.format(rl_iter))
    # Self-play with the best model to generate training data
    run_selfplay(
        best_model_so_far, params.max_games_per_generation, dirs, params)

    # gather selfplay data for training
    print('Gathering game output...')
    gather(dirs.selfplay_dir, dirs.training_chunk_dir, params)

    # train the next generation model
    model_num, _ = utils.get_latest_model(dirs.trained_models_dir)
    print('Training on gathered game data...')
    train(dirs.trained_models_dir, dirs.estimator_model_dir,
          dirs.training_chunk_dir, model_num + 1, params)

    # validate the latest model if needed
    if FLAGS.validation:
      print('Validating on the holdout game data...')
      validate(dirs.trained_models_dir, dirs.holdout_dir,
               dirs.estimator_model_dir, params)

    _, current_model = utils.get_latest_model(dirs.trained_models_dir)

    if FLAGS.evaluation:  # Perform evaluation if needed
      print('Evaluate models between {} and {}'.format(
          best_model_so_far, current_model))
      black_model = os.path.join(dirs.trained_models_dir, best_model_so_far)
      white_model = os.path.join(dirs.trained_models_dir, current_model)
      _ensure_dir_exists(dirs.evaluate_dir)
      with utils.logged_timer('Loading weights'):
        black_net = dualnet.DualNetRunner(black_model, params)
        white_net = dualnet.DualNetRunner(white_model, params)

      best_model_so_far = evaluate(
          best_model_so_far, black_net, current_model, white_net,
          dirs.evaluate_dir, params)
      print('Winner of evaluation: {}!'.format(best_model_so_far))
    else:
      best_model_so_far = current_model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # flags to run the RL pipeline
  parser.add_argument(
      '--base_dir',
      type=str,
      default='/tmp/minigo/',
      metavar='BD',
      help='Base directory for the MiniGo models and datasets.')
  parser.add_argument(
      '--board_size',
      type=int,
      default=9,
      metavar='N',
      choices=[9, 19],
      help='Go board size. The default size is 9.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=None,
      metavar='BS',
      help='Batch size for training. The default size is None')
  # Test the pipeline with a dummy model
  parser.add_argument(
      '--test',
      action='store_true',
      help='A boolean to test RL pipeline with a dummy model.')
  # Run RL pipeline with the validation step
  parser.add_argument(
      '--validation',
      action='store_true',
      help='A boolean to specify validation in the RL pipeline.')
  # Run RL pipeline with the evaluation step
  parser.add_argument(
      '--evaluation',
      action='store_true',
      help='A boolean to specify evaluation in the RL pipeline.')

  # self-play only
  parser.add_argument(
      '--selfplay',
      action='store_true',
      help='A boolean to run self-play only.')
  parser.add_argument(
      '--selfplay_model_name',
      type=str,
      default=None,
      metavar='SM',
      help='The model used for self-play only.')
  parser.add_argument(
      '--selfplay_max_games',
      type=int,
      default=None,
      metavar='SMG',
      help='The number of game data self-play only needs to generate')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
