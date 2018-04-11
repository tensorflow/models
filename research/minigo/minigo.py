# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
After training, validation can be preformed on the holdout data. Besides, given
two models, evaluation can be applied to choose a stronger model.
The training pipeline consists of multiple RL learning iterations to achieve
better models.
"""

# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import shutil
import socket
import time
from tqdm import tqdm

import tensorflow as tf

import dual_net
import evaluation
import model_utils
import preprocessing
import selfplay_mcts
import utils


def _ensure_dir_exists(directory):
  """Check if directory exists. If not, create it.

  Args:
    directory: A given directory
  """
  if directory.startswith('gs://'):  # for cloud logging
    return
  tf.gfile.MakeDirs(directory)


def bootstrap(estimator_working_dir, models_dir, params):
  """Initialize the model with random weights.

  Args:
    estimator_working_dir: tf.estimator working directory.
    models_dir: Where to export the first bootstrapped generation.
    params: Hyper params of the model.
  """
  bootstrap_name = model_utils.generate(0)
  bootstrap_model_path = os.path.join(models_dir, bootstrap_name)

  _ensure_dir_exists(estimator_working_dir)
  _ensure_dir_exists(os.path.dirname(bootstrap_model_path))

  print('Bootstrapping with working dir %s\n Model 0 exported to %s' %
        (estimator_working_dir, bootstrap_model_path))
  dual_net.bootstrap(estimator_working_dir, params)
  dual_net.export_model(estimator_working_dir, bootstrap_model_path)


def selfplay(model_name, models_dir, selfplay_dir, holdout_dir, sgf_dir,
             params):
  """Perform selfplay with a specific model.

  Args:
    model_name: The name of the model used for selfplay.
    models_dir: The path to the network model files.
    selfplay_dir: Where to write the games'='data/selfplay/.
    holdout_dir: Where to write the hold out data'='data/holdout/.
    sgf_dir: Where to write the sgfs"="sgf/.
    params: Hyper params of the model.
  """
  print('Playing a game with model %s' % model_name)
  model_path = os.path.join(models_dir, model_name)
  output_dir = os.path.join(selfplay_dir, model_name)
  holdout_dir = os.path.join(holdout_dir, model_name)
  clean_sgf = os.path.join(sgf_dir, model_name, 'clean')
  full_sgf = os.path.join(sgf_dir, model_name, 'full')

  _ensure_dir_exists(output_dir)
  _ensure_dir_exists(holdout_dir)
  _ensure_dir_exists(clean_sgf)
  _ensure_dir_exists(full_sgf)

  with utils.logged_timer('Loading weights from %s ... ' % model_path):
    network = dual_net.DualNetwork(model_path, params)

  with utils.logged_timer('Playing game'):
    player = selfplay_mcts.play(
        network, params.readouts, params.resign_threshold, params.verbose)

  output_name = '{}-{}'.format(int(time.time()), socket.gethostname())
  game_data = player.extract_data()
  with tf.gfile.GFile(
      os.path.join(clean_sgf, '{}.sgf'.format(output_name)), 'w') as f:
    f.write(player.to_sgf(use_comments=False))
  with tf.gfile.GFile(
      os.path.join(full_sgf, '{}.sgf'.format(output_name)), 'w') as f:
    f.write(player.to_sgf())

  tf_examples = preprocessing.make_dataset_from_selfplay(game_data)

  # Hold out 5% of games for evaluation.
  if random.random() < params.holdout_pct:
    fname = os.path.join(holdout_dir, '{}.tfrecord.zz'.format(output_name))
  else:
    fname = os.path.join(output_dir, '{}.tfrecord.zz'.format(output_name))

  preprocessing.write_tf_examples(fname, tf_examples)


def gather(selfplay_dir, training_chunk_dir, params):
  """Gather selfplay data into large training chunk.

  Args:
    selfplay_dir: Where to look for games'='data/selfplay/.
    training_chunk_dir: where to put collected games'='data/training_chunks/.
    params: Hyper params of the model.
  """
  # Check the selfplay data from the most recent 50 models.
  _ensure_dir_exists(training_chunk_dir)
  models = [model_dir.strip('/')
            for model_dir in sorted(
                tf.gfile.ListDirectory(selfplay_dir)
                )[-params.gather_generation:]]
  with utils.logged_timer('Finding existing tfrecords...'):
    model_gamedata = {
        model: tf.gfile.Glob(
            os.path.join(selfplay_dir, model, '*.tfrecord.zz'))
        for model in models
    }
  print('Found %d models' % len(models))
  for model_name, record_files in sorted(model_gamedata.items()):
    print('    %s: %s files' % (model_name, len(record_files)))

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
    print('Gathering files from %s:' % model_name)
    for i, example_batch in enumerate(tqdm(preprocessing.shuffle_tf_examples(
        params.examples_per_chunk, record_files))):
      output_record = os.path.join(
          training_chunk_dir, '{}-{}.tfrecord.zz'.format(model_name, str(i)))
      preprocessing.write_tf_examples(
          output_record, example_batch, serialize=False)
    already_processed.update(record_files)

  print('Processed %s new files' %
        (len(already_processed) - num_already_processed))
  with tf.gfile.GFile(meta_file, 'w') as f:
    f.write('\n'.join(sorted(already_processed)))


def train(models_dir, estimator_working_dir, training_chunk_dir, params):
  """Train the latest model from gathered data.

  Args:
    models_dir: Where to export the completed generation.
    estimator_working_dir: tf.estimator working directory.
    training_chunk_dir: Directory where gathered training chunks are.
    params: Hyper params of the model.
  """
  model_num, model_name = model_utils.get_latest_model(models_dir)
  print('Initializing from model %s' % model_name)

  new_model_name = model_utils.generate(model_num + 1)
  print('New model will be %s' % new_model_name)
  save_file = os.path.join(models_dir, new_model_name)

  try:
    tf_records = sorted(
        tf.gfile.Glob(os.path.join(training_chunk_dir, '*.tfrecord.zz')))
    tf_records = tf_records[
        -1 * (params.train_window_size // params.examples_per_chunk):]

    print('Training from: %s to %s' % (tf_records[0], tf_records[-1]))

    with utils.logged_timer('Training'):
      dual_net.train(estimator_working_dir, tf_records, model_num + 1, params)
      dual_net.export_model(estimator_working_dir, save_file)
  except tf.errors.UnknownError:
    print('Got an error training, muddling on...')
    tf.logging.error('Train error')


def validate(models_dir, holdout_dir, estimator_working_dir, params):
  """Validate the latest model on the holdout dataset.

  Args:
    models_dir: Directories where the completed generations/models are.
    holdout_dir: Directories where holdout data are.
    estimator_working_dir: tf.estimator working directory.
    params: Hyperparameters.
  """
  model_num, model_name = model_utils.get_latest_model(models_dir)

  # Get the holdout game data
  nums_names = model_utils.get_models(models_dir)
  models = [num_name for num_name in nums_names if num_name[0] < model_num]

  holdout_dirs = [os.path.join(holdout_dir, pair[1])
                  for pair in models[-params.holdout_generation:]]
  checkpoint_name = os.path.join(models_dir, model_name)

  tf_records = []
  with utils.logged_timer('Building lists of holdout files'):
    for record_dir in holdout_dirs:
      if os.path.exists(record_dir):  # make sure holdout dir exists
        tf_records.extend(tf.gfile.Glob(os.path.join(record_dir, '*.zz')))

  print('The length of tf_records is %d.' % len(tf_records))
  with utils.logged_timer('Validating from %s to %s' % (
      os.path.basename(tf_records[0]), os.path.basename(tf_records[-1]))):
    dual_net.validate(estimator_working_dir, tf_records,
                      checkpoint_name, params)


def evaluate(models_dir, black_model_name, white_model_name,
             evaluate_dir, params):
  """Evaluate on two models.

  Args:
    models_dir: Directories where the completed generations/models are.
    black_model_name: The name of the model playing black
    white_model_name: The name of the model playing white
    evaluate_dir: Where to write the evaluation results'='sgf/evaluate/
    params: Hyperparameters.

  Returns:
    The model name of the winner.
  """

  black_model = os.path.join(models_dir, black_model_name)
  white_model = os.path.join(models_dir, white_model_name)

  print('Evaluate models between %s and %s' % (
      black_model_name, white_model_name))

  _ensure_dir_exists(evaluate_dir)

  with utils.logged_timer('Loading weights'):
    black_net = dual_net.DualNetwork(black_model, params)
    white_net = dual_net.DualNetwork(white_model, params)

  with utils.logged_timer('%d games' % params.eval_games):
    winner = evaluation.play_match(
        black_net, white_net, params.eval_games, params.eval_readouts,
        evaluate_dir, params.eval_verbose)

  return black_model_name if winner == 'B' else white_model_name


def main(_):
  """Run the reinforcement learning loop."""
  tf.logging.set_verbosity(tf.logging.INFO)

  params = model_utils.MiniGoModelParams()

  k = utils.round_power_of_two(FLAGS.go_size ** 2 / 3)
  params.num_filters = k  # Number of filters in the convolution layer
  params.fc_width = 2 * k  # Width of each fully connected layer
  params.num_shared_layers = FLAGS.go_size  # Number of shared trunk layers
  params.go_size = FLAGS.go_size  # Board size

  # set the shuffle buffer size smaller
  preprocessing.SHUFFLE_BUFFER_SIZE = 200000

  # How many positions can fit on a graphics card. 256 for 9s, 16 or 32 for 19s.
  if FLAGS.go_size == 9:
    params.batch_size = 256
  else:
    params.batch_size = 32

  # A dummy model for debug/testing purpose with fewer games and iterations
  if FLAGS.debug:
    params.num_filters = 8  # Number of filters in the convolution layer
    params.fc_width = 16  # Width of each fully connected layer
    params.num_shared_layers = 1  # Number of shared trunk layers
    params.batch_size = 16
    params.examples_per_generation = 64
    params.max_games_per_generation = 10
    params.max_iters_per_pipeline = 2
    preprocessing.SHUFFLE_BUFFER_SIZE = 1000

  # Set directories for models and datasets
  if os.path.isdir(FLAGS.base_dir):  # if it already exists, delete it.
    shutil.rmtree(FLAGS.base_dir, ignore_errors=True)
  models_dir = os.path.join(FLAGS.base_dir, 'models')
  estimator_working_dir = os.path.join(FLAGS.base_dir, 'estimator_working_dir/')
  selfplay_dir = os.path.join(FLAGS.base_dir, 'data/selfplay/')
  holdout_dir = os.path.join(FLAGS.base_dir, 'data/holdout/')
  training_chunk_dir = os.path.join(FLAGS.base_dir, 'data/training_chunks/')
  sgf_dir = os.path.join(FLAGS.base_dir, 'sgf/')
  evaluate_dir = os.path.join(FLAGS.base_dir, 'sgf/evaluate/')

  print('Creating random initial weights...')
  bootstrap(estimator_working_dir, models_dir, params)

  # Start from the bootstrap model to play games.
  print('Playing some games...')
  model_name = '000000-bootstrap'
  for rl_iter in range(params.max_iters_per_pipeline):
    print('RL_iteration: %d' % rl_iter)

    # Self-play to generate at least params.max_games_per_generation games
    while True:
      params.holdout_pct = 0.05
      selfplay(model_name, models_dir, selfplay_dir, holdout_dir, sgf_dir,
               params)
      if FLAGS.holdout:
        params.holdout_pct = 1
        selfplay(model_name, models_dir, selfplay_dir, holdout_dir, sgf_dir,
                 params)
      games = tf.gfile.Glob(os.path.join(selfplay_dir, model_name, '*.zz'))
      if len(games) >= params.max_games_per_generation:
        break

    print('Gathering game output...')
    gather(selfplay_dir, training_chunk_dir, params)

    print('Training on gathered game data...')
    train(models_dir, estimator_working_dir, training_chunk_dir, params)

    if FLAGS.holdout:
      print('Validating on the holdout game data...')
      validate(models_dir, holdout_dir, estimator_working_dir, params)

    _, latest_model_name = model_utils.get_latest_model(models_dir)
    if FLAGS.eval:  # Perform evaluation if needed
      print('Evaluating the latest model...')
      model_name = evaluate(
          models_dir, model_name, latest_model_name, evaluate_dir, params)
      print('Winner: %s!' % model_name)
    else:
      model_name = latest_model_name


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--base_dir',
      type=str,
      default='/tmp/minigo/',
      metavar='BD',
      help='Base directory for the MiniGo models and datasets.')
  parser.add_argument(
      '--go_size',
      type=int,
      default=9,
      metavar='N',
      choices=[9, 19],
      help='Go board size. The default size is 9.')
  parser.add_argument(
      '--eval',
      type=bool,
      default=False,
      metavar='EVAL',
      help='A boolean to specify evaluation in the RL pipeline.')
  parser.add_argument(
      '--debug',
      type=bool,
      default=False,
      metavar='DEBUG',
      help='A boolean to indicate debug mode for testing purpose.')
  parser.add_argument(
      '--holdout',
      type=bool,
      default=False,
      metavar='HOLDOUT',
      help='A boolean to explicitly generate holdout data for validation.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
