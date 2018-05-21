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
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import heapq
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

# pylint: disable=g-bad-import-order
from official.recommendation import constants
from official.recommendation import dataset
from official.recommendation import neumf_model

_TOP_K = 10  # Top-k list for evaluation
_EVAL_BATCH_SIZE = 100


def evaluate_model(estimator, batch_size, num_gpus, true_items, all_items,
                   num_parallel_calls):
  """Model evaluation with HR and NDCG metrics.

  The evaluation protocol is to rank the test interacted item (truth items)
  among the randomly chosen 100 items that are not interacted by the user.
  The performance of the ranked list is judged by Hit Ratio (HR) and Normalized
  Discounted Cumulative Gain (NDCG).

  For evaluation, the ranked list is truncated at 10 for both metrics. As such,
  the HR intuitively measures whether the test item is present on the top-10
  list, and the NDCG accounts for the position of the hit by assigning higher
  scores to hits at top ranks. Both metrics are calculated for each test user,
  and the average scores are reported.

  Args:
    estimator: The Estimator.
    batch_size: An integer, the batch size specified by user.
    num_gpus: An integer, the number of gpus specified by user.
    true_items: A list of test items (true items) for HR and NDCG calculation.
      Each item is for one user.
    all_items: A nested list. Each entry is the 101 items (1 ground truth item
      and 100 negative items) for one user.
    num_parallel_calls: An integer, number of cpu cores for parallel input
      processing in input_fn.

  Returns:
    hit: An integer, the average HR scores for all users.
    ndcg: An integer, the average NDCG scores for all users.
  """
  # Define prediction input function
  def pred_input_fn():
    return dataset.input_fn(
        False, per_device_batch_size(batch_size, num_gpus),
        num_parallel_calls=num_parallel_calls)

  # Get predictions
  predictions = estimator.predict(input_fn=pred_input_fn)
  all_predicted_scores = [p[constants.RATING] for p in predictions]

  # Calculate HR score
  def _get_hr(ranklist, true_item):
    return 1 if true_item in ranklist else 0

  # Calculate NDCG score
  def _get_ndcg(ranklist, true_item):
    if true_item in ranklist:
      return math.log(2) / math.log(ranklist.index(true_item) + 2)
    return 0

  hits, ndcgs = [], []
  num_users = len(true_items)
  # Reshape the predicted scores and each user takes one row
  predicted_scores_list = np.asarray(
      all_predicted_scores).reshape(num_users, -1)

  for i in range(num_users):
    items = all_items[i]
    predicted_scores = predicted_scores_list[i]
    # Map item and score for each user
    map_item_score = {}
    for j, item in enumerate(items):
      score = predicted_scores[j]
      map_item_score[item] = score

    # Evaluate top rank list with HR and NDCG
    ranklist = heapq.nlargest(_TOP_K, map_item_score, key=map_item_score.get)
    true_item = true_items[i]
    hr = _get_hr(ranklist, true_item)
    ndcg = _get_ndcg(ranklist, true_item)
    hits.append(hr)
    ndcgs.append(ndcg)

  # Get average HR and NDCG scores
  hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
  return hr, ndcg


def get_num_gpus(num_gpus):
  """Treat num_gpus=-1 as "use all"."""
  if num_gpus != -1:
    return num_gpus

  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def convert_keras_to_estimator(keras_model, num_gpus, model_dir):
  """Configure and convert keras model to Estimator.

  Args:
    keras_model: A Keras model object.
    num_gpus: An integer, the number of gpus.
    model_dir: A string, the directory to save and restore checkpoints.

  Returns:
    est_model: The converted Estimator.

  """
  # TODO(b/79866338): update GradientDescentOptimizer with AdamOptimizer
  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=FLAGS.learning_rate)
  keras_model.compile(optimizer=optimizer, loss="binary_crossentropy")

  if num_gpus == 0:
    distribution = tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    distribution = tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

  run_config = tf.estimator.RunConfig(train_distribute=distribution)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir=model_dir, config=run_config)

  return estimator


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.

  Returns:
    Batch size per device.

  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ("When running with multiple GPUs, batch size "
           "must be a multiple of the number of available GPUs. Found {} "
           "GPUs with a batch size of {}; try --batch_size={} instead."
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)


def main(_):
  # Data preprocessing
  # The file name of training and test dataset
  train_fname = os.path.join(
      FLAGS.data_dir, FLAGS.dataset + "-" + constants.TRAIN_RATINGS_FILENAME)
  test_fname = os.path.join(
      FLAGS.data_dir, FLAGS.dataset + "-" + constants.TEST_RATINGS_FILENAME)
  neg_fname = os.path.join(
      FLAGS.data_dir, FLAGS.dataset + "-" + constants.TEST_NEG_FILENAME)
  t1 = time.time()
  ncf_dataset = dataset.data_preprocessing(
      train_fname, test_fname, neg_fname, FLAGS.num_neg)
  tf.logging.info("Data preprocessing: {:.1f} s".format(time.time() - t1))

  # Create NeuMF model and convert it to Estimator
  tf.logging.info("Creating Estimator from Keras model...")
  keras_model = neumf_model.NeuMF(
      ncf_dataset.num_users, ncf_dataset.num_items, FLAGS.num_factors,
      ast.literal_eval(FLAGS.layers), FLAGS.batch_size, FLAGS.mf_regularization)
  num_gpus = get_num_gpus(FLAGS.num_gpus)
  estimator = convert_keras_to_estimator(keras_model, num_gpus, FLAGS.model_dir)

  # Training and evaluation cycle
  def train_input_fn():
    return dataset.input_fn(
        True, per_device_batch_size(FLAGS.batch_size, num_gpus),
        FLAGS.epochs_between_evals, ncf_dataset, FLAGS.num_parallel_calls)

  total_training_cycle = (FLAGS.train_epochs //
                          FLAGS.epochs_between_evals)
  for cycle_index in range(total_training_cycle):
    tf.logging.info("Starting a training cycle: {}/{}".format(
        cycle_index, total_training_cycle - 1))

    # Train the model
    train_cycle_begin = time.time()
    estimator.train(input_fn=train_input_fn,
                    hooks=[tf.train.ProfilerHook(save_steps=10000)])
    train_cycle_end = time.time()

    # Evaluate the model
    eval_cycle_begin = time.time()
    hr, ndcg = evaluate_model(
        estimator, FLAGS.batch_size, num_gpus, ncf_dataset.eval_true_items,
        ncf_dataset.eval_all_items, FLAGS.num_parallel_calls)
    eval_cycle_end = time.time()

    # Log the train time, evaluation time, and HR and NDCG results.
    tf.logging.info(
        "Iteration {} [{:.1f} s]: HR = {:.4f}, NDCG = {:.4f}, [{:.1f} s]"
        .format(cycle_index, train_cycle_end - train_cycle_begin, hr, ndcg,
                eval_cycle_end - eval_cycle_begin))

  # Remove temporary files
  os.remove(constants.TRAIN_DATA)
  os.remove(constants.TEST_DATA)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--model_dir", nargs="?", default="/tmp/ncf/",
      help="Model directory.")
  parser.add_argument(
      "--data_dir", nargs="?", default="/tmp/movielens-data/",
      help="Input data directory. Should be the same as downloaded data dir.")
  parser.add_argument(
      "--dataset", nargs="?", default="ml-1m", choices=["ml-1m", "ml-20m"],
      help="Choose a dataset.")
  parser.add_argument(
      "--train_epochs", type=int, default=20,
      help="Number of epochs.")
  parser.add_argument(
      "--batch_size", type=int, default=256,
      help="Batch size.")
  parser.add_argument(
      "--num_factors", type=int, default=8,
      help="Embedding size of MF model.")
  parser.add_argument(
      "--layers", nargs="?", default="[64,32,16,8]",
      help="Size of hidden layers for MLP.")
  parser.add_argument(
      "--mf_regularization", type=float, default=0,
      help="Regularization for MF embeddings.")
  parser.add_argument(
      "--num_neg", type=int, default=4,
      help="Number of negative instances to pair with a positive instance.")
  parser.add_argument(
      "--num_parallel_calls", type=int, default=8,
      help="Number of parallel calls.")
  parser.add_argument(
      "--epochs_between_evals", type=int, default=1,
      help="Number of epochs between model evaluation.")
  parser.add_argument(
      "--learning_rate", type=float, default=0.001,
      help="Learning rate.")
  parser.add_argument(
      "--num_gpus", type=int, default=1 if tf.test.is_gpu_available() else 0,
      help="How many GPUs to use with the DistributionStrategies API. The "
           "default is 1 if TensorFlow can detect a GPU, and 0 otherwise.")

  FLAGS, unparsed = parser.parse_known_args()
  with tf.Graph().as_default():
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
