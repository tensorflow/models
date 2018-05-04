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

The NeuMF model ensembles both MF and MLP models under the NCF framework. Check
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

import tensorflow as tf  # pylint: disable=g-bad-import-order

import data_download
import ncf_dataset
import neumf_model
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin


def evaluate_model(est_model, user_input, item_input, gt_items, top_k):
  """Model evaluation with HR and NDCG metrics.

  The evaluation protocol is to rank the test interacted item among the randomly
  chosen 100 items that are not interacted by the user. The performance of the
  ranked list is judged by Hit Ratio (HR) and Normalized Discounted Cumulative
  Gain (NDCG).

  For evaluation, the ranked list is truncated at 10 for both metrics. As such,
  the HR intuitively measures whether the test item is present on the top-10
  list, and the NDCG accounts for the position of the hit by assigning higher
  scores to hits at top ranks. Both metrics are calculated for each test user,
  and the average scores are reported.

  Args:
    est_model: The Estimator.
    user_input: The user input for evaluation.
    item_input: The item input for evaluation.
    gt_items: The test item for HR and NDCG calculation.
    top_k: The top-k list. It is 10 by default.

  Returns:
    hits: HR score.
    ndcgs: NDCG score.
  """
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'user_input': user_input, 'item_input': item_input},
      batch_size=100,
      num_epochs=1,
      shuffle=False
  )
  # Get predictions
  predictions = est_model.predict(input_fn=predict_input_fn)
  all_predicted_scores = [p['prediction'] for p in predictions]

  # Calculate HR score
  def _get_hr(ranklist, gt_item):
    for item in ranklist:
      if item == gt_item:
        return 1
    return 0

  # Calculate NDCG score
  def _get_ndcg(ranklist, gt_item):
    for i in range(len(ranklist)):
      item = ranklist[i]
      if item == gt_item:
        return math.log(2) / math.log(i+2)
    return 0

  hits, ndcgs = [], []
  num_users = len(gt_items)
  step = len(user_input) // num_users  # Step for each user
  # Evaluation on each user
  for idx in range(num_users):
    start = idx * step
    end = (idx + 1) * step

    items = item_input[start:end]
    predicted_scores = all_predicted_scores[start:end]

    map_item_score = {}
    for i in range(len(items)):
      item = items[i][0]
      score = predicted_scores[i][0]
      map_item_score[item] = score

    # Evaluate top rank list with HR and NDCG
    ranklist = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)
    gt_item = gt_items[idx]
    hr = _get_hr(ranklist, gt_item)
    ndcg = _get_ndcg(ranklist, gt_item)
    hits.append(hr)
    ndcgs.append(ndcg)

  return hits, ndcgs


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  print('NeuMF arguments: {}'.format(FLAGS))

  # Load data
  t1 = time.time()
  train = ncf_dataset.load_train_matrix(os.path.join(
      FLAGS.data_dir, FLAGS.dataset + '-' + data_download.TRAIN_RATINGS_FILENAME
  ))
  test_ratings = ncf_dataset.load_test_ratings(os.path.join(
      FLAGS.data_dir, FLAGS.dataset + '-' + data_download.TEST_RATINGS_FILENAME
  ))
  test_negatives = ncf_dataset.load_test_negs(os.path.join(
      FLAGS.data_dir, FLAGS.dataset + '-' + data_download.TEST_NEG_FILENAME
  ))
  num_users, num_items = train.shape
  print('Load data done [{:.1f} s]. #user={}, #item={}, #train={}, #test={}'
        .format(time.time()-t1, num_users, num_items, train.nnz,
                len(test_ratings)))

  # Create NeuMF model from tf.keras Model
  neumf = neumf_model.NeuMF(num_users, num_items, FLAGS.num_factors,
                            ast.literal_eval(FLAGS.layers), FLAGS.learning_rate)
  model = neumf(FLAGS.multi_gpu, FLAGS.batch_size)

  # Convert tf.keras model to Estimator
  est_model = tf.keras.estimator.model_to_estimator(
      keras_model=model, model_dir=FLAGS.model_dir)

  # Initial performance
  top_k = 10  # Top-k list for evaluation
  eval_user_input, eval_item_input, eval_gt_items = ncf_dataset.get_eval_input(
      test_ratings, test_negatives)
  hits, ndcgs = evaluate_model(
      est_model, eval_user_input, eval_item_input, eval_gt_items, top_k)
  hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
  print('Init: HR = {:.4f}, NDCG = {:.4f}'.format(hr, ndcg))
  best_hr, best_ndcg, best_iter = hr, ndcg, -1

  # Train multiple epochs followed by evaluation
  for epoch in xrange(FLAGS.epochs):
    t1 = time.time()
    # Get training instances
    train_user_input, train_item_input, train_labels = (
        ncf_dataset.get_train_input(train, FLAGS.num_neg))
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'user_input': train_user_input, 'item_input': train_item_input},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=True)
    est_model.train(
        input_fn=train_input_fn,
        hooks=[tf.train.ProfilerHook(save_steps=1000)])
    t2 = time.time()

    # Evaluation of the model for each epoch
    print('Model Evaluation at epoch: {}'.format(epoch))
    hits, ndcgs = evaluate_model(
        est_model, eval_user_input, eval_item_input, eval_gt_items, top_k)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Iteration {} [{:.1f} s]: HR = {:.4f}, NDCG = {:.4f}, [{:.1f} s]'
          .format(epoch, t2 - t1, hr, ndcg, time.time() - t2))

    # Update the best hr
    if hr > best_hr:
      best_hr, best_ndcg, best_iter = hr, ndcg, epoch
    # We hit the threshold of hr
    if FLAGS.hr_threshold is not None:
      if hr >= FLAGS.hr_threshold:
        print('Hit threshold of {}'.format(FLAGS.hr_threshold))
        return 0

  print('End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}.'
        .format(best_iter, best_hr, best_ndcg))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--model_dir', nargs='?', default='/tmp/ncf/',
      help='Model directory.')
  parser.add_argument(
      '--data_dir', nargs='?', default='/tmp/ml_data/',
      help='Input data directory. Should be the same as downloaded data dir.')
  parser.add_argument(
      '--dataset', nargs='?', default='ml-1m', choices=['ml-1m', 'ml-20m'],
      help='Choose a dataset.')
  parser.add_argument(
      '--epochs', type=int, default=20,
      help='Number of epochs.')
  parser.add_argument(
      '--batch_size', type=int, default=256,
      help='Batch size.')
  parser.add_argument(
      '--num_factors', type=int, default=8,
      help='Embedding size of MF model.')
  parser.add_argument(
      '--layers', nargs='?', default='[64,32,16,8]',
      help='Size of hidden layers for MLP.')
  parser.add_argument(
      '--num_neg', type=int, default=4,
      help='Number of negative instances to pair with a positive instance.')
  parser.add_argument(
      '--learning_rate', type=float, default=0.001,
      help='Learning rate.')
  parser.add_argument(
      '--hr_threshold', type=float, default=0.68, choices=[0.68, 0.95],
      help='Stop training early at HR threshold.')
  parser.add_argument(
      '--multi_gpu', action='store_true',
      help='A boolean to set multiple gpus.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
