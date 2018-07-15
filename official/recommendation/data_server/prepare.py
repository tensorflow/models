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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import multiprocessing
import os
import pickle
import timeit

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.datasets import movielens
from official.utils.flags import core as flags_core


TEST_POSITIVES = "test_positives"

_CACHE_SUBDIR = "ncf_recommendation_cache"
_TRAIN_SHARD_SUBDIR = "training_shards"
_APPROX_PTS_PER_SHARD = 32000

# In both datasets, each user has at least 20 ratings.
_MIN_NUM_RATINGS = 20

# The number of negative examples attached with a positive example
# in training dataset.
_NUMBER_NEGATIVES = 999


class NCFDataset(object):
  def __init__(self, cache_dir, test_data, num_users, num_items,
               num_data_readers, train_data=None):
    self.cache_dir = cache_dir
    self.train_shard_dir = os.path.join(cache_dir, _TRAIN_SHARD_SUBDIR)
    self.num_data_readers = num_data_readers
    self.test_data = test_data
    true_ind = np.argwhere(test_data[1])[:, 0]
    assert true_ind.shape[0] == num_users
    self.eval_true_items = {
      test_data[0][movielens.USER_COLUMN][i]:
        test_data[0][movielens.ITEM_COLUMN][i] for i in true_ind
    }
    self.eval_all_items = {}
    stride = _NUMBER_NEGATIVES + 1
    for i in range(num_users):
      user = test_data[0][movielens.USER_COLUMN][i * stride]
      items = test_data[0][movielens.ITEM_COLUMN][i * stride: (i + 1) * stride]
      self.eval_all_items[user] = items.tolist()
      assert len(self.eval_all_items[user]) == len(self.eval_all_items[user])

    self.num_users = num_users
    self.num_items = num_items

    # Used for testing the data pipeline. The actual training pipeline uses the
    # shards found in `self.train_shard_dir `
    self.train_data = train_data  # type: dict


def _filter_index_sort(raw_rating_path):
  df = pd.read_csv(raw_rating_path)

  # Get the info of users who have more than 20 ratings on items
  grouped = df.groupby(movielens.USER_COLUMN)
  df = grouped.filter(lambda x: len(x) >= _MIN_NUM_RATINGS)  # type: pd.DataFrame

  original_users = df[movielens.USER_COLUMN].unique()
  original_items = df[movielens.ITEM_COLUMN].unique()

  # Map the ids of user and item to 0 based index for following processing
  tf.logging.info("Generating user_map and item_map...")
  user_map = {user: index for index, user in enumerate(original_users)}
  item_map = {item: index for index, item in enumerate(original_items)}

  df[movielens.USER_COLUMN] = df[movielens.USER_COLUMN].apply(
      lambda user: user_map[user])
  df[movielens.ITEM_COLUMN] = df[movielens.ITEM_COLUMN].apply(
      lambda item: item_map[item])

  num_users = len(original_users)
  num_items = len(original_items)

  assert num_users <= np.iinfo(np.int32).max
  assert num_items <= np.iinfo(np.uint16).max
  assert df[movielens.USER_COLUMN].max() == num_users - 1
  assert df[movielens.ITEM_COLUMN].max() == num_items - 1

  # This sort is used to shard the dataframe by user, and later to select
  # the last item for a user to be used in validation.
  tf.logging.info("Sorting by user, timestamp...")
  df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                 inplace=True)

  return df, num_users, num_items


def construct_false_negatives(num_items, positive_set, n, replacement=True):
  if not isinstance(positive_set, set):
    positive_set = set(positive_set)

  p = 1 - len(positive_set) /  num_items
  n_attempt = int(n * (1 / p) * 1.2)  # factor of 1.2 for safety

  if replacement:
    negatives = []
  else:
    negatives = set()

  while len(negatives) < n:
    negative_candidates = np.random.randint(
        low=0, high=num_items, size=(n_attempt,))
    if replacement:
      negatives.extend(
          [i for i in negative_candidates if i not in positive_set]
      )
    else:
      negatives |= (set(negative_candidates) - positive_set)

  if not replacement:
    negatives = list(negatives)
    np.random.shuffle(negatives)

  return negatives[:n]


def _train_eval_map_fn(shard, shard_id, cache_dir, num_items):
  users = shard[movielens.USER_COLUMN]
  items = shard[movielens.ITEM_COLUMN]
  delta = users[1:] - users[:-1]
  boundaries = ([0] + (np.argwhere(delta)[:, 0] + 1).tolist() +
                [users.shape[0]])

  train_blocks = []
  test_blocks = []
  test_positives = []
  for i in range(len(boundaries) - 1):
    # This is simply a vector of repeated values such that the shard could be
    # represented compactly with a tuple of tuples:
    #   ((user_id, items), (user_id, items), ...)
    # rather than:
    #   user_id_vector, item_id_vector
    # However the additional nested structure significantly increases the
    # serialization and deserialization cost such that it is not worthwhile.
    block_user = users[boundaries[i]:boundaries[i+1]]
    assert len(set(block_user)) == 1

    block_items = items[boundaries[i]:boundaries[i+1]]
    train_blocks.append((block_user[:-1], block_items[:-1]))

    test_negatives = construct_false_negatives(
        num_items=num_items, positive_set=set(block_items), n=_NUMBER_NEGATIVES,
        replacement=False)
    test_blocks.append((
      block_user[0] * np.ones((_NUMBER_NEGATIVES + 1,), dtype=np.int32),
      np.array([block_items[-1]] + test_negatives, dtype=np.uint16)
    ))
    test_positives.append((block_user[0], block_items[-1]))

  train_users = np.concatenate([i[0] for i in train_blocks])
  train_items = np.concatenate([i[1] for i in train_blocks])

  train_shard_fname = "train_positive_shard_{}.pickle".format(
      str(shard_id).zfill(5))
  train_shard_fpath = os.path.join(
      cache_dir, _TRAIN_SHARD_SUBDIR, train_shard_fname)

  with tf.gfile.Open(train_shard_fpath, "wb") as f:
    pickle.dump({
      movielens.USER_COLUMN: train_users,
      movielens.ITEM_COLUMN: train_items,
      TEST_POSITIVES: test_positives,  # can be excluded from false negative
                                       # generation.
    }, f)


  test_users = np.concatenate([i[0] for i in test_blocks])
  test_items = np.concatenate([i[1] for i in test_blocks])
  assert test_users.shape == test_items.shape
  assert test_items.shape[0] % (_NUMBER_NEGATIVES + 1) == 0

  return {
    movielens.USER_COLUMN: test_users,
    movielens.ITEM_COLUMN: test_items,
  }


def generate_train_eval_data(df, approx_num_shards, cache_dir, num_items):
  num_rows = len(df)
  approximate_partitions = np.linspace(0, num_rows, approx_num_shards + 1).astype("int")
  start_ind, end_ind = 0, 0
  shards = []

  for i in range(1, approx_num_shards + 1):
    end_ind = approximate_partitions[i]
    while (end_ind < num_rows and df[movielens.USER_COLUMN][end_ind - 1] ==
           df[movielens.USER_COLUMN][end_ind]):
      end_ind += 1

    if not end_ind > start_ind:
      continue  # imbalance from prior shard.

    df_shard = df[start_ind:end_ind]

    shards.append({
      movielens.USER_COLUMN: df_shard[movielens.USER_COLUMN].values.astype(np.int32),
      movielens.ITEM_COLUMN: df_shard[movielens.ITEM_COLUMN].values.astype(np.uint16),
    })

    start_ind = end_ind
  assert end_ind == num_rows
  approx_num_shards = len(shards)

  tf.logging.info("Splitting train and test data and generating {} test "
                  "negatives per user...".format(_NUMBER_NEGATIVES))
  tf.gfile.MakeDirs(os.path.join(cache_dir, _TRAIN_SHARD_SUBDIR))
  map_args = [(shards[i], i, cache_dir, num_items) for i in range(approx_num_shards)]
  ctx = multiprocessing.get_context("spawn")
  with contextlib.closing(ctx.Pool(multiprocessing.cpu_count())) as pool:
    test_shards = pool.starmap(_train_eval_map_fn, map_args)

  tf.logging.info("Merging test shards...")
  test_users = np.concatenate([i[movielens.USER_COLUMN] for i in test_shards])
  test_items = np.concatenate([i[movielens.ITEM_COLUMN] for i in test_shards])

  assert test_users.shape == test_items.shape
  assert test_items.shape[0] % (_NUMBER_NEGATIVES + 1) == 0

  test_labels = np.zeros(shape=test_users.shape)
  test_labels[0::(_NUMBER_NEGATIVES + 1)] = 1

  return ({
    movielens.USER_COLUMN: test_users,
    movielens.ITEM_COLUMN: test_items,
  }, test_labels)


def construct_cache(dataset, data_dir, num_data_readers, num_neg, debug):
  pts_per_epoch = movielens.NUM_RATINGS[dataset] * (1 + num_neg)
  num_data_readers = num_data_readers or int(multiprocessing.cpu_count() / 2) or 1
  approx_num_shards = int(pts_per_epoch // _APPROX_PTS_PER_SHARD) or 1

  st = timeit.default_timer()
  cache_dir = os.path.join(data_dir, _CACHE_SUBDIR, dataset)
  if tf.gfile.Exists(cache_dir):
    tf.gfile.DeleteRecursively(cache_dir)
  tf.gfile.MakeDirs(cache_dir)

  raw_rating_path = os.path.join(data_dir, dataset, movielens.RATINGS_FILE)
  df, num_users, num_items = _filter_index_sort(raw_rating_path)

  test_data = generate_train_eval_data(
      df=df, approx_num_shards=approx_num_shards, cache_dir=cache_dir,
      num_items=num_items)
  del approx_num_shards  # value may have changed.

  train_data = None
  if debug:
    users = df[movielens.USER_COLUMN].values
    items = df[movielens.ITEM_COLUMN].values
    train_ind = np.argwhere(np.equal(users[:-1], users[1:]))[:, 0]
    train_data = {
      movielens.USER_COLUMN: users[train_ind],
      movielens.ITEM_COLUMN: items[train_ind],
    }

  ncf_dataset = NCFDataset(cache_dir=cache_dir, test_data=test_data,
                           num_items=num_items, num_users=num_users,
                           num_data_readers=num_data_readers,
                           train_data=train_data)
  run_time = timeit.default_timer() - st
  tf.logging.info("Cache construction complete. Time: {:.1f} sec."
                  .format(run_time))
  return ncf_dataset


def run(dataset, data_dir, num_data_readers=None, num_neg=4, debug=False):
  movielens.download(dataset=dataset, data_dir=data_dir)
  return construct_cache(dataset=dataset, data_dir=data_dir,
                         num_data_readers=num_data_readers, num_neg=num_neg,
                         debug=debug)


def main(_):
  run(dataset=flags.FLAGS.dataset, data_dir=flags.FLAGS.data_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  movielens.define_data_download_flags()
  flags.adopt_module_key_flags(movielens)
  flags_core.set_defaults(
      dataset="ml-20m",
      data_dir="/tmp/movielens_test"
  )
  absl_app.run(main)


