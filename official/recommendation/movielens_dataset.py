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
"""Prepare MovieLens dataset for NCF recommendation model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import tempfile
import time

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
import numpy as np
import pandas as pd
from six.moves import xrange
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.datasets import movielens
from official.utils.data import file_io
from official.utils.flags import core as flags_core


_BUFFER_SUBDIR = "ncf_recommendation_buffer"
_TRAIN_RATINGS_FILENAME = 'train-ratings.csv'
_TEST_RATINGS_FILENAME = 'test-ratings.csv'
_TEST_NEG_FILENAME = 'test-negative.csv'

# The number of negative examples attached with a positive example
# in training dataset. It is set as 100 in the paper.
_NUMBER_NEGATIVES = 100

# In both datasets, each user has at least 20 ratings.
_MIN_NUM_RATINGS = 20

# The buffer size for shuffling train dataset.
_SHUFFLE_BUFFER_SIZE = 1024

_FEATURE_MAP = {
    movielens.USER_COLUMN: tf.FixedLenFeature([1], dtype=tf.int64),
    movielens.ITEM_COLUMN: tf.FixedLenFeature([1], dtype=tf.int64),
    movielens.RATING_COLUMN: tf.FixedLenFeature([1], dtype=tf.int64),
}

_FEATURE_MAP_EVAL = {
    movielens.USER_COLUMN: tf.FixedLenFeature([1], dtype=tf.int64),
    movielens.ITEM_COLUMN: tf.FixedLenFeature([1], dtype=tf.int64),
}

_COLUMNS = [movielens.USER_COLUMN, movielens.ITEM_COLUMN,
            movielens.RATING_COLUMN]
_EVAL_COLUMNS = _COLUMNS[:2]


_EVAL_BUFFER_SIZE = {
    movielens.ML_1M: 34130690,
    movielens.ML_20M: 800961490,
}


def generate_train_eval_data(df, original_users, original_items):
  """Generate the dataset for model training and evaluation.

  Given all user and item interaction information, for each user, first sort
  the interactions based on timestamp. Then the latest one is taken out as
  Test ratings (leave-one-out evaluation) and the remaining data for training.
  The Test negatives are randomly sampled from all non-interacted items, and the
  number of Test negatives is 100 by default (defined as _NUMBER_NEGATIVES).

  Args:
    df: The DataFrame of ratings data.
    original_users: A list of the original unique user ids in the dataset.
    original_items: A list of the original unique item ids in the dataset.

  Returns:
    all_ratings: A list of the [user_id, item_id] with interactions.
    test_ratings: A list of [user_id, item_id], and each line is the latest
      user_item interaction for the user.
    test_negs: A list of item ids with shape [num_users, 100].
      Each line consists of 100 item ids for the user with no interactions.
  """
  # Need to sort before popping to get last item
  tf.logging.info("Sorting user_item_map by timestamp...")
  df.sort_values(by=movielens.TIMESTAMP_COLUMN, inplace=True)
  all_ratings = set(zip(df[movielens.USER_COLUMN], df[movielens.ITEM_COLUMN]))
  user_to_items = collections.defaultdict(list)

  # Generate user_item rating matrix for training
  t1 = time.time()
  row_count = 0
  for row in df.itertuples():
    user_to_items[getattr(row, movielens.USER_COLUMN)].append(
        getattr(row, movielens.ITEM_COLUMN))
    row_count += 1
    if row_count % 50000 == 0:
      tf.logging.info("Processing user_to_items row: {}".format(row_count))
  tf.logging.info(
      "Process {} rows in [{:.1f}]s".format(row_count, time.time() - t1))

  # Generate test ratings and test negatives
  t2 = time.time()
  test_ratings = []
  test_negs = []
  # Generate the 0-based index for each item, and put it into a set
  all_items = set(range(len(original_items)))
  for user in range(len(original_users)):
    test_item = user_to_items[user].pop()  # Get the latest item id

    all_ratings.remove((user, test_item))  # Remove the test item
    all_negs = all_items.difference(user_to_items[user])
    all_negs = sorted(list(all_negs))  # determinism

    test_ratings.append((user, test_item))
    test_negs.append(list(np.random.choice(all_negs, _NUMBER_NEGATIVES)))

    if user % 1000 == 0:
      tf.logging.info("Processing user: {}".format(user))

  tf.logging.info("Process {} users in {:.1f}s".format(
      len(original_users), time.time() - t2))

  all_ratings = list(all_ratings)  # convert set to list
  return all_ratings, test_ratings, test_negs


def _csv_buffer_paths(data_dir, dataset):
  buffer_dir = os.path.join(data_dir, _BUFFER_SUBDIR)
  return (
      os.path.join(buffer_dir, dataset + "-" + _TRAIN_RATINGS_FILENAME),
      os.path.join(buffer_dir, dataset + "-" + _TEST_RATINGS_FILENAME),
      os.path.join(buffer_dir, dataset + "-" + _TEST_NEG_FILENAME)
  )


def construct_train_eval_csv(data_dir, dataset):
  """Parse the raw data to csv file to be used in model training and evaluation.

  ml-1m dataset is small in size (~25M), while ml-20m is large (~500M). It may
  take several minutes to process ml-20m dataset.

  Args:
    data_dir: A string, the root directory of the movielens dataset.
    dataset: A string, the dataset name to be processed.
  """
  assert dataset in movielens.DATASETS

  if all([tf.gfile.Exists(i) for i in _csv_buffer_paths(data_dir, dataset)]):
    return

  # Use random seed as parameter
  np.random.seed(0)

  df = movielens.ratings_csv_to_dataframe(data_dir=data_dir, dataset=dataset)

  # Get the info of users who have more than 20 ratings on items
  grouped = df.groupby(movielens.USER_COLUMN)
  df = grouped.filter(lambda x: len(x) >= _MIN_NUM_RATINGS)
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
  assert df[movielens.USER_COLUMN].max() == len(original_users) - 1
  assert df[movielens.ITEM_COLUMN].max() == len(original_items) - 1

  # Generate data for train and test
  all_ratings, test_ratings, test_negs = generate_train_eval_data(
      df, original_users, original_items)

  # Serialize to csv file. Each csv file contains three columns
  # (user_id, item_id, interaction)
  tf.gfile.MakeDirs(os.path.join(data_dir, _BUFFER_SUBDIR))
  train_ratings_file, test_ratings_file, test_negs_file = _csv_buffer_paths(
      data_dir, dataset)

  # As there are only two fields (user_id, item_id) in all_ratings and
  # test_ratings, we need to add a fake rating to make three columns
  df_train_ratings = pd.DataFrame(all_ratings)
  df_train_ratings["fake_rating"] = 1
  with tf.gfile.Open(train_ratings_file, "w") as f:
    df_train_ratings.to_csv(f, index=False, header=False, sep="\t")
  tf.logging.info("Train ratings is {}".format(train_ratings_file))

  df_test_ratings = pd.DataFrame(test_ratings)
  df_test_ratings["fake_rating"] = 1
  with tf.gfile.Open(test_ratings_file, "w") as f:
    df_test_ratings.to_csv(f, index=False, header=False, sep="\t")
  tf.logging.info("Test ratings is {}".format(test_ratings_file))

  df_test_negs = pd.DataFrame(test_negs)
  with tf.gfile.Open(test_negs_file, "w") as f:
    df_test_negs.to_csv(f, index=False, header=False, sep="\t")
  tf.logging.info("Test negatives is {}".format(test_negs_file))


class NCFDataSet(object):
  """A class containing data information for model training and evaluation."""

  def __init__(self, train_data, num_users, num_items, num_negatives,
               true_items, all_items, all_eval_data):
    """Initialize NCFDataset class.

    Args:
      train_data: A list containing the positive training instances.
      num_users: An integer, the number of users in training dataset.
      num_items: An integer, the number of items in training dataset.
      num_negatives: An integer, the number of negative instances for each user
        in train dataset.
      true_items: A list, the ground truth (positive) items of users for
        evaluation. Each entry is a latest positive instance for one user.
      all_items: A nested list, all items for evaluation, and each entry is the
        evaluation items for one user.
      all_eval_data: A numpy array of eval/test dataset.
    """
    self.train_data = train_data
    self.num_users = num_users
    self.num_items = num_items
    self.num_negatives = num_negatives
    self.eval_true_items = true_items
    self.eval_all_items = all_items
    self.all_eval_data = all_eval_data


def load_data(file_name):
  """Load data from a csv file which splits on tab key."""
  lines = tf.gfile.Open(file_name, "r").readlines()

  # Process the file line by line
  def _process_line(line):
    return [int(col) for col in line.split("\t")]

  data = [_process_line(line) for line in lines]
  return data


def data_preprocessing(data_dir, dataset, num_negatives):
  """Preprocess the train and test dataset.

  In data preprocessing, the training positive instances are loaded into memory
  for random negative instance generation in each training epoch. The test
  dataset are generated from test positive and negative instances.

  Args:
    data_dir: A string, the root directory of the movielens dataset.
    dataset: A string, the dataset name to be processed.
    num_negatives: An integer, the number of negative instances for each user
      in train dataset.

  Returns:
    ncf_dataset: A NCFDataset object containing information about training and
      evaluation/test dataset.
  """
  train_fname, test_fname, test_neg_fname = _csv_buffer_paths(
      data_dir, dataset)

  # Load training positive instances into memory for later train data generation
  train_data = load_data(train_fname)
  # Get total number of users in the dataset
  num_users = len(np.unique(np.array(train_data)[:, 0]))

  # Process test dataset to csv file
  test_ratings = load_data(test_fname)
  test_negatives = load_data(test_neg_fname)
  # Get the total number of items in both train dataset and test dataset (the
  # whole dataset)
  num_items = len(
      set(np.array(train_data)[:, 1]) | set(np.array(test_ratings)[:, 1]))

  # Generate test instances for each user
  true_items, all_items = [], []
  all_test_data = []
  for idx in range(num_users):
    items = test_negatives[idx]
    rating = test_ratings[idx]
    user = rating[0]  # User
    true_item = rating[1]  # Positive item as ground truth

    # All items with first 100 as negative and last one positive
    items.append(true_item)
    users = np.full(len(items), user, dtype=np.int32)

    users_items = list(zip(users, items))  # User-item list
    true_items.append(true_item)  # all ground truth items
    all_items.append(items)  # All items (including positive and negative items)
    all_test_data.extend(users_items)  # Generate test dataset

  # Create NCFDataset object
  ncf_dataset = NCFDataSet(
      train_data, num_users, num_items, num_negatives, true_items, all_items,
      np.asarray(all_test_data)
  )

  return ncf_dataset


def generate_train_dataset(train_data, num_items, num_negatives):
  """Generate train dataset for each epoch.

  Given positive training instances, randomly generate negative instances to
  form the training dataset.

  Args:
    train_data: A list of positive training instances.
    num_items: An integer, the number of items in positive training instances.
    num_negatives: An integer, the number of negative training instances
      following positive training instances. It is 4 by default.

  Returns:
    A numpy array of training dataset.
  """
  all_train_data = []
  # A set with user-item tuples
  train_data_set = set((u, i) for u, i, _ in train_data)
  for u, i, _ in train_data:
    # Positive instance
    all_train_data.append([u, i, 1])
    # Negative instances, randomly generated
    for _ in xrange(num_negatives):
      j = np.random.randint(num_items)
      while (u, j) in train_data_set:
        j = np.random.randint(num_items)
      all_train_data.append([u, j, 0])

  return np.asarray(all_train_data)


def _deserialize_train(examples_serialized):
  features = tf.parse_example(examples_serialized, _FEATURE_MAP)
  train_features = {
      movielens.USER_COLUMN: features[movielens.USER_COLUMN],
      movielens.ITEM_COLUMN: features[movielens.ITEM_COLUMN],
  }
  return train_features, features[movielens.RATING_COLUMN]


def _deserialize_eval(examples_serialized):
  features = tf.parse_example(examples_serialized, _FEATURE_MAP_EVAL)
  return features


def get_input_fn(training, batch_size, ncf_dataset, data_dir, dataset,
                 repeat=1):
  """Input function for model training and evaluation.

  The train input consists of 1 positive instance (user and item have
  interactions) followed by some number of negative instances in which the items
  are randomly chosen. The number of negative instances is "num_negatives" which
  is 4 by default. Note that for each epoch, we need to re-generate the negative
  instances. Together with positive instances, they form a new train dataset.

  Args:
    training: A boolean flag for training mode.
    batch_size: An integer, batch size for training and evaluation.
    ncf_dataset: An NCFDataSet object, which contains the information about
      training and test data.
    repeat: An integer, how many times to repeat the dataset.

  Returns:
    dataset: A tf.data.Dataset object containing examples loaded from the files.
  """
  # Generate random negative instances for training in each epoch
  if training:
    tf.logging.info("Generating training data.")
    train_data = generate_train_dataset(
        ncf_dataset.train_data, ncf_dataset.num_items,
        ncf_dataset.num_negatives)

    df = pd.DataFrame(data=train_data, columns=_COLUMNS)

    if data_dir.startswith("gs://"):
      buffer_dir = os.path.join(data_dir, _BUFFER_SUBDIR)
    else:
      buffer_dir = None

    buffer_path = file_io.write_to_temp_buffer(df, buffer_dir, _COLUMNS)
    map_fn = _deserialize_train

  else:
    df = pd.DataFrame(ncf_dataset.all_eval_data, columns=_EVAL_COLUMNS)
    buffer_path = os.path.join(
        data_dir, _BUFFER_SUBDIR, dataset + "_eval_buffer")

    file_io.write_to_buffer(
        dataframe=df, buffer_path=buffer_path, columns=_EVAL_COLUMNS,
        expected_size=_EVAL_BUFFER_SIZE[dataset])
    map_fn = _deserialize_eval


  def input_fn():  # pylint: disable=missing-docstring
    dataset = tf.data.TFRecordDataset(buffer_path)
    if training:
      dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_fn, num_parallel_calls=16)
    dataset = dataset.repeat(repeat)

    # Prefetch to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

  return input_fn


def main(_):
  movielens.download(dataset=flags.FLAGS.dataset, data_dir=flags.FLAGS.data_dir)
  construct_train_eval_csv(flags.FLAGS.data_dir, flags.FLAGS.dataset)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  movielens.define_data_download_flags()
  flags.adopt_module_key_flags(movielens)
  flags_core.set_defaults(dataset="ml-1m")
  absl_app.run(main)
