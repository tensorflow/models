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
"""Download and extract the movielens dataset from grouplens website.

Download the dataset, and perform data-preprocessing to convert the raw dataset
into csv file to be used in model training and evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import os
import sys
import time
import urllib
import zipfile

import tensorflow as tf  # pylint: disable=g-bad-import-order

import numpy as np
import pandas as pd

USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'
NUMBER_NEGATIVES = 100

TRAIN_RATINGS_FILENAME = 'train-ratings.csv'
TEST_RATINGS_FILENAME = 'test-ratings.csv'
TEST_NEG_FILENAME = 'test-negative.csv'

DATA_URL = 'http://files.grouplens.org/datasets/movielens/'

RatingData = collections.namedtuple(
    'RatingData', ['items', 'users', 'ratings', 'min_date', 'max_date'])


def _describe_ratings(ratings):
  """Describe the rating dataset information.

  Args:
    ratings: A pandas DataFrame of the rating dataset.
  """
  info = RatingData(items=len(ratings['item_id'].unique()),
                    users=len(ratings['user_id'].unique()),
                    ratings=len(ratings),
                    min_date=ratings['timestamp'].min(),
                    max_date=ratings['timestamp'].max())
  print('{ratings} ratings on {items} items from {users} users'
        ' from {min_date} to {max_date}'
        .format(**(info._asdict())))


def process_movielens(ratings, sort=True):
  """Process the movielens dataset.

  Args:
    ratings: A pandas DataFrame of the rating dataset.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    ratings: The processed DataFrame.
  """
  ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
  if sort:
    ratings.sort_values(by='timestamp', inplace=True)
  _describe_ratings(ratings)
  return ratings


def load_ml_1m(file_name, sort=True):
  """Load the ml-1m dataset.

  Args:
    file_name: A string of the file name to be loaded.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    A processed pandas DataFrame of the rating dataset.
  """
  names = ['user_id', 'item_id', 'rating', 'timestamp']
  ratings = pd.read_csv(file_name, sep='::', names=names, engine='python')
  return process_movielens(ratings, sort=sort)


def load_ml_20m(file_name, sort=True):
  """Load the ml-20m dataset.

  Args:
    file_name: A string of the file name to be loaded.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    A processed pandas DataFrame of the rating dataset.
  """
  ratings = pd.read_csv(file_name)
  ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
  names = {'userId': 'user_id', 'movieId': 'item_id'}
  ratings.rename(columns=names, inplace=True)
  return process_movielens(ratings, sort=sort)


def load_file_to_df(file_name, sort=True):
  """Load rating dataset into DataFrame.

  Two data loading functions are defined to handle dataset ml-1m and ml-20m,
  as they are provided with different formats.

  Args:
    file_name: A string of the file name to be loaded.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    A pandas DataFrame of the rating dataset.
  """
  dataset_name = os.path.basename(file_name).split('.')[0]
  # ml-1m with extension .dat
  file_extension = '.dat'
  func = load_ml_1m
  if dataset_name == 'ml-20m':
    file_extension = '.csv'
    func = load_ml_20m
  ratings_file = os.path.join(file_name, 'ratings' + file_extension)
  return func(ratings_file, sort=sort)


def generate_train_eval_data(df, original_users, original_items):
  """Generate the dataset for model training and evaluation.

  Given all user and item interaction information, for each user, first to sort
  the interactions based on timestamp. Then the latest one is taken out as
  Test ratings (leave-one-out evaluation) and the remaining data for training.
  The Test negatives are randomly sampled from all non-interacted items, and the
  number of Test negatives is 100 (defined as NUMBER_NEGATIVES).

  Args:
    df: The DataFrame of ratings data.
    original_users: The original unique user ids in the dataset.
    original_items: The original unique item ids in the dataset.

  Returns:
    all_ratings: A set of the [user_id, item_id] with interactions.
    test_ratings: A list of [user_id, item_id], and each line is the latest
      user_item interaction for the user.
    test_negs: A list of item ids with shape [num_users, 100].
      Each line consists of 100 item ids for the user with no interactions.
  """
  # Need to sort before popping to get last item
  print('Sorting user_item_map by timestamp...')
  df.sort_values(by='timestamp', inplace=True)
  all_ratings = set(zip(df[USER_COLUMN], df[ITEM_COLUMN]))
  user_to_items = collections.defaultdict(list)

  # Generate user_item rating matrix for training
  t1 = time.time()
  row_count = 0
  for row in df.itertuples():
    user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))
    row_count += 1
    if row_count % 50000 == 0:
      print('Processing user_to_items row: {}'.format(row_count))
  print('Process {} rows in [{:.1f}]s'.format(row_count, time.time() - t1))

  # Generate test ratings and test negatives
  t2 = time.time()
  test_ratings = []
  test_negs = []
  all_items = set(range(len(original_items)))
  for user in range(len(original_users)):
    test_item = user_to_items[user].pop()  # Get the latest one

    all_ratings.remove((user, test_item))  # Remove the test item
    all_negs = all_items - set(user_to_items[user])
    all_negs = sorted(list(all_negs))  # determinism

    test_ratings.append((user, test_item))
    test_negs.append(list(np.random.choice(all_negs, NUMBER_NEGATIVES)))

    if user % 1000 == 0:
      print('Processing user: {}'.format(user))

  print('Process {} users in [{:.1f}]s'.format(
      len(original_users), time.time() - t2))

  return all_ratings, test_ratings, test_negs


def parse_file_to_csv(data_dir, dataset_name):
  """Parse the raw data to csv file to be used in model training and evaluation.

  ml-1m dataset is small in size (~25M), while ml-20m is large (~500M). It may
  take several minutes to process ml-20m dataset.

  Args:
    data_dir: The directory with the unzipped dataset.
    dataset_name: The dataset name to be processed.
  """

  # Use random seed as parameter
  np.random.seed(0)

  # Load the file as DataFrame
  file_path = os.path.join(data_dir, dataset_name)
  df = load_file_to_df(file_path, sort=False)

  # Get the info of users who have more than 20 ratings on items.
  grouped = df.groupby(USER_COLUMN)
  df = grouped.filter(lambda x: len(x) >= 20)
  original_users = df[USER_COLUMN].unique()
  original_items = df[ITEM_COLUMN].unique()

  print('Generate user_map and item_map...')
  user_map = {user: index for index, user in enumerate(original_users)}
  item_map = {item: index for index, item in enumerate(original_items)}

  df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
  df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])
  assert df[USER_COLUMN].max() == len(original_users) - 1
  assert df[ITEM_COLUMN].max() == len(original_items) - 1

  # Generate data for train and test
  all_ratings, test_ratings, test_negs = generate_train_eval_data(
      df, original_users, original_items)

  # serialize to csv file
  df_train_ratings = pd.DataFrame(list(all_ratings))
  df_train_ratings['fake_rating'] = 1
  df_train_ratings.to_csv(
      os.path.join(FLAGS.data_dir, dataset_name + '-' + TRAIN_RATINGS_FILENAME),
      index=False, header=False, sep='\t')

  df_test_ratings = pd.DataFrame(test_ratings)
  df_test_ratings['fake_rating'] = 1
  df_test_ratings.to_csv(
      os.path.join(FLAGS.data_dir, dataset_name + '-' + TEST_RATINGS_FILENAME),
      index=False, header=False, sep='\t')

  df_test_negs = pd.DataFrame(test_negs)
  df_test_negs.to_csv(
      os.path.join(FLAGS.data_dir, dataset_name + '-' + TEST_NEG_FILENAME),
      index=False, header=False, sep='\t')


def main(_):
  """Download and extract the data from grouplens website."""
  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Download the dataset
  dataset_zip = FLAGS.dataset + '.zip'
  file_path = os.path.join(FLAGS.data_dir, dataset_zip)
  if not os.path.exists(file_path):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          file_path, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    file_path, _ = urllib.request.urlretrieve(
        DATA_URL + dataset_zip, file_path, _progress)
    statinfo = os.stat(file_path)
    print('\nSuccessfully downloaded', file_path, statinfo.st_size, 'bytes.')

  # Unzip the dataset
  if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
    zipfile.ZipFile(file_path, 'r').extractall(FLAGS.data_dir)

  # Preprocess and parse the dataset to csv
  if not os.path.exists(
      os.path.join(
          FLAGS.data_dir, FLAGS.dataset + '-' +TRAIN_RATINGS_FILENAME)):
    parse_file_to_csv(FLAGS.data_dir, FLAGS.dataset)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir', type=str, default='/tmp/ml_data',
      help='Directory to download data and extract the zip.')
  parser.add_argument(
      '--dataset', type=str, default='ml-1m', choices=['ml-1m', 'ml-20m'],
      help='Dataset to be trained and evaluated.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
