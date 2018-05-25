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
"""Download and extract the MovieLens dataset from GroupLens website.

Download the dataset, and perform data-preprocessing to convert the raw dataset
into csv file to be used in model training and evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
import zipfile

# pylint: disable=g-bad-import-order
import numpy as np
import pandas as pd
from six.moves import urllib  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.recommendation import constants
from official.utils.flags import core as flags_core

# URL to download dataset
_DATA_URL = "http://files.grouplens.org/datasets/movielens/"

_USER_COLUMN = "user_id"
_ITEM_COLUMN = "item_id"
_RATING_COLUMN = "rating"
_TIMESTAMP_COLUMN = "timestamp"
# The number of negative examples attached with a positive example
# in training dataset. It is set as 100 in the paper.
_NUMBER_NEGATIVES = 100
# In both datasets, each user has at least 20 ratings.
_MIN_NUM_RATINGS = 20

RatingData = collections.namedtuple(
    "RatingData", ["items", "users", "ratings", "min_date", "max_date"])


def _print_ratings_description(ratings):
  """Describe the rating dataset information.

  Args:
    ratings: A pandas DataFrame of the rating dataset.
  """
  info = RatingData(items=len(ratings[_ITEM_COLUMN].unique()),
                    users=len(ratings[_USER_COLUMN].unique()),
                    ratings=len(ratings),
                    min_date=ratings[_TIMESTAMP_COLUMN].min(),
                    max_date=ratings[_TIMESTAMP_COLUMN].max())
  tf.logging.info("{ratings} ratings on {items} items from {users} users"
                  " from {min_date} to {max_date}".format(**(info._asdict())))


def process_movielens(ratings, sort=True):
  """Sort and convert timestamp of the MovieLens dataset.

  Args:
    ratings: A pandas DataFrame of the rating dataset.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    ratings: The processed pandas DataFrame.
  """
  ratings[_TIMESTAMP_COLUMN] = pd.to_datetime(
      ratings[_TIMESTAMP_COLUMN], unit="s")
  if sort:
    ratings.sort_values(by=_TIMESTAMP_COLUMN, inplace=True)
  _print_ratings_description(ratings)
  return ratings


def load_movielens_1_million(file_name, sort=True):
  """Load the MovieLens 1 million dataset.

  The file has no header row, and each line is in the following format:
    UserID::MovieID::Rating::Timestamp
      - UserIDs range between 1 and 6040
      - MovieIDs range between 1 and 3952
      - Ratings are made on a 5-star scale (whole-star ratings only)
      - Timestamp is represented in seconds since midnight Coordinated Universal
        Time (UTC) of January 1, 1970.
      - Each user has at least 20 ratings

  Args:
    file_name: A string of the file name to be loaded.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    A processed pandas DataFrame of the rating dataset.
  """
  names = [_USER_COLUMN, _ITEM_COLUMN, _RATING_COLUMN, _TIMESTAMP_COLUMN]
  ratings = pd.read_csv(file_name, sep="::", names=names, engine="python")
  return process_movielens(ratings, sort=sort)


def load_movielens_20_million(file_name, sort=True):
  """Load the MovieLens 20 million dataset.

  Each line of this file after the header row represents one rating of one movie
  by one user, and has the following format:
    userId,movieId,rating,timestamp
    - The lines within this file are ordered first by userId, then, within user,
      by movieId.
    - Ratings are made on a 5-star scale, with half-star increments
      (0.5 stars - 5.0 stars).
    - Timestamps represent seconds since midnight Coordinated Universal Time
      (UTC) of January 1, 1970.
    - All the users had rated at least 20 movies.


  Args:
    file_name: A string of the file name to be loaded.
    sort: A boolean to indicate whether to sort the data based on timestamp.

  Returns:
    A processed pandas DataFrame of the rating dataset.
  """
  ratings = pd.read_csv(file_name)
  names = {"userId": _USER_COLUMN, "movieId": _ITEM_COLUMN}
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
  dataset_name = os.path.basename(file_name).split(".")[0]
  # ml-1m with extension .dat
  file_extension = ".dat"
  func = load_movielens_1_million
  if dataset_name == "ml-20m":
    file_extension = ".csv"
    func = load_movielens_20_million
  ratings_file = os.path.join(file_name, "ratings" + file_extension)
  return func(ratings_file, sort=sort)


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
  df.sort_values(by=_TIMESTAMP_COLUMN, inplace=True)
  all_ratings = set(zip(df[_USER_COLUMN], df[_ITEM_COLUMN]))
  user_to_items = collections.defaultdict(list)

  # Generate user_item rating matrix for training
  t1 = time.time()
  row_count = 0
  for row in df.itertuples():
    user_to_items[getattr(row, _USER_COLUMN)].append(getattr(row, _ITEM_COLUMN))
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


def parse_file_to_csv(data_dir, dataset_name):
  """Parse the raw data to csv file to be used in model training and evaluation.

  ml-1m dataset is small in size (~25M), while ml-20m is large (~500M). It may
  take several minutes to process ml-20m dataset.

  Args:
    data_dir: A string, the directory with the unzipped dataset.
    dataset_name: A string, the dataset name to be processed.
  """

  # Use random seed as parameter
  np.random.seed(0)

  # Load the file as DataFrame
  file_path = os.path.join(data_dir, dataset_name)
  df = load_file_to_df(file_path, sort=False)

  # Get the info of users who have more than 20 ratings on items
  grouped = df.groupby(_USER_COLUMN)
  df = grouped.filter(lambda x: len(x) >= _MIN_NUM_RATINGS)
  original_users = df[_USER_COLUMN].unique()
  original_items = df[_ITEM_COLUMN].unique()

  # Map the ids of user and item to 0 based index for following processing
  tf.logging.info("Generating user_map and item_map...")
  user_map = {user: index for index, user in enumerate(original_users)}
  item_map = {item: index for index, item in enumerate(original_items)}

  df[_USER_COLUMN] = df[_USER_COLUMN].apply(lambda user: user_map[user])
  df[_ITEM_COLUMN] = df[_ITEM_COLUMN].apply(lambda item: item_map[item])
  assert df[_USER_COLUMN].max() == len(original_users) - 1
  assert df[_ITEM_COLUMN].max() == len(original_items) - 1

  # Generate data for train and test
  all_ratings, test_ratings, test_negs = generate_train_eval_data(
      df, original_users, original_items)

  # Serialize to csv file. Each csv file contains three columns
  # (user_id, item_id, interaction)
  # As there are only two fields (user_id, item_id) in all_ratings and
  # test_ratings, we need to add a fake rating to make three columns
  df_train_ratings = pd.DataFrame(all_ratings)
  df_train_ratings["fake_rating"] = 1
  train_ratings_file = os.path.join(
      FLAGS.data_dir, dataset_name + "-" + constants.TRAIN_RATINGS_FILENAME)
  df_train_ratings.to_csv(
      train_ratings_file,
      index=False, header=False, sep="\t")
  tf.logging.info("Train ratings is {}".format(train_ratings_file))

  df_test_ratings = pd.DataFrame(test_ratings)
  df_test_ratings["fake_rating"] = 1
  test_ratings_file = os.path.join(
      FLAGS.data_dir, dataset_name + "-" + constants.TEST_RATINGS_FILENAME)
  df_test_ratings.to_csv(
      test_ratings_file,
      index=False, header=False, sep="\t")
  tf.logging.info("Test ratings is {}".format(test_ratings_file))

  df_test_negs = pd.DataFrame(test_negs)
  test_negs_file = os.path.join(
      FLAGS.data_dir, dataset_name + "-" + constants.TEST_NEG_FILENAME)
  df_test_negs.to_csv(
      test_negs_file,
      index=False, header=False, sep="\t")
  tf.logging.info("Test negatives is {}".format(test_negs_file))


def make_dir(file_dir):
  if not tf.gfile.Exists(file_dir):
    tf.logging.info("Creating directory {}".format(file_dir))
    tf.gfile.MakeDirs(file_dir)


def main(_):
  """Download and extract the data from GroupLens website."""
  tf.logging.set_verbosity(tf.logging.INFO)

  make_dir(FLAGS.data_dir)

  assert FLAGS.dataset, (
      "Please specify which dataset to download. "
      "Two datasets are available: ml-1m and ml-20m.")

  # Download the zip dataset
  dataset_zip = FLAGS.dataset + ".zip"
  file_path = os.path.join(FLAGS.data_dir, dataset_zip)
  if not tf.gfile.Exists(file_path):
    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:.1f}%".format(
          file_path, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    file_path, _ = urllib.request.urlretrieve(
        _DATA_URL + dataset_zip, file_path, _progress)
    statinfo = os.stat(file_path)
    # A new line to clear the carriage return from download progress
    # tf.logging.info is not applicable here
    print()
    tf.logging.info(
        "Successfully downloaded {} {} bytes".format(
            file_path, statinfo.st_size))

  # Unzip the dataset
  if not tf.gfile.Exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
    zipfile.ZipFile(file_path, "r").extractall(FLAGS.data_dir)

  # Preprocess and parse the dataset to csv
  train_ratings = FLAGS.dataset + "-" + constants.TRAIN_RATINGS_FILENAME
  if not tf.gfile.Exists(os.path.join(FLAGS.data_dir, train_ratings)):
    parse_file_to_csv(FLAGS.data_dir, FLAGS.dataset)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/movielens-data/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))

  flags.DEFINE_enum(
      name="dataset", default=None,
      enum_values=["ml-1m", "ml-20m"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Dataset to be trained and evaluated. Two datasets are available "
          ": ml-1m and ml-20m."))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
