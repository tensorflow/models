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

Download the dataset, and perform basic preprocessing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import zipfile

# pylint: disable=g-bad-import-order
import numpy as np
import pandas as pd
import six
from six.moves import urllib  # pylint: disable=redefined-builtin
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.utils.flags import core as flags_core


ML_1M = "ml-1m"
ML_20M = "ml-20m"
DATASETS = [ML_1M, ML_20M]

RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"

# URL to download dataset
_DATA_URL = "http://files.grouplens.org/datasets/movielens/"

GENRE_COLUMN = "genres"
ITEM_COLUMN = "item_id"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "user_id"

GENRES = [
    'Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX", 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
N_GENRE = len(GENRES)

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]

# Note: Users are indexed [1, k], not [0, k-1]
NUM_USER_IDS = {
    ML_1M: 6040,
    ML_20M: 138493,
}

# Note: Movies are indexed [1, k], not [0, k-1]
# Both the 1m and 20m datasets use the same movie set.
NUM_ITEM_IDS = 3952

MAX_RATING = 5

NUM_RATINGS = {
    ML_1M: 1000209,
    ML_20M: 20000263
}


def _download_and_clean(dataset, data_dir):
  """Download MovieLens dataset in a standard format.

  This function downloads the specified MovieLens format and coerces it into a
  standard format. The only difference between the ml-1m and ml-20m datasets
  after this point (other than size, of course) is that the 1m dataset uses
  whole number ratings while the 20m dataset allows half integer ratings.
  """
  if dataset not in DATASETS:
    raise ValueError("dataset {} is not in {{{}}}".format(
        dataset, ",".join(DATASETS)))

  data_subdir = os.path.join(data_dir, dataset)

  expected_files = ["{}.zip".format(dataset), RATINGS_FILE, MOVIES_FILE]

  tf.io.gfile.makedirs(data_subdir)
  if set(expected_files).intersection(
      tf.io.gfile.listdir(data_subdir)) == set(expected_files):
    logging.info("Dataset {} has already been downloaded".format(dataset))
    return

  url = "{}{}.zip".format(_DATA_URL, dataset)

  temp_dir = tempfile.mkdtemp()
  try:
    zip_path = os.path.join(temp_dir, "{}.zip".format(dataset))
    zip_path, _ = urllib.request.urlretrieve(url, zip_path)
    statinfo = os.stat(zip_path)
    # A new line to clear the carriage return from download progress
    # logging.info is not applicable here
    print()
    logging.info(
        "Successfully downloaded {} {} bytes".format(
            zip_path, statinfo.st_size))

    zipfile.ZipFile(zip_path, "r").extractall(temp_dir)

    if dataset == ML_1M:
      _regularize_1m_dataset(temp_dir)
    else:
      _regularize_20m_dataset(temp_dir)

    for fname in tf.io.gfile.listdir(temp_dir):
      if not tf.io.gfile.exists(os.path.join(data_subdir, fname)):
        tf.io.gfile.copy(os.path.join(temp_dir, fname),
                         os.path.join(data_subdir, fname))
      else:
        logging.info("Skipping copy of {}, as it already exists in the "
                     "destination folder.".format(fname))

  finally:
    tf.io.gfile.rmtree(temp_dir)


def _transform_csv(input_path, output_path, names, skip_first, separator=","):
  """Transform csv to a regularized format.

  Args:
    input_path: The path of the raw csv.
    output_path: The path of the cleaned csv.
    names: The csv column names.
    skip_first: Boolean of whether to skip the first line of the raw csv.
    separator: Character used to separate fields in the raw csv.
  """
  if six.PY2:
    names = [six.ensure_text(n, "utf-8") for n in names]

  with tf.io.gfile.GFile(output_path, "wb") as f_out, \
      tf.io.gfile.GFile(input_path, "rb") as f_in:

    # Write column names to the csv.
    f_out.write(",".join(names).encode("utf-8"))
    f_out.write(b"\n")
    for i, line in enumerate(f_in):
      if i == 0 and skip_first:
        continue  # ignore existing labels in the csv

      line = six.ensure_text(line, "utf-8", errors="ignore")
      fields = line.split(separator)
      if separator != ",":
        fields = ['"{}"'.format(field) if "," in field else field
                  for field in fields]
      f_out.write(",".join(fields).encode("utf-8"))


def _regularize_1m_dataset(temp_dir):
  """
  ratings.dat
    The file has no header row, and each line is in the following format:
    UserID::MovieID::Rating::Timestamp
      - UserIDs range from 1 and 6040
      - MovieIDs range from 1 and 3952
      - Ratings are made on a 5-star scale (whole-star ratings only)
      - Timestamp is represented in seconds since midnight Coordinated Universal
        Time (UTC) of January 1, 1970.
      - Each user has at least 20 ratings

  movies.dat
    Each line has the following format:
    MovieID::Title::Genres
      - MovieIDs range from 1 and 3952
  """
  working_dir = os.path.join(temp_dir, ML_1M)

  _transform_csv(
      input_path=os.path.join(working_dir, "ratings.dat"),
      output_path=os.path.join(temp_dir, RATINGS_FILE),
      names=RATING_COLUMNS, skip_first=False, separator="::")

  _transform_csv(
      input_path=os.path.join(working_dir, "movies.dat"),
      output_path=os.path.join(temp_dir, MOVIES_FILE),
      names=MOVIE_COLUMNS, skip_first=False, separator="::")

  tf.io.gfile.rmtree(working_dir)


def _regularize_20m_dataset(temp_dir):
  """
  ratings.csv
    Each line of this file after the header row represents one rating of one
    movie by one user, and has the following format:
    userId,movieId,rating,timestamp
    - The lines within this file are ordered first by userId, then, within user,
      by movieId.
    - Ratings are made on a 5-star scale, with half-star increments
      (0.5 stars - 5.0 stars).
    - Timestamps represent seconds since midnight Coordinated Universal Time
      (UTC) of January 1, 1970.
    - All the users had rated at least 20 movies.

  movies.csv
    Each line has the following format:
    MovieID,Title,Genres
      - MovieIDs range from 1 and 3952
  """
  working_dir = os.path.join(temp_dir, ML_20M)

  _transform_csv(
      input_path=os.path.join(working_dir, "ratings.csv"),
      output_path=os.path.join(temp_dir, RATINGS_FILE),
      names=RATING_COLUMNS, skip_first=True, separator=",")

  _transform_csv(
      input_path=os.path.join(working_dir, "movies.csv"),
      output_path=os.path.join(temp_dir, MOVIES_FILE),
      names=MOVIE_COLUMNS, skip_first=True, separator=",")

  tf.io.gfile.rmtree(working_dir)


def download(dataset, data_dir):
  if dataset:
    _download_and_clean(dataset, data_dir)
  else:
    _ = [_download_and_clean(d, data_dir) for d in DATASETS]


def ratings_csv_to_dataframe(data_dir, dataset):
  with tf.io.gfile.GFile(os.path.join(data_dir, dataset, RATINGS_FILE)) as f:
    return pd.read_csv(f, encoding="utf-8")


def csv_to_joint_dataframe(data_dir, dataset):
  ratings = ratings_csv_to_dataframe(data_dir, dataset)

  with tf.io.gfile.GFile(os.path.join(data_dir, dataset, MOVIES_FILE)) as f:
    movies = pd.read_csv(f, encoding="utf-8")

  df = ratings.merge(movies, on=ITEM_COLUMN)
  df[RATING_COLUMN] = df[RATING_COLUMN].astype(np.float32)

  return df


def integerize_genres(dataframe):
  """Replace genre string with a binary vector.

  Args:
    dataframe: a pandas dataframe of movie data.

  Returns:
    The transformed dataframe.
  """
  def _map_fn(entry):
    entry.replace("Children's", "Children")  # naming difference.
    movie_genres = entry.split("|")
    output = np.zeros((len(GENRES),), dtype=np.int64)
    for i, genre in enumerate(GENRES):
      if genre in movie_genres:
        output[i] = 1
    return output

  dataframe[GENRE_COLUMN] = dataframe[GENRE_COLUMN].apply(_map_fn)

  return dataframe


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/movielens-data/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))

  flags.DEFINE_enum(
      name="dataset", default=None,
      enum_values=DATASETS, case_sensitive=False,
      help=flags_core.help_wrap("Dataset to be trained and evaluated."))


def main(_):
  """Download and extract the data from GroupLens website."""
  download(flags.FLAGS.dataset, flags.FLAGS.data_dir)


if __name__ == "__main__":
  define_data_download_flags()
  FLAGS = flags.FLAGS
  app.run(main)
