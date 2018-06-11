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
"""Download Kaggle movie dataset and construct input_fn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile

from absl import app as absl_app
from absl import flags
import numpy as np
import pandas as pd
import six
from six.moves import urllib  # pylint: disable=redefined-builtin
import tensorflow as tf

from official.utils.flags import core as flags_core


_RATINGS = "ratings.csv"
_RATINGS_SMALL = "ratings_small.csv"
_METADATA = "movies_metadata.csv"
_FILES = ["credits.csv", "keywords.csv", "links.csv", "links_small.csv",
          _METADATA, "ratings.csv", "ratings_small.csv"]
_ZIP = "the-movies-dataset.zip"


_BUFFER_SUBDIR = "data_buffer"
_FEATURE_MAP = {
    "userId": tf.FixedLenFeature([1], dtype=tf.int64),
    "movieId": tf.FixedLenFeature([1], dtype=tf.int64),
    "year": tf.FixedLenFeature([1], dtype=tf.int64),
    "original_language": tf.FixedLenFeature([], dtype=tf.string),
    "genres_0": tf.FixedLenFeature([], dtype=tf.string),
    "genres_1": tf.FixedLenFeature([], dtype=tf.string),
    "budget": tf.FixedLenFeature([1], dtype=tf.float32),
    "rating": tf.FixedLenFeature([1], dtype=tf.float32)
}

_BUFFER_SIZE = {
    "small_train": 6797296,
    "small_eval": 1699301,
    "train": 1738258785,
    "eval": 434556365
}


def download_and_extract(data_dir):
  """Download Kaggle movie dataset."""
  skip = all([tf.gfile.Exists(os.path.join(data_dir, i)) for i in _FILES])

  if not skip:
    try:
      subprocess.check_output(args=["kaggle", "-h"])
    except:
      tf.logging.error("Please ensure that you have installed kaggle.")
      raise

    if tf.gfile.Exists(data_dir):
      tf.gfile.DeleteRecursively(data_dir)
    tf.gfile.MakeDirs(data_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
      subprocess.call(args=["kaggle", "datasets", "download", "-d",
                            "rounakbanik/the-movies-dataset", "-p", temp_dir])
      tf.gfile.Remove(os.path.join(temp_dir, _ZIP))
      files = os.listdir(temp_dir)
      assert set(files) == set(_FILES)
      for i in files:
        tf.gfile.Copy(os.path.join(temp_dir, i), os.path.join(data_dir, i))
        print(i.ljust(20), "copied")


def read_and_process_data(data_dir, small=True):
  """Convert raw csv's into processed dataframes."""
  ratings_file = _RATINGS_SMALL if small else _RATINGS
  with tf.gfile.Open(os.path.join(data_dir, ratings_file)) as f:
    ratings = pd.read_csv(f)

  with tf.gfile.Open(os.path.join(data_dir, _METADATA)) as f:
    metadata = pd.read_csv(f)

  metadata.rename(columns={"id": "movieId"}, inplace=True)
  metadata["movieId"] = pd.to_numeric(metadata["movieId"], downcast="integer",
                                      errors="coerce")
  metadata = metadata[pd.notna(metadata["movieId"])]
  metadata["movieId"] = metadata["movieId"].astype("int")

  metadata = metadata[metadata["movieId"].isin(ratings["movieId"])]
  ratings = ratings[ratings["movieId"].isin(metadata["movieId"])]

  metadata["budget"] = metadata["budget"].astype("float")

  metadata["genres"] = metadata["genres"].fillna("[]").apply(
      ast.literal_eval).apply(
          lambda x: [i["name"] for i in x] if isinstance(x, list) else [])

  # Ideally we should generate sparse tensors for Genre, since there are
  # multiple genres for each movie and their number is not fixed. Creating
  # sparse tensors from a pandas dataframe is not straightforward. To keep our
  # code simple (maintainable), we"ll pick only top two genres as a feature.
  metadata["genres_0"] = metadata["genres"].apply(
      lambda x: x[0] if x else "NA")
  metadata["genres_1"] = metadata["genres"].apply(
      lambda x: x[1] if len(x) > 1 else "NA")

  metadata["original_language"] = metadata["original_language"].fillna("NA")

  metadata["year"] = pd.to_datetime(
      metadata["release_date"], errors="coerce").apply(
          lambda x: str(x).split("-")[0] if x != np.nan else np.nan)
  metadata["year"] = metadata["year"].apply(
      lambda x: int(x) if x != "NaT" else 0)

  return ratings, metadata


def construct_feature_columns(metadata, ratings):
  """Construct feature columns for processed movie data."""
  user_ids = sorted(ratings["userId"].unique())
  movie_ids = sorted(ratings["movieId"].unique())

  bucketed_budgets = list(metadata["budget"].quantile(
      np.linspace(0.05, 1., num=19, endpoint=False)).unique())

  fc_user_id = tf.feature_column.categorical_column_with_vocabulary_list(
      "userId", user_ids)
  fc_user_embedding = tf.feature_column.embedding_column(fc_user_id, 8)
  fc_movie_id = tf.feature_column.categorical_column_with_vocabulary_list(
      "movieId", movie_ids)
  fc_movie_embedding = tf.feature_column.embedding_column(fc_movie_id, 8)

  # Bucket budgets
  fc_budget = tf.feature_column.bucketized_column(
      tf.feature_column.numeric_column("budget"), boundaries=bucketed_budgets)

  fc_genres_0, fc_genres_1 = [
      tf.feature_column.indicator_column(
          tf.feature_column.categorical_column_with_vocabulary_list(
              i, sorted(metadata[i].unique())))
      for i in ["genres_0", "genres_1"]
  ]

  fc_original_language = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          "original_language", sorted(metadata["original_language"].unique())))

  fc_year = tf.feature_column.bucketized_column(
      tf.feature_column.numeric_column("year"),
      boundaries=[1800] + list(range(1940, 2010, 10)))

  return [fc_movie_embedding, fc_user_embedding, fc_genres_0, fc_budget,
          fc_genres_1, fc_original_language, fc_year]


def _write_to_buffer(dataframe, buffer_path, columns):
  """Write a dataframe to a binary file for a dataset to consume."""
  if tf.gfile.Exists(buffer_path):
    expected_size = _BUFFER_SIZE[os.path.split(buffer_path)[1]]
    actual_size = os.stat(buffer_path).st_size
    if expected_size == actual_size:
      return
    tf.logging.warning(
        "Existing buffer {} has size {}. Expected size {}. Deleting and "
        "rebuilding buffer.".format(buffer_path, actual_size, expected_size))
    tf.gfile.Remove(buffer_path)

  tf.logging.info("Constructing {}".format(buffer_path))
  with tf.python_io.TFRecordWriter(buffer_path) as writer:
    for row in dataframe.itertuples():
      i = getattr(row, "Index")
      features = {}
      for key in columns:
        value = getattr(row, key)
        if isinstance(value, int):
          features[key] = tf.train.Feature(
              int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
          features[key] = tf.train.Feature(
              float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, six.text_type):
          if not isinstance(value, six.binary_type):
            value = value.encode("utf-8")
          features[key] = tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[value]))
        else:
          raise ValueError("Unknown type.")
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      if (i + 1) % 50000 == 0:
        tf.logging.info(
            "{}/{} examples written.".format(str(i+1).ljust(8), len(dataframe)))


def _deserialize(example_serialized):
  features = tf.parse_single_example(example_serialized, _FEATURE_MAP)
  return features, features["rating"]


def get_input_fns(data_dir, small=True, repeat=1, batch_size=32):
  """Construct training and eval input_fns."""
  ratings, metadata = read_and_process_data(data_dir, small)
  feature_columns = construct_feature_columns(metadata, ratings)

  def model_column_fn():
    # There are no wide columns. Only deep columns for now.
    return [], feature_columns

  movie_columns = ["movieId", "year", "original_language",
                   "genres_0", "genres_1", "budget"]

  merged_data = pd.merge(ratings, metadata[movie_columns], on="movieId")
  all_columns = ["userId"] + movie_columns + ["rating"]

  train_df = merged_data.sample(frac=0.8, random_state=0)
  eval_df = merged_data.drop(train_df.index)

  train_df = train_df.reset_index(drop=True)
  eval_df = eval_df.reset_index(drop=True)

  tf.logging.info("Training: {} examples".format(len(train_df)))
  tf.logging.info("Eval:     {} examples".format(len(eval_df)))

  buffer_dir = os.path.join(data_dir, _BUFFER_SUBDIR)
  tf.gfile.MakeDirs(buffer_dir)

  def _construct_input_fn(dataframe, buffer_name, repeat=1, batch_size=1):
    """Construct a file backed dataset and input_fn from a dataframe."""
    buffer_path = os.path.join(buffer_dir, buffer_name)
    _write_to_buffer(
        dataframe=dataframe, buffer_path=buffer_path, columns=all_columns)

    def input_fn():
      dataset = tf.data.TFRecordDataset(buffer_path)
      dataset = dataset.repeat(repeat)
      dataset = dataset.map(_deserialize)
      dataset = dataset.batch(batch_size)
      return dataset.prefetch(1)

    return input_fn

  prefix = "small_" if small else ""
  train_input_fn = _construct_input_fn(
      dataframe=train_df, buffer_name="{}train".format(prefix),
      repeat=repeat, batch_size=batch_size)
  eval_input_fn = _construct_input_fn(
      dataframe=eval_df, buffer_name="{}eval".format(prefix),
      repeat=repeat, batch_size=batch_size)

  return train_input_fn, eval_input_fn, model_column_fn


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/kaggle-movies/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))
  flags.DEFINE_boolean(
      name="construct_buffers", short_name="buffer",
      default=False, help=flags_core.help_wrap(
          "Construct binary files for the dataset pipeline now rather than at "
          "the start of training."))


def main(_):
  download_and_extract(flags.FLAGS.data_dir)
  if flags.FLAGS.construct_buffers:
    tf.logging.info("Constructing small dataset buffers.")
    get_input_fns(flags.FLAGS.data_dir, small=True)
    tf.logging.info("Constructing large dataset buffers.")
    get_input_fns(flags.FLAGS.data_dir, small=False)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
