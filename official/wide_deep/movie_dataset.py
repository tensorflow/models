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

import ast
import os
import subprocess
import sys
import tempfile
import zipfile

from absl import app as absl_app
from absl import flags
import numpy as np
import pandas as pd
from six.moves import urllib  # pylint: disable=redefined-builtin
import tensorflow as tf

from official.utils.flags import core as flags_core


_RATINGS = "ratings_small.csv"
_METADATA = "movies_metadata.csv"
_FILES = ["credits.csv", "keywords.csv", "links.csv", "links_small.csv",
          _METADATA, _RATINGS, "ratings_small.csv"]
_ZIP = "the-movies-dataset.zip"



def download_and_extract(data_dir):
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


def read_and_process_data(data_dir):
  with tf.gfile.Open(os.path.join(data_dir, _RATINGS)) as f:
    ratings = pd.read_csv(f)

  with tf.gfile.Open(os.path.join(data_dir, _METADATA)) as f:
    metadata = pd.read_csv(f)

  metadata.rename(columns={"id":"movieId"}, inplace=True)
  metadata["movieId"] = pd.to_numeric(metadata["movieId"], downcast="integer",
                                      errors="coerce")
  metadata = metadata[pd.notna(metadata["movieId"])]
  metadata["movieId"] = metadata["movieId"].astype("int")

  metadata = metadata[metadata["movieId"].isin(ratings["movieId"])]
  ratings = ratings[ratings["movieId"].isin(metadata["movieId"])]

  metadata['genres'] = metadata['genres'].fillna('[]').apply(
      ast.literal_eval).apply(
      lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

  # Ideally we should generate sparse tensors for Genre, since there are
  # multiple genres for each movie and their number is not fixed. Creating
  # sparse tensors from a pandas dataframe is not straightforward. To keep our
  # code simple (maintainable), we'll pick only top two genres as a feature.
  metadata['genres_0'] = metadata['genres'].apply(
      lambda x: x[0] if len(x) > 0 else 'NA')
  metadata['genres_1'] = metadata['genres'].apply(
      lambda x: x[1] if len(x) > 1 else 'NA')

  metadata['year'] = pd.to_datetime(
      metadata['release_date'], errors='coerce').apply(
      lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
  metadata['year'] = metadata['year'].apply(
      lambda x: int(x) if x != 'NaT' else 0)

  return ratings, metadata


def construct_feature_columns(metadata, ratings):
  user_ids = sorted(ratings['userId'].unique())
  movie_ids = sorted(ratings['movieId'].unique())

  bucketed_budgets = list(metadata['budget'].astype("int").quantile(
      np.linspace(0.05, 1., num=19, endpoint=False)).unique())

  fc_user_id = tf.feature_column.categorical_column_with_vocabulary_list(
      'userId', user_ids)
  fc_user_embedding = tf.feature_column.embedding_column(fc_user_id, 8)
  fc_movie_id = tf.feature_column.categorical_column_with_vocabulary_list(
      'movieId', movie_ids)
  fc_movie_embedding = tf.feature_column.embedding_column(fc_movie_id, 8)

  # Bucket budgets
  fc_budget = tf.feature_column.bucketized_column(
      tf.feature_column.numeric_column('budget'), boundaries = bucketed_budgets)

  fc_genres_0, fc_genres_1 = [
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            i, sorted(metadata[i].unique())))
    for i in ["genres_0", "genres_1"]
  ]

  fc_original_language = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          'original_language', sorted(metadata['original_language'].unique())))

  fc_year = tf.feature_column.bucketized_column(
      tf.feature_column.numeric_column('year'),
      boundaries = [1800] + list(range(1940, 2010, 10)))

  return [fc_user_id, fc_user_embedding, fc_genres_0, fc_budget, fc_genres_1,
          fc_original_language, fc_year]



def get_input_fns(data_dir, repeat=1, batch_size=32):
  ratings, metadata = read_and_process_data(data_dir)
  feature_columns = construct_feature_columns(metadata, ratings)

  def model_column_fn():
    return feature_columns

  movie_columns = ['movieId', 'year', 'original_language',
                   'genres_0', 'genres_1', 'budget']

  merged_data = pd.merge(ratings, metadata[movie_columns], on='movieId')

  train_df=merged_data.sample(frac=0.8,random_state=0)
  eval_df=merged_data.drop(train_df.index)

  train_df = train_df.reset_index(drop=True)
  eval_df = eval_df.reset_index(drop=True)

  def _construct_input_fn(dataframe, repeat=1, batch_size=1, shuffle=False):
    def input_fn():
      dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))
      if shuffle:
        dataset = dataset.shuffle(dataframe.shape[0])
      dataset = dataset.repeat(repeat)

      dataset = dataset.apply(
          tf.contrib.data.map_and_batch(
              lambda features: (features, features['rating']),
              batch_size=batch_size,
              num_parallel_batches=1,
              drop_remainder=False))

      return dataset.prefetch(1)
    return input_fn

  train_input_fn = _construct_input_fn(train_df, repeat, batch_size, True)
  eval_input_fn = _construct_input_fn(eval_df, 1, batch_size, False)

  return train_input_fn, eval_input_fn, model_column_fn


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/kaggle-movies/",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))


def main(_):
  download_and_extract(flags.FLAGS.data_dir)
  get_input_fns(flags.FLAGS.data_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
