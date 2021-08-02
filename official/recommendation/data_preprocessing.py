# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Preprocess dataset and construct any necessary artifacts."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
import pickle
import time
import timeit
import typing
from typing import Dict, Text, Tuple

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import movielens


_EXPECTED_CACHE_KEYS = (rconst.TRAIN_USER_KEY, rconst.TRAIN_ITEM_KEY,
                        rconst.EVAL_USER_KEY, rconst.EVAL_ITEM_KEY,
                        rconst.USER_MAP, rconst.ITEM_MAP)


def read_dataframe(
    raw_rating_path: Text
) -> Tuple[Dict[int, int], Dict[int, int], pd.DataFrame]:
  """Read in data CSV, and output DataFrame for downstream processing.

  This function reads in the raw CSV of positive items, and performs three
  preprocessing transformations:

  1)  Filter out all users who have not rated at least a certain number
      of items. (Typically 20 items)

  2)  Zero index the users and items such that the largest user_id is
      `num_users - 1` and the largest item_id is `num_items - 1`

  3)  Sort the dataframe by user_id, with timestamp as a secondary sort key.
      This allows the dataframe to be sliced by user in-place, and for the last
      item to be selected simply by calling the `-1` index of a user's slice.

  Args:
    raw_rating_path: The path to the CSV which contains the raw dataset.

  Returns:
    A dict mapping raw user IDs to regularized user IDs, a dict mapping raw
    item IDs to regularized item IDs, and a filtered, zero-index remapped,
    sorted dataframe.
  """
  with tf.io.gfile.GFile(raw_rating_path) as f:
    df = pd.read_csv(f)

  # Get the info of users who have more than 20 ratings on items
  grouped = df.groupby(movielens.USER_COLUMN)
  df = grouped.filter(
      lambda x: len(x) >= rconst.MIN_NUM_RATINGS)  # type: pd.DataFrame

  original_users = df[movielens.USER_COLUMN].unique()
  original_items = df[movielens.ITEM_COLUMN].unique()

  # Map the ids of user and item to 0 based index for following processing
  logging.info("Generating user_map and item_map...")
  user_map = {user: index for index, user in enumerate(original_users)}
  item_map = {item: index for index, item in enumerate(original_items)}

  df[movielens.USER_COLUMN] = df[movielens.USER_COLUMN].apply(
      lambda user: user_map[user])
  df[movielens.ITEM_COLUMN] = df[movielens.ITEM_COLUMN].apply(
      lambda item: item_map[item])

  num_users = len(original_users)
  num_items = len(original_items)

  assert num_users <= np.iinfo(rconst.USER_DTYPE).max
  assert num_items <= np.iinfo(rconst.ITEM_DTYPE).max
  assert df[movielens.USER_COLUMN].max() == num_users - 1
  assert df[movielens.ITEM_COLUMN].max() == num_items - 1

  # This sort is used to shard the dataframe by user, and later to select
  # the last item for a user to be used in validation.
  logging.info("Sorting by user, timestamp...")

  # This sort is equivalent to
  #   df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
  #   inplace=True)
  # except that the order of items with the same user and timestamp are
  # sometimes different. For some reason, this sort results in a better
  # hit-rate during evaluation, matching the performance of the MLPerf
  # reference implementation.
  df.sort_values(by=movielens.TIMESTAMP_COLUMN, inplace=True)
  df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                 inplace=True,
                 kind="mergesort")

  # The dataframe does not reconstruct indices in the sort or filter steps.
  return user_map, item_map, df.reset_index()


def _filter_index_sort(raw_rating_path: Text,
                       cache_path: Text) -> Tuple[pd.DataFrame, bool]:
  """Read in data CSV, and output structured data.

  This function reads in the raw CSV of positive items, and performs three
  preprocessing transformations:

  1)  Filter out all users who have not rated at least a certain number
      of items. (Typically 20 items)

  2)  Zero index the users and items such that the largest user_id is
      `num_users - 1` and the largest item_id is `num_items - 1`

  3)  Sort the dataframe by user_id, with timestamp as a secondary sort key.
      This allows the dataframe to be sliced by user in-place, and for the last
      item to be selected simply by calling the `-1` index of a user's slice.

  While all of these transformations are performed by Pandas (and are therefore
  single-threaded), they only take ~2 minutes, and the overhead to apply a
  MapReduce pattern to parallel process the dataset adds significant complexity
  for no computational gain. For a larger dataset parallelizing this
  preprocessing could yield speedups. (Also, this preprocessing step is only
  performed once for an entire run.

  Args:
    raw_rating_path: The path to the CSV which contains the raw dataset.
    cache_path: The path to the file where results of this function are saved.

  Returns:
    A filtered, zero-index remapped, sorted dataframe, a dict mapping raw user
    IDs to regularized user IDs, and a dict mapping raw item IDs to regularized
    item IDs.
  """
  valid_cache = tf.io.gfile.exists(cache_path)
  if valid_cache:
    with tf.io.gfile.GFile(cache_path, "rb") as f:
      cached_data = pickle.load(f)

    # (nnigania)disabled this check as the dataset is not expected to change
    # cache_age = time.time() - cached_data.get("create_time", 0)
    # if cache_age > rconst.CACHE_INVALIDATION_SEC:
    #   valid_cache = False

    for key in _EXPECTED_CACHE_KEYS:
      if key not in cached_data:
        valid_cache = False

    if not valid_cache:
      logging.info("Removing stale raw data cache file.")
      tf.io.gfile.remove(cache_path)

  if valid_cache:
    data = cached_data
  else:
    user_map, item_map, df = read_dataframe(raw_rating_path)

    grouped = df.groupby(movielens.USER_COLUMN, group_keys=False)
    eval_df, train_df = grouped.tail(1), grouped.apply(lambda x: x.iloc[:-1])

    data = {
        rconst.TRAIN_USER_KEY:
            train_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE),
        rconst.TRAIN_ITEM_KEY:
            train_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE),
        rconst.EVAL_USER_KEY:
            eval_df[movielens.USER_COLUMN].values.astype(rconst.USER_DTYPE),
        rconst.EVAL_ITEM_KEY:
            eval_df[movielens.ITEM_COLUMN].values.astype(rconst.ITEM_DTYPE),
        rconst.USER_MAP:
            user_map,
        rconst.ITEM_MAP:
            item_map,
        "create_time":
            time.time(),
    }

    logging.info("Writing raw data cache.")
    with tf.io.gfile.GFile(cache_path, "wb") as f:
      pickle.dump(data, f, protocol=4)

  # TODO(robieta): MLPerf cache clear.
  return data, valid_cache


def instantiate_pipeline(dataset,
                         data_dir,
                         params,
                         constructor_type=None,
                         deterministic=False,
                         epoch_dir=None,
                         generate_data_offline=False):
  # type: (str, str, dict, typing.Optional[str], bool, typing.Optional[str], bool) -> (int, int, data_pipeline.BaseDataConstructor)
  """Load and digest data CSV into a usable form.

  Args:
    dataset: The name of the dataset to be used.
    data_dir: The root directory of the dataset.
    params: dict of parameters for the run.
    constructor_type: The name of the constructor subclass that should be used
      for the input pipeline.
    deterministic: Tell the data constructor to produce deterministically.
    epoch_dir: Directory in which to store the training epochs.
    generate_data_offline: Boolean, whether current pipeline is done offline or
      while training.
  """
  logging.info("Beginning data preprocessing.")

  st = timeit.default_timer()
  raw_rating_path = os.path.join(data_dir, dataset, movielens.RATINGS_FILE)
  cache_path = os.path.join(data_dir, dataset, rconst.RAW_CACHE_FILE)

  raw_data, _ = _filter_index_sort(raw_rating_path, cache_path)
  user_map, item_map = raw_data["user_map"], raw_data["item_map"]
  num_users, num_items = movielens.DATASET_TO_NUM_USERS_AND_ITEMS[dataset]

  if num_users != len(user_map):
    raise ValueError("Expected to find {} users, but found {}".format(
        num_users, len(user_map)))
  if num_items != len(item_map):
    raise ValueError("Expected to find {} items, but found {}".format(
        num_items, len(item_map)))

  producer = data_pipeline.get_constructor(constructor_type or "materialized")(
      maximum_number_epochs=params["train_epochs"],
      num_users=num_users,
      num_items=num_items,
      user_map=user_map,
      item_map=item_map,
      train_pos_users=raw_data[rconst.TRAIN_USER_KEY],
      train_pos_items=raw_data[rconst.TRAIN_ITEM_KEY],
      train_batch_size=params["batch_size"],
      batches_per_train_step=params["batches_per_step"],
      num_train_negatives=params["num_neg"],
      eval_pos_users=raw_data[rconst.EVAL_USER_KEY],
      eval_pos_items=raw_data[rconst.EVAL_ITEM_KEY],
      eval_batch_size=params["eval_batch_size"],
      batches_per_eval_step=params["batches_per_step"],
      stream_files=params["stream_files"],
      deterministic=deterministic,
      epoch_dir=epoch_dir,
      create_data_offline=generate_data_offline)

  run_time = timeit.default_timer() - st
  logging.info(
      "Data preprocessing complete. Time: {:.1f} sec.".format(run_time))

  print(producer)
  return num_users, num_items, producer
