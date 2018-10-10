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
"""Preprocess dataset and construct any necessary artifacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import contextlib
import gc
import hashlib
import multiprocessing
import json
import os
import pickle
import signal
import socket
import subprocess
import time
import timeit
import typing

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
import numpy as np
import pandas as pd
import six
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import stat_utils
from official.recommendation import popen_helper


DATASET_TO_NUM_USERS_AND_ITEMS = {
    "ml-1m": (6040, 3706),
    "ml-20m": (138493, 26744)
}


# Number of batches to run per epoch when using synthetic data. At high batch
# sizes, we run for more batches than with real data, which is good since
# running more batches reduces noise when measuring the average batches/second.
_SYNTHETIC_BATCHES_PER_EPOCH = 2000


class NCFDataset(object):
  """Container for training and testing data."""

  def __init__(self, user_map, item_map, num_data_readers, cache_paths,
               num_train_positives, deterministic=False):
    # type: (dict, dict, int, rconst.Paths) -> None
    """Assign key values for recommendation dataset.

    Args:
      user_map: Dict mapping raw user ids to regularized ids.
      item_map: Dict mapping raw item ids to regularized ids.
      num_data_readers: The number of reader Datasets used during training.
      cache_paths: Object containing locations for various cache files.
      num_train_positives: The number of positive training examples in the
        dataset.
      deterministic: Operations should use deterministic, order preserving
        methods, even at the cost of performance.
    """

    self.user_map = {int(k): int(v) for k, v in user_map.items()}
    self.item_map = {int(k): int(v) for k, v in item_map.items()}
    self.num_users = len(user_map)
    self.num_items = len(item_map)
    self.num_data_readers = num_data_readers
    self.cache_paths = cache_paths
    self.num_train_positives = num_train_positives
    self.deterministic = deterministic


def _filter_index_sort(raw_rating_path, match_mlperf):
  # type: (str, bool) -> (pd.DataFrame, dict, dict)
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
    match_mlperf: If True, change the sorting algorithm to match the MLPerf
      reference implementation.

  Returns:
    A filtered, zero-index remapped, sorted dataframe, a dict mapping raw user
    IDs to regularized user IDs, and a dict mapping raw item IDs to regularized
    item IDs.
  """
  with tf.gfile.Open(raw_rating_path) as f:
    df = pd.read_csv(f)

  # Get the info of users who have more than 20 ratings on items
  grouped = df.groupby(movielens.USER_COLUMN)
  df = grouped.filter(
      lambda x: len(x) >= rconst.MIN_NUM_RATINGS) # type: pd.DataFrame

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

  if match_mlperf:
    # This sort is equivalent to the non-MLPerf sort, except that the order of
    # items with the same user and timestamp are sometimes different. For some
    # reason, this sort results in a better hit-rate during evaluation, matching
    # the performance of the MLPerf reference implementation.
    df.sort_values(by=movielens.TIMESTAMP_COLUMN, inplace=True)
    df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                   inplace=True, kind="mergesort")
  else:
    df.sort_values([movielens.USER_COLUMN, movielens.TIMESTAMP_COLUMN],
                   inplace=True)

  df = df.reset_index()  # The dataframe does not reconstruct indicies in the
  # sort or filter steps.

  return df, user_map, item_map


def _train_eval_map_fn(args):
  # type: (...) -> typing.Dict(np.ndarray)
  """Split training and testing data and generate testing negatives.

  This function is called as part of a multiprocessing map. The principle
  input is a shard, which contains a sorted array of users and corresponding
  items for each user, where items have already been sorted in ascending order
  by timestamp. (Timestamp is not passed to avoid the serialization cost of
  sending it to the map function.)

  For each user, all but the last item is written into a pickle file which the
  training data producer can consume on as needed. The last item for a user
  is a validation point; for each validation point a number of negatives are
  generated (typically 999). The validation data is returned by this function,
  as it is held in memory for the remainder of the run.

  Args:
    shard: A dict containing the user and item arrays.
    shard_id: The id of the shard provided. This is used to number the training
      shard pickle files.
    num_items: The cardinality of the item set, which determines the set from
      which validation negatives should be drawn.
    cache_paths: rconst.Paths object containing locations for various cache
      files.
    seed: Random seed to be used when generating testing negatives.
    match_mlperf: If True, sample eval negative with replacements, which the
      MLPerf reference implementation does.

  Returns:
    A dict containing the evaluation data for a given shard.
  """

  shard, shard_id, num_items, cache_paths, seed, match_mlperf = args
  np.random.seed(seed)

  users = shard[movielens.USER_COLUMN]
  items = shard[movielens.ITEM_COLUMN]

  # This produces index boundaries which can be used to slice by user.
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

    test_negatives = stat_utils.sample_with_exclusion(
        num_items=num_items, positive_set=set(block_items),
        n=rconst.NUM_EVAL_NEGATIVES, replacement=match_mlperf)
    test_blocks.append((
        block_user[0] * np.ones((rconst.NUM_EVAL_NEGATIVES + 1,),
                                dtype=np.int32),
        np.array([block_items[-1]] + test_negatives, dtype=np.uint16)
    ))
    test_positives.append((block_user[0], block_items[-1]))

  train_users = np.concatenate([i[0] for i in train_blocks])
  train_items = np.concatenate([i[1] for i in train_blocks])

  train_shard_fpath = cache_paths.train_shard_template.format(
      str(shard_id).zfill(5))

  with tf.gfile.Open(train_shard_fpath, "wb") as f:
    pickle.dump({
        movielens.USER_COLUMN: train_users,
        movielens.ITEM_COLUMN: train_items,
    }, f)

  test_users = np.concatenate([i[0] for i in test_blocks])
  test_items = np.concatenate([i[1] for i in test_blocks])
  assert test_users.shape == test_items.shape
  assert test_items.shape[0] % (rconst.NUM_EVAL_NEGATIVES + 1) == 0

  return {
      movielens.USER_COLUMN: test_users,
      movielens.ITEM_COLUMN: test_items,
  }


def generate_train_eval_data(df, approx_num_shards, num_items, cache_paths,
                             match_mlperf):
  # type: (pd.DataFrame, int, int, rconst.Paths, bool) -> None
  """Construct training and evaluation datasets.

  This function manages dataset construction and validation that the
  transformations have produced correct results. The particular logic of
  transforming the data is performed in _train_eval_map_fn().

  Args:
    df: The dataframe containing the entire dataset. It is essential that this
      dataframe be produced by _filter_index_sort(), as subsequent
      transformations rely on `df` having particular structure.
    approx_num_shards: The approximate number of similarly sized shards to
      construct from `df`. The MovieLens has severe imbalances where some users
      have interacted with many items; this is common among datasets involving
      user data. Rather than attempt to aggressively balance shard size, this
      function simply allows shards to "overflow" which can produce a number of
      shards which is less than `approx_num_shards`. This small degree of
      imbalance does not impact performance; however it does mean that one
      should not expect approx_num_shards to be the ACTUAL number of shards.
    num_items: The cardinality of the item set.
    cache_paths: rconst.Paths object containing locations for various cache
      files.
    match_mlperf: If True, sample eval negative with replacements, which the
      MLPerf reference implementation does.
  """

  num_rows = len(df)
  approximate_partitions = np.linspace(
      0, num_rows, approx_num_shards + 1).astype("int")
  start_ind, end_ind = 0, 0
  shards = []

  for i in range(1, approx_num_shards + 1):
    end_ind = approximate_partitions[i]
    while (end_ind < num_rows and df[movielens.USER_COLUMN][end_ind - 1] ==
           df[movielens.USER_COLUMN][end_ind]):
      end_ind += 1

    if end_ind <= start_ind:
      continue  # imbalance from prior shard.

    df_shard = df[start_ind:end_ind]
    user_shard = df_shard[movielens.USER_COLUMN].values.astype(np.int32)
    item_shard = df_shard[movielens.ITEM_COLUMN].values.astype(np.uint16)

    shards.append({
        movielens.USER_COLUMN: user_shard,
        movielens.ITEM_COLUMN: item_shard,
    })

    start_ind = end_ind
  assert end_ind == num_rows
  approx_num_shards = len(shards)

  tf.logging.info("Splitting train and test data and generating {} test "
                  "negatives per user...".format(rconst.NUM_EVAL_NEGATIVES))
  tf.gfile.MakeDirs(cache_paths.train_shard_subdir)

  # We choose a different random seed for each process, so that the processes
  # will not all choose the same random numbers.
  process_seeds = [np.random.randint(2**32) for _ in range(approx_num_shards)]
  map_args = [(shards[i], i, num_items, cache_paths, process_seeds[i],
               match_mlperf)
              for i in range(approx_num_shards)]
  with contextlib.closing(
      multiprocessing.Pool(multiprocessing.cpu_count())) as pool:
    test_shards = pool.map(_train_eval_map_fn, map_args)  # pylint: disable=no-member

  tf.logging.info("Merging test shards...")
  test_users = np.concatenate([i[movielens.USER_COLUMN] for i in test_shards])
  test_items = np.concatenate([i[movielens.ITEM_COLUMN] for i in test_shards])

  assert test_users.shape == test_items.shape
  assert test_items.shape[0] % (rconst.NUM_EVAL_NEGATIVES + 1) == 0

  test_labels = np.zeros(shape=test_users.shape)
  test_labels[0::(rconst.NUM_EVAL_NEGATIVES + 1)] = 1
  eval_data = ({
      movielens.USER_COLUMN: test_users,
      movielens.ITEM_COLUMN: test_items,
  }, test_labels)

  tf.logging.info("Writing test data to file.")
  tf.gfile.MakeDirs(cache_paths.eval_data_subdir)
  with tf.gfile.Open(cache_paths.eval_raw_file, "wb") as f:
    pickle.dump(eval_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def construct_cache(dataset, data_dir, num_data_readers, match_mlperf,
                    deterministic):
  # type: (str, str, int, bool) -> NCFDataset
  """Load and digest data CSV into a usable form.

  Args:
    dataset: The name of the dataset to be used.
    data_dir: The root directory of the dataset.
    num_data_readers: The number of parallel processes which will request
      data during training.
    match_mlperf: If True, change the behavior of the cache construction to
      match the MLPerf reference implementation.
    deterministic: Try to enforce repeatable behavior, even at the cost of
      performance.
  """
  cache_paths = rconst.Paths(data_dir=data_dir)
  num_data_readers = (num_data_readers or int(multiprocessing.cpu_count() / 2)
                      or 1)
  approx_num_shards = int(movielens.NUM_RATINGS[dataset]
                          // rconst.APPROX_PTS_PER_TRAIN_SHARD) or 1

  st = timeit.default_timer()
  cache_root = os.path.join(data_dir, cache_paths.cache_root)
  if tf.gfile.Exists(cache_root):
    raise ValueError("{} unexpectedly already exists."
                     .format(cache_paths.cache_root))
  tf.logging.info("Creating cache directory. This should be deleted on exit.")
  tf.gfile.MakeDirs(cache_paths.cache_root)

  raw_rating_path = os.path.join(data_dir, dataset, movielens.RATINGS_FILE)
  df, user_map, item_map = _filter_index_sort(raw_rating_path, match_mlperf)
  num_users, num_items = DATASET_TO_NUM_USERS_AND_ITEMS[dataset]

  if num_users != len(user_map):
    raise ValueError("Expected to find {} users, but found {}".format(
        num_users, len(user_map)))
  if num_items != len(item_map):
    raise ValueError("Expected to find {} items, but found {}".format(
        num_items, len(item_map)))

  generate_train_eval_data(df=df, approx_num_shards=approx_num_shards,
                           num_items=len(item_map), cache_paths=cache_paths,
                           match_mlperf=match_mlperf)
  del approx_num_shards  # value may have changed.

  ncf_dataset = NCFDataset(user_map=user_map, item_map=item_map,
                           num_data_readers=num_data_readers,
                           cache_paths=cache_paths,
                           num_train_positives=len(df) - len(user_map),
                           deterministic=deterministic)

  run_time = timeit.default_timer() - st
  tf.logging.info("Cache construction complete. Time: {:.1f} sec."
                  .format(run_time))

  return ncf_dataset


def _shutdown(proc):
  # type: (subprocess.Popen) -> None
  """Convenience function to cleanly shut down async generation process."""

  tf.logging.info("Shutting down train data creation subprocess.")
  try:
    proc.send_signal(signal.SIGINT)
    time.sleep(1)
    if proc.returncode is not None:
      return  # SIGINT was handled successfully within 1 sec

  except socket.error:
    pass

  # Otherwise another second of grace period and then forcibly kill the process.
  time.sleep(1)
  proc.terminate()


def instantiate_pipeline(dataset, data_dir, batch_size, eval_batch_size,
                         num_data_readers=None, num_neg=4, epochs_per_cycle=1,
                         match_mlperf=False, deterministic=False):
  # type: (...) -> (NCFDataset, typing.Callable)
  """Preprocess data and start negative generation subprocess."""

  tf.logging.info("Beginning data preprocessing.")
  ncf_dataset = construct_cache(dataset=dataset, data_dir=data_dir,
                                num_data_readers=num_data_readers,
                                match_mlperf=match_mlperf,
                                deterministic=deterministic)

  tf.logging.info("Creating training file subprocess.")

  subproc_env = os.environ.copy()

  # The subprocess uses TensorFlow for tf.gfile, but it does not need GPU
  # resources and by default will try to allocate GPU memory. This would cause
  # contention with the main training process.
  subproc_env["CUDA_VISIBLE_DEVICES"] = ""

  # By limiting the number of workers we guarantee that the worker
  # pool underlying the training generation doesn't starve other processes.
  num_workers = int(multiprocessing.cpu_count() * 0.75) or 1

  subproc_args = popen_helper.INVOCATION + [
      "--data_dir", data_dir,
      "--cache_id", str(ncf_dataset.cache_paths.cache_id),
      "--num_neg", str(num_neg),
      "--num_train_positives", str(ncf_dataset.num_train_positives),
      "--num_items", str(ncf_dataset.num_items),
      "--num_readers", str(ncf_dataset.num_data_readers),
      "--epochs_per_cycle", str(epochs_per_cycle),
      "--train_batch_size", str(batch_size),
      "--eval_batch_size", str(eval_batch_size),
      "--num_workers", str(num_workers),
      "--spillover", "True",  # This allows the training input function to
                              # guarantee batch size and significantly improves
                              # performance. (~5% increase in examples/sec on
                              # GPU, and needed for TPU XLA.)
      "--redirect_logs", "True"
  ]
  if ncf_dataset.deterministic:
    subproc_args.extend(["--seed", str(int(stat_utils.random_int32()))])

  tf.logging.info(
      "Generation subprocess command: {}".format(" ".join(subproc_args)))

  proc = subprocess.Popen(args=subproc_args, shell=False, env=subproc_env)

  cleanup_called = {"finished": False}
  @atexit.register
  def cleanup():
    """Remove files and subprocess from data generation."""
    if cleanup_called["finished"]:
      return

    _shutdown(proc)
    try:
      tf.gfile.DeleteRecursively(ncf_dataset.cache_paths.cache_root)
    except tf.errors.NotFoundError:
      pass

    cleanup_called["finished"] = True

  for _ in range(300):
    if tf.gfile.Exists(ncf_dataset.cache_paths.subproc_alive):
      break
    time.sleep(1)  # allow `alive` file to be written
  if not tf.gfile.Exists(ncf_dataset.cache_paths.subproc_alive):
    raise ValueError("Generation subprocess did not start correctly. Data will "
                     "not be available; exiting to avoid waiting forever.")

  return ncf_dataset, cleanup


def make_deserialize(params, batch_size, training=False):
  """Construct deserialize function for training and eval fns."""
  feature_map = {
      movielens.USER_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
      movielens.ITEM_COLUMN: tf.FixedLenFeature([], dtype=tf.string),
  }
  if training:
    feature_map["labels"] = tf.FixedLenFeature([], dtype=tf.string)
  else:
    feature_map[rconst.DUPLICATE_MASK] = tf.FixedLenFeature([], dtype=tf.string)

  def deserialize(examples_serialized):
    """Called by Dataset.map() to convert batches of records to tensors."""
    features = tf.parse_single_example(examples_serialized, feature_map)
    users = tf.reshape(tf.decode_raw(
        features[movielens.USER_COLUMN], tf.int32), (batch_size,))
    items = tf.reshape(tf.decode_raw(
        features[movielens.ITEM_COLUMN], tf.uint16), (batch_size,))

    if params["use_tpu"]:
      items = tf.cast(items, tf.int32)  # TPU doesn't allow uint16 infeed.

    if not training:
      dupe_mask = tf.reshape(tf.cast(tf.decode_raw(
          features[rconst.DUPLICATE_MASK], tf.int8), tf.bool), (batch_size,))
      return {
          movielens.USER_COLUMN: users,
          movielens.ITEM_COLUMN: items,
          rconst.DUPLICATE_MASK: dupe_mask,
      }

    labels = tf.reshape(tf.cast(tf.decode_raw(
        features["labels"], tf.int8), tf.bool), (batch_size,))

    return {
        movielens.USER_COLUMN: users,
        movielens.ITEM_COLUMN: items,
    }, labels
  return deserialize


def hash_pipeline(dataset, deterministic):
  # type: (tf.data.Dataset, bool) -> None
  """Utility function for detecting non-determinism in the data pipeline.

  Args:
    dataset: a tf.data.Dataset generated by the input_fn
    deterministic: Does the input_fn expect the dataset to be deterministic.
      (i.e. fixed seed, sloppy=False, etc.)
  """
  if not deterministic:
    tf.logging.warning("Data pipeline is not marked as deterministic. Hash "
                       "values are not expected to be meaningful.")

  batch = dataset.make_one_shot_iterator().get_next()
  md5 = hashlib.md5()
  count = 0
  first_batch_hash = b""
  with tf.Session() as sess:
    while True:
      try:
        result = sess.run(batch)
        if isinstance(result, tuple):
          result = result[0]  # only hash features
      except tf.errors.OutOfRangeError:
        break

      count += 1
      md5.update(memoryview(result[movielens.USER_COLUMN]).tobytes())
      md5.update(memoryview(result[movielens.ITEM_COLUMN]).tobytes())
      if count == 1:
        first_batch_hash = md5.hexdigest()
  overall_hash = md5.hexdigest()
  tf.logging.info("Batch count: {}".format(count))
  tf.logging.info("  [pipeline_hash] First batch hash: {}".format(
      first_batch_hash))
  tf.logging.info("  [pipeline_hash] All batches hash: {}".format(overall_hash))


def make_train_input_fn(ncf_dataset):
  # type: (typing.Optional[NCFDataset]) -> (typing.Callable, str, int)
  """Construct training input_fn for the current epoch."""

  if ncf_dataset is None:
    return make_train_synthetic_input_fn()

  if not tf.gfile.Exists(ncf_dataset.cache_paths.subproc_alive):
    # The generation subprocess must have been alive at some point, because we
    # earlier checked that the subproc_alive file existed.
    raise ValueError("Generation subprocess unexpectedly died. Data will not "
                     "be available; exiting to avoid waiting forever.")

  train_epoch_dir = ncf_dataset.cache_paths.train_epoch_dir
  while not tf.gfile.Exists(train_epoch_dir):
    tf.logging.info("Waiting for {} to exist.".format(train_epoch_dir))
    time.sleep(1)

  train_data_dirs = tf.gfile.ListDirectory(train_epoch_dir)
  while not train_data_dirs:
    tf.logging.info("Waiting for data folder to be created.")
    time.sleep(1)
    train_data_dirs = tf.gfile.ListDirectory(train_epoch_dir)
  train_data_dirs.sort()  # names are zfilled so that
                          # lexicographic sort == numeric sort
  record_dir = os.path.join(train_epoch_dir, train_data_dirs[0])

  ready_file = os.path.join(record_dir, rconst.READY_FILE)
  while not tf.gfile.Exists(ready_file):
    tf.logging.info("Waiting for records in {} to be ready".format(record_dir))
    time.sleep(1)

  with tf.gfile.Open(ready_file, "r") as f:
    epoch_metadata = json.load(f)

  # The data pipeline uses spillover to guarantee static batch sizes. This
  # means that an extra batch will need to be run every few epochs. TPUs
  # require that the number of batches to be run is known at the time that
  # estimator.train() is called, so having the generation pipeline report
  # number of batches guarantees that this count is correct.
  batch_count = epoch_metadata["batch_count"]

  def input_fn(params):
    """Generated input_fn for the given epoch."""
    batch_size = params["batch_size"]

    if epoch_metadata["batch_size"] != batch_size:
      raise ValueError(
          "Records were constructed with batch size {}, but input_fn was given "
          "a batch size of {}. This will result in a deserialization error in "
          "tf.parse_single_example."
          .format(epoch_metadata["batch_size"], batch_size))

    record_files = tf.data.Dataset.list_files(
        os.path.join(record_dir, rconst.TRAIN_RECORD_TEMPLATE.format("*")),
        shuffle=False)

    interleave = tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=4,
        block_length=100000,
        sloppy=not ncf_dataset.deterministic,
        prefetch_input_elements=4,
    )

    deserialize = make_deserialize(params, batch_size, True)
    dataset = record_files.apply(interleave)
    dataset = dataset.map(deserialize, num_parallel_calls=4)
    dataset = dataset.prefetch(32)

    if params.get("hash_pipeline"):
      hash_pipeline(dataset, ncf_dataset.deterministic)

    return dataset

  return input_fn, record_dir, batch_count


def make_train_synthetic_input_fn():
  """Construct training input_fn that uses synthetic data."""
  def input_fn(params):
    """Generated input_fn for the given epoch."""
    batch_size = params["batch_size"]
    num_users = params["num_users"]
    num_items = params["num_items"]

    users = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                              maxval=num_users)
    items = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                              maxval=num_items)
    labels = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                               maxval=2)

    data = {
        movielens.USER_COLUMN: users,
        movielens.ITEM_COLUMN: items,
    }, labels
    dataset = tf.data.Dataset.from_tensors(data).repeat(
        _SYNTHETIC_BATCHES_PER_EPOCH)
    dataset = dataset.prefetch(32)
    return dataset

  return input_fn, None, _SYNTHETIC_BATCHES_PER_EPOCH


def make_pred_input_fn(ncf_dataset):
  # type: (typing.Optional[NCFDataset]) -> typing.Callable
  """Construct input_fn for metric evaluation."""

  if ncf_dataset is None:
    return make_synthetic_pred_input_fn()

  def input_fn(params):
    """Input function based on eval batch size."""

    # Estimator has "eval_batch_size" included in the params, but TPUEstimator
    # populates "batch_size" to the appropriate value.
    batch_size = params.get("eval_batch_size") or params["batch_size"]
    record_file = ncf_dataset.cache_paths.eval_record_template.format(
        batch_size)
    while not tf.gfile.Exists(record_file):
      tf.logging.info(
          "Waiting for eval data to be written to {}".format(record_file))
      time.sleep(1)
    dataset = tf.data.TFRecordDataset(record_file)

    deserialize = make_deserialize(params, batch_size, False)
    dataset = dataset.map(deserialize, num_parallel_calls=4)
    dataset = dataset.prefetch(16)

    if params.get("hash_pipeline"):
      hash_pipeline(dataset, ncf_dataset.deterministic)

    return dataset

  return input_fn


def make_synthetic_pred_input_fn():
  """Construct input_fn for metric evaluation that uses synthetic data."""

  def input_fn(params):
    """Generated input_fn for the given epoch."""
    batch_size = params["eval_batch_size"]
    num_users = params["num_users"]
    num_items = params["num_items"]

    users = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                              maxval=num_users)
    items = tf.random_uniform([batch_size], dtype=tf.int32, minval=0,
                              maxval=num_items)
    dupe_mask = tf.cast(tf.random_uniform([batch_size], dtype=tf.int32,
                                          minval=0, maxval=2), tf.bool)

    data = {
        movielens.USER_COLUMN: users,
        movielens.ITEM_COLUMN: items,
        rconst.DUPLICATE_MASK: dupe_mask,
    }
    dataset = tf.data.Dataset.from_tensors(data).repeat(
        _SYNTHETIC_BATCHES_PER_EPOCH)
    dataset = dataset.prefetch(16)
    return dataset

  return input_fn
