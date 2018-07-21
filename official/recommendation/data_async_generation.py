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
import functools
import logging
import multiprocessing
import os
import pickle
import signal
import sys
import time
import timeit
import traceback

import numpy as np
import tensorflow as tf

from absl import app as absl_app
from absl import logging as absl_logging
from absl import flags

from official.datasets import movielens
from official.recommendation import stat_utils


_CYCLES_TO_BUFFER = 3
READY_FILE = "ready"
RECORD_FILE_PREFIX = "training_records_"


def get_cycle_folder_name(i):
  return "cycle_{}".format(str(i).zfill(5))


def _process_shard(shard_path, num_items, num_neg):
  # type: (str, int, int) -> (np.ndarray, np.ndarray, np.ndarray)
  """Read a shard of training data and return training vectors.

  Args:
    shard_path: The filepath of the positive instance training shard.
    num_items: The cardinality of the item set.
    num_neg: The number of negatives to generate per positive example.
  """

  # The choice to store the training shards in files rather than in memory
  # is motivated by the fact that multiprocessing serializes arguments,
  # transmits them to map workers, and then deserializes them. By storing the
  # training shards in files, the serialization work only needs to be done once.
  #
  # A similar effect could be achieved by simply holding pickled bytes in
  # memory, however the processing is not I/O bound and is therefore
  # unnecessary.
  with tf.gfile.Open(shard_path, "rb") as f:
    shard = pickle.load(f)

  users = shard[movielens.USER_COLUMN]
  items = shard[movielens.ITEM_COLUMN]

  delta = users[1:] - users[:-1]
  boundaries = ([0] + (np.argwhere(delta)[:, 0] + 1).tolist() +
                [users.shape[0]])

  user_blocks = []
  item_blocks = []
  label_blocks = []
  for i in range(len(boundaries) - 1):
    assert len(set(users[boundaries[i]:boundaries[i+1]])) == 1
    positive_set = set(items[boundaries[i]:boundaries[i+1]])
    n_pos = len(positive_set)

    negatives = stat_utils.sample_with_exclusion(
        num_items, positive_set, n_pos * num_neg)

    user_blocks.append(users[boundaries[i]] * np.ones(
        (n_pos * (1 + num_neg),), dtype=np.int32))
    item_blocks.append(
        np.array(list(positive_set) + negatives, dtype=np.uint16))
    labels_for_user = np.zeros((n_pos * (1 + num_neg),), dtype=np.int8)
    labels_for_user[:n_pos] = 1
    label_blocks.append(labels_for_user)

  users_out = np.concatenate(user_blocks)
  items_out = np.concatenate(item_blocks)
  labels_out = np.concatenate(label_blocks)

  assert users_out.shape == items_out.shape == labels_out.shape
  return users_out, items_out, labels_out


def sigint_handler(signal, frame):
  absl_logging.info("Shutting down worker.")


def init_worker():
  signal.signal(signal.SIGINT, sigint_handler)


def generation_loop(num_workers, shard_dir, output_root, num_readers, num_neg,
                    num_items, spillover, epochs_per_cycle, batch_size):
  training_shards = [os.path.join(shard_dir, i) for i in tf.gfile.ListDirectory(shard_dir)]
  training_shards = [i for i in training_shards
                     if not tf.gfile.IsDirectory(i)]

  map_args = [i for i in training_shards * epochs_per_cycle]
  map_fn = functools.partial(_process_shard, num_neg=num_neg,
                             num_items=num_items)

  absl_logging.info("Entering generation loop.")
  tf.gfile.MakeDirs(output_root)

  with contextlib.closing(multiprocessing.Pool(
      processes=num_workers, initializer=init_worker)) as pool:
    empty = (
      np.zeros((0,), dtype=np.int32),
      np.zeros((0,), dtype=np.uint16),
      np.zeros((0,), dtype=np.int8),
    )
    old_results = empty

    cycle_number = 0
    while True:
      cycle_number += 1
      gen_wait_counter = 0
      while len(tf.gfile.ListDirectory(output_root)) > _CYCLES_TO_BUFFER:
        time.sleep(1)
        gen_wait_counter += 1
        if gen_wait_counter ** 0.5 == int(gen_wait_counter ** 0.5):
          absl_logging.info("Waiting for train loop to consume data. Waited {} "
                            "times.".format(gen_wait_counter))

      absl_logging.info("Beginning cycle {}".format(cycle_number))
      st = timeit.default_timer()

      residual = old_results[0].shape[0]
      results = [old_results] + pool.map(map_fn, map_args)
      result_arrays = [
        np.concatenate([j[i] for j in results]) for i in range(3)
      ]

      n = result_arrays[0].shape[0]
      shuffle_indices = np.random.permutation(n - residual) + residual
      for i in range(3):
        result_arrays[i][residual:] = result_arrays[i][shuffle_indices]

      batches = []
      for i in range(int(np.ceil(n / batch_size))):
        batches.append((
          result_arrays[0][i * batch_size: (i + 1) * batch_size],
          result_arrays[1][i * batch_size: (i + 1) * batch_size],
          result_arrays[2][i * batch_size: (i + 1) * batch_size]
        ))

      old_results = empty
      if batches[-1][0].shape[0] < batch_size and spillover:
        old_results = batches.pop()

      records = [
        tf.train.Example(features=tf.train.Features(feature={
          movielens.USER_COLUMN: tf.train.Feature(int64_list=tf.train.Int64List(value=i[0].astype(np.int64))),
          movielens.ITEM_COLUMN: tf.train.Feature(int64_list=tf.train.Int64List(value=i[1].astype(np.int64))),
          "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=i[2].astype(np.int64))),
        })).SerializeToString() for i in batches
      ]

      record_shards = [[] for i in range(num_readers)]
      for i, record_bytes in enumerate(records):
        record_shards[i % num_readers].append(record_bytes)

      record_dir = os.path.join(output_root, get_cycle_folder_name(cycle_number))
      tf.gfile.MakeDirs(record_dir)
      for i, shard_byte_list in enumerate(record_shards):
        fpath = os.path.join(record_dir, RECORD_FILE_PREFIX + str(i).zfill(5))
        absl_logging.info("Writing {}".format(fpath))
        with tf.python_io.TFRecordWriter(fpath) as writer:
          for example in shard_byte_list:
            writer.write(example)

      with tf.gfile.Open(os.path.join(record_dir, READY_FILE), "wb") as f:
        f.write(b"")

      absl_logging.info("Cycle {} complete. Total time: {:.1f} seconds"
                        .format(cycle_number, timeit.default_timer() - st))


def main(_):
  num_workers = flags.FLAGS.num_workers
  shard_dir = flags.FLAGS.shard_dir
  output_root = flags.FLAGS.output_root
  num_readers = flags.FLAGS.num_readers
  num_neg = flags.FLAGS.num_neg
  num_items = flags.FLAGS.num_items
  spillover = flags.FLAGS.spillover
  epochs_per_cycle = flags.FLAGS.epochs_per_cycle
  batch_size = flags.FLAGS.batch_size
  log_dir = os.path.join(shard_dir, "logs")
  tf.gfile.MakeDirs(log_dir)

  # This server is generally run in a subprocess.
  print("Redirecting stdout and stderr to files in {}".format(log_dir))
  stdout = open(os.path.join(log_dir, "stdout.log"), "wt")
  stderr = open(os.path.join(log_dir, "stderr.log"), "wt")
  try:
    absl_logging.get_absl_logger().addHandler(hdlr=logging.StreamHandler(stream=stdout))
    sys.stdout = stdout
    sys.stderr = stderr
    print("Logs redirected.")
    try:
      generation_loop(
          num_workers=num_workers,
          shard_dir=shard_dir,
          output_root=output_root,
          num_readers=num_readers,
          num_neg=num_neg,
          num_items=num_items,
          spillover=spillover,
          epochs_per_cycle=epochs_per_cycle,
          batch_size=batch_size,
      )
    except:
      traceback.print_exc()
      raise
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout.close()
    stderr.close()


def define_flags():
  """Construct flags for the server.

  This function does not use offical.utils.flags, as these flags are not meant
  to be used by humans. Rather, they should be passed as part of a subprocess
  call.
  """
  flags.DEFINE_integer(name="num_workers", default=multiprocessing.cpu_count(),
                       help="Size of the negative generation worker pool.")
  flags.DEFINE_string(name="shard_dir", default=None,
                      help="Location of the sharded test positives.")
  flags.DEFINE_string(name="output_root", default=None,
                      help="The root directory where training data shards "
                           "should be written.")
  flags.DEFINE_integer(name="num_readers", default=4,
                      help="Number of reader datasets in training. This sets"
                           "how the epoch files are sharded.")
  flags.DEFINE_integer(name="num_neg", default=None,
                       help="The Number of negative instances to pair with a "
                            "positive instance.")
  flags.DEFINE_integer(name="num_items", default=None,
                       help="Number of items from which to select negatives.")
  flags.DEFINE_integer(name="epochs_per_cycle", default=1,
                       help="The number of epochs of training data to produce"
                            "at a time.")
  flags.DEFINE_integer(name="batch_size", default=None,
                       help="The batch size with which TFRecords will be chunked.")
  flags.DEFINE_boolean(
      name="spillover", default=True,
      help="If a complete batch cannot be provided, return an empty batch and "
           "start the next epoch from a non-empty buffer. This guarantees "
           "fixed batch sizes.")

  flags.mark_flags_as_required(
      ["shard_dir", "output_root", "num_neg", "num_items", "batch_size"])



if __name__ == "__main__":
  define_flags()
  absl_app.run(main)
