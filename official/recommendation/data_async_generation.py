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
"""Asynchronously generate TFRecords files for NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import contextlib
import datetime
import gc
import multiprocessing
import json
import os
import pickle
import signal
import sys
import tempfile
import time
import timeit
import traceback
import typing

import numpy as np
import tensorflow as tf

from absl import app as absl_app
from absl import flags

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import stat_utils


_log_file = None


def log_msg(msg):
  """Include timestamp info when logging messages to a file."""
  if flags.FLAGS.redirect_logs:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print("[{}] {}".format(timestamp, msg), file=_log_file)
  else:
    print(msg, file=_log_file)
  if _log_file:
    _log_file.flush()


def get_cycle_folder_name(i):
  return "cycle_{}".format(str(i).zfill(5))


def _process_shard(args):
  # type: ((str, int, int, int)) -> (np.ndarray, np.ndarray, np.ndarray)
  """Read a shard of training data and return training vectors.

  Args:
    shard_path: The filepath of the positive instance training shard.
    num_items: The cardinality of the item set.
    num_neg: The number of negatives to generate per positive example.
    seed: Random seed to be used when generating negatives.
  """
  shard_path, num_items, num_neg, seed = args
  np.random.seed(seed)

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
    positive_items = items[boundaries[i]:boundaries[i+1]]
    positive_set = set(positive_items)
    if positive_items.shape[0] != len(positive_set):
      raise ValueError("Duplicate entries detected.")
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


def _construct_record(users, items, labels=None):
  """Convert NumPy arrays into a TFRecords entry."""
  feature_dict = {
      movielens.USER_COLUMN: tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[memoryview(users).tobytes()])),
      movielens.ITEM_COLUMN: tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[memoryview(items).tobytes()])),
  }
  if labels is not None:
    feature_dict["labels"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[memoryview(labels).tobytes()]))

  return tf.train.Example(
      features=tf.train.Features(feature=feature_dict)).SerializeToString()


def sigint_handler(signal, frame):
  log_msg("Shutting down worker.")


def init_worker():
  signal.signal(signal.SIGINT, sigint_handler)


def _construct_training_records(
    train_cycle,          # type: int
    num_workers,          # type: int
    cache_paths,          # type: rconst.Paths
    num_readers,          # type: int
    num_neg,              # type: int
    num_train_positives,  # type: int
    num_items,            # type: int
    epochs_per_cycle,     # type: int
    train_batch_size,     # type: int
    training_shards,      # type: typing.List[str]
    spillover,            # type: bool
    carryover=None,       # type: typing.Union[typing.List[np.ndarray], None]
    deterministic=False   # type: bool
    ):
  """Generate false negatives and write TFRecords files.

  Args:
    train_cycle: Integer of which cycle the generated data is for.
    num_workers: Number of multiprocessing workers to use for negative
      generation.
    cache_paths: Paths object with information of where to write files.
    num_readers: The number of reader datasets in the train input_fn.
    num_neg: The number of false negatives per positive example.
    num_train_positives: The number of positive examples. This value is used
      to pre-allocate arrays while the imap is still running. (NumPy does not
      allow dynamic arrays.)
    num_items: The cardinality of the item set.
    epochs_per_cycle: The number of epochs worth of data to construct.
    train_batch_size: The expected batch size used during training. This is used
      to properly batch data when writing TFRecords.
    training_shards: The picked positive examples from which to generate
      negatives.
    spillover: If the final batch is incomplete, push it to the next
      cycle (True) or include a partial batch (False).
    carryover: The data points to be spilled over to the next cycle.
  """

  st = timeit.default_timer()
  num_workers = min([num_workers, len(training_shards) * epochs_per_cycle])
  carryover = carryover or [
      np.zeros((0,), dtype=np.int32),
      np.zeros((0,), dtype=np.uint16),
      np.zeros((0,), dtype=np.int8),
  ]
  num_carryover = carryover[0].shape[0]
  num_pts = num_carryover + num_train_positives * (1 + num_neg)

  # We choose a different random seed for each process, so that the processes
  # will not all choose the same random numbers.
  process_seeds = [np.random.randint(2**32)
                   for _ in training_shards * epochs_per_cycle]
  map_args = [(shard, num_items, num_neg, process_seeds[i])
              for i, shard in enumerate(training_shards * epochs_per_cycle)]

  with contextlib.closing(multiprocessing.Pool(
      processes=num_workers, initializer=init_worker)) as pool:
    map_fn = pool.imap if deterministic else pool.imap_unordered  # pylint: disable=no-member
    data_generator = map_fn(_process_shard, map_args)
    data = [
        np.zeros(shape=(num_pts,), dtype=np.int32) - 1,
        np.zeros(shape=(num_pts,), dtype=np.uint16),
        np.zeros(shape=(num_pts,), dtype=np.int8),
    ]

    # The carryover data is always first.
    for i in range(3):
      data[i][:num_carryover] = carryover[i]
    index_destinations = np.random.permutation(
        num_train_positives * (1 + num_neg)) + num_carryover
    start_ind = 0
    for data_segment in data_generator:
      n_in_segment = data_segment[0].shape[0]
      dest = index_destinations[start_ind:start_ind + n_in_segment]
      start_ind += n_in_segment
      for i in range(3):
        data[i][dest] = data_segment[i]

  # Check that no points were dropped.
  assert (num_pts - num_carryover) == start_ind
  assert not np.sum(data[0] == -1)

  record_dir = os.path.join(cache_paths.train_epoch_dir,
                            get_cycle_folder_name(train_cycle))
  tf.gfile.MakeDirs(record_dir)

  batches_per_file = np.ceil(num_pts / train_batch_size / num_readers)
  current_file_id = -1
  current_batch_id = -1
  batches_by_file = [[] for _ in range(num_readers)]

  output_carryover = [
      np.zeros(shape=(0,), dtype=np.int32),
      np.zeros(shape=(0,), dtype=np.uint16),
      np.zeros(shape=(0,), dtype=np.int8),
  ]

  while True:
    current_batch_id += 1
    if (current_batch_id % batches_per_file) == 0:
      current_file_id += 1
    end_ind = (current_batch_id + 1) * train_batch_size
    if end_ind > num_pts:
      if spillover:
        output_carryover = [data[i][current_batch_id*train_batch_size:num_pts]
                            for i in range(3)]
        break
      else:
        batches_by_file[current_file_id].append(current_batch_id)
        break
    batches_by_file[current_file_id].append(current_batch_id)

  batch_count = 0
  for i in range(num_readers):
    fpath = os.path.join(record_dir, rconst.TRAIN_RECORD_TEMPLATE.format(i))
    log_msg("Writing {}".format(fpath))
    with tf.python_io.TFRecordWriter(fpath) as writer:
      for j in batches_by_file[i]:
        start_ind = j * train_batch_size
        end_ind = start_ind + train_batch_size
        batch_bytes = _construct_record(
            users=data[0][start_ind:end_ind],
            items=data[1][start_ind:end_ind],
            labels=data[2][start_ind:end_ind],
        )

        writer.write(batch_bytes)
        batch_count += 1


  if spillover:
    written_pts = output_carryover[0].shape[0] + batch_count * train_batch_size
    if num_pts != written_pts:
      raise ValueError("Error detected: point counts do not match: {} vs. {}"
                       .format(num_pts, written_pts))

  # We write to a temp file then atomically rename it to the final file, because
  # writing directly to the final file can cause the main process to read a
  # partially written JSON file.
  ready_file_temp = os.path.join(record_dir, rconst.READY_FILE_TEMP)
  with tf.gfile.Open(ready_file_temp, "w") as f:
    json.dump({
        "batch_size": train_batch_size,
        "batch_count": batch_count,
    }, f)
  ready_file = os.path.join(record_dir, rconst.READY_FILE)
  tf.gfile.Rename(ready_file_temp, ready_file)

  log_msg("Cycle {} complete. Total time: {:.1f} seconds"
          .format(train_cycle, timeit.default_timer() - st))

  return output_carryover


def _construct_eval_record(cache_paths, eval_batch_size):
  """Convert Eval data to a single TFRecords file."""

  log_msg("Beginning construction of eval TFRecords file.")
  raw_fpath = cache_paths.eval_raw_file
  intermediate_fpath = cache_paths.eval_record_template_temp
  dest_fpath = cache_paths.eval_record_template.format(eval_batch_size)
  with tf.gfile.Open(raw_fpath, "rb") as f:
    eval_data = pickle.load(f)

  users = eval_data[0][movielens.USER_COLUMN]
  items = eval_data[0][movielens.ITEM_COLUMN]
  assert users.shape == items.shape
  # eval_data[1] is the labels, but during evaluation they are infered as they
  # have a set structure. They are included the the data artifact for debug
  # purposes.

  # This packaging assumes that the caller knows to drop the padded values.
  n_pts = users.shape[0]
  n_pad = eval_batch_size - (n_pts % eval_batch_size)
  assert not (n_pts + n_pad) % eval_batch_size

  users = np.concatenate([users, np.zeros(shape=(n_pad,), dtype=np.int32)])\
    .reshape((-1, eval_batch_size))
  items = np.concatenate([items, np.zeros(shape=(n_pad,), dtype=np.uint16)])\
    .reshape((-1, eval_batch_size))

  num_batches = users.shape[0]
  with tf.python_io.TFRecordWriter(intermediate_fpath) as writer:
    for i in range(num_batches):
      batch_bytes = _construct_record(
          users=users[i, :],
          items=items[i, :]
      )
      writer.write(batch_bytes)
  tf.gfile.Rename(intermediate_fpath, dest_fpath)
  log_msg("Eval TFRecords file successfully constructed.")


def _generation_loop(num_workers,           # type: int
                     cache_paths,           # type: rconst.Paths
                     num_readers,           # type: int
                     num_neg,               # type: int
                     num_train_positives,   # type: int
                     num_items,             # type: int
                     spillover,             # type: bool
                     epochs_per_cycle,      # type: int
                     train_batch_size,      # type: int
                     eval_batch_size,       # type: int
                     deterministic          # type: bool
                    ):
  # type: (...) -> None
  """Primary run loop for data file generation."""

  log_msg("Signaling that I am alive.")
  with tf.gfile.Open(cache_paths.subproc_alive, "w") as f:
    f.write("Generation subproc has started.")

  @atexit.register
  def remove_alive_file():
    try:
      tf.gfile.Remove(cache_paths.subproc_alive)
    except tf.errors.NotFoundError:
      return  # Main thread has already deleted the entire cache dir.

  log_msg("Entering generation loop.")
  tf.gfile.MakeDirs(cache_paths.train_epoch_dir)

  training_shards = [os.path.join(cache_paths.train_shard_subdir, i) for i in
                     tf.gfile.ListDirectory(cache_paths.train_shard_subdir)]

  # Training blocks on the creation of the first epoch, so the num_workers
  # limit is not respected for this invocation
  train_cycle = 0
  carryover = _construct_training_records(
      train_cycle=train_cycle, num_workers=multiprocessing.cpu_count(),
      cache_paths=cache_paths, num_readers=num_readers, num_neg=num_neg,
      num_train_positives=num_train_positives, num_items=num_items,
      epochs_per_cycle=epochs_per_cycle, train_batch_size=train_batch_size,
      training_shards=training_shards, spillover=spillover, carryover=None,
      deterministic=deterministic)

  _construct_eval_record(cache_paths=cache_paths,
                         eval_batch_size=eval_batch_size)

  wait_count = 0
  start_time = time.time()
  while True:
    ready_epochs = tf.gfile.ListDirectory(cache_paths.train_epoch_dir)
    if len(ready_epochs) >= rconst.CYCLES_TO_BUFFER:
      wait_count += 1
      sleep_time = max([0, wait_count * 5 - (time.time() - start_time)])
      time.sleep(sleep_time)

      if (wait_count % 10) == 0:
        log_msg("Waited {} times for data to be consumed."
                .format(wait_count))

      if time.time() - start_time > rconst.TIMEOUT_SECONDS:
        log_msg("Waited more than {} seconds. Concluding that this "
                "process is orphaned and exiting gracefully."
                .format(rconst.TIMEOUT_SECONDS))
        sys.exit()

      continue

    train_cycle += 1
    carryover = _construct_training_records(
        train_cycle=train_cycle, num_workers=num_workers,
        cache_paths=cache_paths, num_readers=num_readers, num_neg=num_neg,
        num_train_positives=num_train_positives, num_items=num_items,
        epochs_per_cycle=epochs_per_cycle, train_batch_size=train_batch_size,
        training_shards=training_shards, spillover=spillover,
        carryover=carryover, deterministic=deterministic)

    wait_count = 0
    start_time = time.time()
    gc.collect()


def main(_):
  global _log_file
  redirect_logs = flags.FLAGS.redirect_logs
  cache_paths = rconst.Paths(
      data_dir=flags.FLAGS.data_dir, cache_id=flags.FLAGS.cache_id)


  log_file_name = "data_gen_proc_{}.log".format(cache_paths.cache_id)
  log_path = os.path.join(cache_paths.data_dir, log_file_name)
  if log_path.startswith("gs://") and redirect_logs:
    fallback_log_file = os.path.join(tempfile.gettempdir(), log_file_name)
    print("Unable to log to {}. Falling back to {}"
          .format(log_path, fallback_log_file))
    log_path = fallback_log_file

  # This server is generally run in a subprocess.
  if redirect_logs:
    print("Redirecting output of data_async_generation.py process to {}"
          .format(log_path))
    _log_file = open(log_path, "wt")  # Note: not tf.gfile.Open().
  try:
    log_msg("sys.argv: {}".format(" ".join(sys.argv)))

    if flags.FLAGS.seed is not None:
      np.random.seed(flags.FLAGS.seed)

    _generation_loop(
        num_workers=flags.FLAGS.num_workers,
        cache_paths=cache_paths,
        num_readers=flags.FLAGS.num_readers,
        num_neg=flags.FLAGS.num_neg,
        num_train_positives=flags.FLAGS.num_train_positives,
        num_items=flags.FLAGS.num_items,
        spillover=flags.FLAGS.spillover,
        epochs_per_cycle=flags.FLAGS.epochs_per_cycle,
        train_batch_size=flags.FLAGS.train_batch_size,
        eval_batch_size=flags.FLAGS.eval_batch_size,
        deterministic=flags.FLAGS.seed is not None,
    )
  except KeyboardInterrupt:
    log_msg("KeyboardInterrupt registered.")
  except:
    traceback.print_exc(file=_log_file)
    raise
  finally:
    log_msg("Shutting down generation subprocess.")
    sys.stdout.flush()
    sys.stderr.flush()
    if redirect_logs:
      _log_file.close()


def define_flags():
  """Construct flags for the server.

  This function does not use offical.utils.flags, as these flags are not meant
  to be used by humans. Rather, they should be passed as part of a subprocess
  call.
  """
  flags.DEFINE_integer(name="num_workers", default=multiprocessing.cpu_count(),
                       help="Size of the negative generation worker pool.")
  flags.DEFINE_string(name="data_dir", default=None,
                      help="The data root. (used to construct cache paths.)")
  flags.DEFINE_string(name="cache_id", default=None,
                      help="The cache_id generated in the main process.")
  flags.DEFINE_integer(name="num_readers", default=4,
                       help="Number of reader datasets in training. This sets"
                            "how the epoch files are sharded.")
  flags.DEFINE_integer(name="num_neg", default=None,
                       help="The Number of negative instances to pair with a "
                            "positive instance.")
  flags.DEFINE_integer(name="num_train_positives", default=None,
                       help="The number of positive training examples.")
  flags.DEFINE_integer(name="num_items", default=None,
                       help="Number of items from which to select negatives.")
  flags.DEFINE_integer(name="epochs_per_cycle", default=1,
                       help="The number of epochs of training data to produce"
                            "at a time.")
  flags.DEFINE_integer(name="train_batch_size", default=None,
                       help="The batch size with which training TFRecords will "
                            "be chunked.")
  flags.DEFINE_integer(name="eval_batch_size", default=None,
                       help="The batch size with which evaluation TFRecords "
                            "will be chunked.")
  flags.DEFINE_boolean(
      name="spillover", default=True,
      help="If a complete batch cannot be provided, return an empty batch and "
           "start the next epoch from a non-empty buffer. This guarantees "
           "fixed batch sizes.")
  flags.DEFINE_boolean(name="redirect_logs", default=False,
                       help="Catch logs and write them to a file. "
                            "(Useful if this is run as a subprocess)")
  flags.DEFINE_integer(name="seed", default=None,
                       help="NumPy random seed to set at startup. If not "
                            "specified, a seed will not be set.")

  flags.mark_flags_as_required(
      ["data_dir", "cache_id", "num_neg", "num_train_positives", "num_items",
       "train_batch_size", "eval_batch_size"])



if __name__ == "__main__":
  define_flags()
  absl_app.run(main)
