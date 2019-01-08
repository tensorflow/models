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
from official.recommendation import popen_helper
from official.utils.logs import mlperf_helper


_log_file = None


def log_msg(msg):
  """Include timestamp info when logging messages to a file."""
  if flags.FLAGS.use_tf_logging:
    tf.logging.info(msg)
    return

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
  # type: ((str, int, int, int, bool)) -> (np.ndarray, np.ndarray, np.ndarray)
  """Read a shard of training data and return training vectors.

  Args:
    shard_path: The filepath of the positive instance training shard.
    num_items: The cardinality of the item set.
    num_neg: The number of negatives to generate per positive example.
    seed: Random seed to be used when generating negatives.
    is_training: Generate training (True) or eval (False) data.
    match_mlperf: Match the MLPerf reference behavior
  """
  shard_path, num_items, num_neg, seed, is_training, match_mlperf = args
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

  users = shard[rconst.TRAIN_KEY][movielens.USER_COLUMN]
  items = shard[rconst.TRAIN_KEY][movielens.ITEM_COLUMN]

  if not is_training:
    # For eval, there is one positive which was held out from the training set.
    test_positive_dict = dict(zip(
        shard[rconst.EVAL_KEY][movielens.USER_COLUMN],
        shard[rconst.EVAL_KEY][movielens.ITEM_COLUMN]))

  delta = users[1:] - users[:-1]
  boundaries = ([0] + (np.argwhere(delta)[:, 0] + 1).tolist() +
                [users.shape[0]])

  user_blocks = []
  item_blocks = []
  label_blocks = []
  for i in range(len(boundaries) - 1):
    assert len(set(users[boundaries[i]:boundaries[i+1]])) == 1
    current_user = users[boundaries[i]]

    positive_items = items[boundaries[i]:boundaries[i+1]]
    positive_set = set(positive_items)
    if positive_items.shape[0] != len(positive_set):
      raise ValueError("Duplicate entries detected.")

    if is_training:
      n_pos = len(positive_set)
      negatives = stat_utils.sample_with_exclusion(
          num_items, positive_set, n_pos * num_neg, replacement=True)

    else:
      if not match_mlperf:
        # The mlperf reference allows the holdout item to appear as a negative.
        # Including it in the positive set makes the eval more stringent,
        # because an appearance of the test item would be removed by
        # deduplication rules. (Effectively resulting in a minute reduction of
        # NUM_EVAL_NEGATIVES)
        positive_set.add(test_positive_dict[current_user])

      negatives = stat_utils.sample_with_exclusion(
          num_items, positive_set, num_neg, replacement=match_mlperf)
      positive_set = [test_positive_dict[current_user]]
      n_pos = len(positive_set)
      assert n_pos == 1

    user_blocks.append(current_user * np.ones(
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


def _construct_record(users, items, labels=None, dupe_mask=None):
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

  if dupe_mask is not None:
    feature_dict[rconst.DUPLICATE_MASK] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[memoryview(dupe_mask).tobytes()]))

  return tf.train.Example(
      features=tf.train.Features(feature=feature_dict)).SerializeToString()


def sigint_handler(signal, frame):
  log_msg("Shutting down worker.")


def init_worker():
  signal.signal(signal.SIGINT, sigint_handler)


def _construct_records(
    is_training,          # type: bool
    train_cycle,          # type: typing.Optional[int]
    num_workers,          # type: int
    cache_paths,          # type: rconst.Paths
    num_readers,          # type: int
    num_neg,              # type: int
    num_positives,        # type: int
    num_items,            # type: int
    epochs_per_cycle,     # type: int
    batch_size,           # type: int
    training_shards,      # type: typing.List[str]
    deterministic=False,  # type: bool
    match_mlperf=False    # type: bool
    ):
  """Generate false negatives and write TFRecords files.

  Args:
    is_training: Are training records (True) or eval records (False) created.
    train_cycle: Integer of which cycle the generated data is for.
    num_workers: Number of multiprocessing workers to use for negative
      generation.
    cache_paths: Paths object with information of where to write files.
    num_readers: The number of reader datasets in the input_fn. This number is
      approximate; fewer shards will be created if not all shards are assigned
      batches. This can occur due to discretization in the assignment process.
    num_neg: The number of false negatives per positive example.
    num_positives: The number of positive examples. This value is used
      to pre-allocate arrays while the imap is still running. (NumPy does not
      allow dynamic arrays.)
    num_items: The cardinality of the item set.
    epochs_per_cycle: The number of epochs worth of data to construct.
    batch_size: The expected batch size used during training. This is used
      to properly batch data when writing TFRecords.
    training_shards: The picked positive examples from which to generate
      negatives.
  """
  st = timeit.default_timer()

  if is_training:
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_STEP_TRAIN_NEG_GEN)
    mlperf_helper.ncf_print(
        key=mlperf_helper.TAGS.INPUT_HP_NUM_NEG, value=num_neg)

    # set inside _process_shard()
    mlperf_helper.ncf_print(
        key=mlperf_helper.TAGS.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT, value=True)

  else:
    # Later logic assumes that all items for a given user are in the same batch.
    assert not batch_size % (rconst.NUM_EVAL_NEGATIVES + 1)
    assert num_neg == rconst.NUM_EVAL_NEGATIVES

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_STEP_EVAL_NEG_GEN)

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_HP_NUM_USERS,
                            value=num_positives)

  assert epochs_per_cycle == 1 or is_training
  num_workers = min([num_workers, len(training_shards) * epochs_per_cycle])

  num_pts = num_positives * (1 + num_neg)

  # Equivalent to `int(ceil(num_pts / batch_size)) * batch_size`, but without
  # precision concerns
  num_pts_with_padding = (num_pts + batch_size - 1) // batch_size * batch_size
  num_padding = num_pts_with_padding - num_pts

  # We choose a different random seed for each process, so that the processes
  # will not all choose the same random numbers.
  process_seeds = [stat_utils.random_int32()
                   for _ in training_shards * epochs_per_cycle]
  map_args = [
      (shard, num_items, num_neg, process_seeds[i], is_training, match_mlperf)
      for i, shard in enumerate(training_shards * epochs_per_cycle)]

  with popen_helper.get_pool(num_workers, init_worker) as pool:
    map_fn = pool.imap if deterministic else pool.imap_unordered  # pylint: disable=no-member
    data_generator = map_fn(_process_shard, map_args)
    data = [
        np.zeros(shape=(num_pts_with_padding,), dtype=np.int32) - 1,
        np.zeros(shape=(num_pts_with_padding,), dtype=np.uint16),
        np.zeros(shape=(num_pts_with_padding,), dtype=np.int8),
    ]

    # Training data is shuffled. Evaluation data MUST not be shuffled.
    # Downstream processing depends on the fact that evaluation data for a given
    # user is grouped within a batch.
    if is_training:
      index_destinations = np.random.permutation(num_pts)
      mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_ORDER)
    else:
      index_destinations = np.arange(num_pts)

    start_ind = 0
    for data_segment in data_generator:
      n_in_segment = data_segment[0].shape[0]
      dest = index_destinations[start_ind:start_ind + n_in_segment]
      start_ind += n_in_segment
      for i in range(3):
        data[i][dest] = data_segment[i]

  assert np.sum(data[0] == -1) == num_padding

  if is_training:
    if num_padding:
      # In order to have a full batch, randomly include points from earlier in
      # the batch.

      mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_ORDER)
      pad_sample_indices = np.random.randint(
          low=0, high=num_pts, size=(num_padding,))
      dest = np.arange(start=start_ind, stop=start_ind + num_padding)
      start_ind += num_padding
      for i in range(3):
        data[i][dest] = data[i][pad_sample_indices]
  else:
    # For Evaluation, padding is all zeros. The evaluation input_fn knows how
    # to interpret and discard the zero padded entries.
    data[0][num_pts:] = 0

  # Check that no points were overlooked.
  assert not np.sum(data[0] == -1)

  if is_training:
    # The number of points is slightly larger than num_pts due to padding.
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_SIZE,
                            value=int(data[0].shape[0]))
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_BATCH_SIZE,
                            value=batch_size)
  else:
    # num_pts is logged instead of int(data[0].shape[0]), because the size
    # of the data vector includes zero pads which are ignored.
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_SIZE, value=num_pts)

  batches_per_file = np.ceil(num_pts_with_padding / batch_size / num_readers)
  current_file_id = -1
  current_batch_id = -1
  batches_by_file = [[] for _ in range(num_readers)]

  while True:
    current_batch_id += 1
    if (current_batch_id % batches_per_file) == 0:
      current_file_id += 1

    start_ind = current_batch_id * batch_size
    end_ind = start_ind + batch_size
    if end_ind > num_pts_with_padding:
      if start_ind != num_pts_with_padding:
        raise ValueError("Batch padding does not line up")
      break
    batches_by_file[current_file_id].append(current_batch_id)

  # Drop shards which were not assigned batches
  batches_by_file = [i for i in batches_by_file if i]
  num_readers = len(batches_by_file)

  if is_training:
    # Empirically it is observed that placing the batch with repeated values at
    # the start rather than the end improves convergence.
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.INPUT_ORDER)
    batches_by_file[0][0], batches_by_file[-1][-1] = \
      batches_by_file[-1][-1], batches_by_file[0][0]

  if is_training:
    template = rconst.TRAIN_RECORD_TEMPLATE
    record_dir = os.path.join(cache_paths.train_epoch_dir,
                              get_cycle_folder_name(train_cycle))
    tf.gfile.MakeDirs(record_dir)
  else:
    template = rconst.EVAL_RECORD_TEMPLATE
    record_dir = cache_paths.eval_data_subdir

  batch_count = 0
  for i in range(num_readers):
    fpath = os.path.join(record_dir, template.format(i))
    log_msg("Writing {}".format(fpath))
    with tf.python_io.TFRecordWriter(fpath) as writer:
      for j in batches_by_file[i]:
        start_ind = j * batch_size
        end_ind = start_ind + batch_size
        record_kwargs = dict(
            users=data[0][start_ind:end_ind],
            items=data[1][start_ind:end_ind],
        )

        if is_training:
          record_kwargs["labels"] = data[2][start_ind:end_ind]
        else:
          record_kwargs["dupe_mask"] = stat_utils.mask_duplicates(
              record_kwargs["items"].reshape(-1, num_neg + 1),
              axis=1).flatten().astype(np.int8)

        batch_bytes = _construct_record(**record_kwargs)

        writer.write(batch_bytes)
        batch_count += 1

  # We write to a temp file then atomically rename it to the final file, because
  # writing directly to the final file can cause the main process to read a
  # partially written JSON file.
  ready_file_temp = os.path.join(record_dir, rconst.READY_FILE_TEMP)
  with tf.gfile.Open(ready_file_temp, "w") as f:
    json.dump({
        "batch_size": batch_size,
        "batch_count": batch_count,
    }, f)
  ready_file = os.path.join(record_dir, rconst.READY_FILE)
  tf.gfile.Rename(ready_file_temp, ready_file)

  if is_training:
    log_msg("Cycle {} complete. Total time: {:.1f} seconds"
            .format(train_cycle, timeit.default_timer() - st))
  else:
    log_msg("Eval construction complete. Total time: {:.1f} seconds"
            .format(timeit.default_timer() - st))


def _generation_loop(num_workers,           # type: int
                     cache_paths,           # type: rconst.Paths
                     num_readers,           # type: int
                     num_neg,               # type: int
                     num_train_positives,   # type: int
                     num_items,             # type: int
                     num_users,             # type: int
                     epochs_per_cycle,      # type: int
                     num_cycles,            # type: int
                     train_batch_size,      # type: int
                     eval_batch_size,       # type: int
                     deterministic,         # type: bool
                     match_mlperf           # type: bool
                    ):
  # type: (...) -> None
  """Primary run loop for data file generation."""

  log_msg("Entering generation loop.")
  tf.gfile.MakeDirs(cache_paths.train_epoch_dir)
  tf.gfile.MakeDirs(cache_paths.eval_data_subdir)

  training_shards = [os.path.join(cache_paths.train_shard_subdir, i) for i in
                     tf.gfile.ListDirectory(cache_paths.train_shard_subdir)]

  shared_kwargs = dict(
      num_workers=multiprocessing.cpu_count(), cache_paths=cache_paths,
      num_readers=num_readers, num_items=num_items,
      training_shards=training_shards, deterministic=deterministic,
      match_mlperf=match_mlperf
  )

  # Training blocks on the creation of the first epoch, so the num_workers
  # limit is not respected for this invocation
  train_cycle = 0
  _construct_records(
      is_training=True, train_cycle=train_cycle, num_neg=num_neg,
      num_positives=num_train_positives, epochs_per_cycle=epochs_per_cycle,
      batch_size=train_batch_size, **shared_kwargs)

  # Construct evaluation set.
  shared_kwargs["num_workers"] = num_workers
  _construct_records(
      is_training=False, train_cycle=None, num_neg=rconst.NUM_EVAL_NEGATIVES,
      num_positives=num_users, epochs_per_cycle=1, batch_size=eval_batch_size,
      **shared_kwargs)

  wait_count = 0
  start_time = time.time()
  while train_cycle < num_cycles:
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
    _construct_records(
        is_training=True, train_cycle=train_cycle, num_neg=num_neg,
        num_positives=num_train_positives, epochs_per_cycle=epochs_per_cycle,
        batch_size=train_batch_size, **shared_kwargs)

    wait_count = 0
    start_time = time.time()
    gc.collect()


def wait_for_path(fpath):
  start_time = time.time()
  while not tf.gfile.Exists(fpath):
    if time.time() - start_time > rconst.TIMEOUT_SECONDS:
      log_msg("Waited more than {} seconds. Concluding that this "
              "process is orphaned and exiting gracefully."
              .format(rconst.TIMEOUT_SECONDS))
      sys.exit()
    time.sleep(1)

def _parse_flagfile(flagfile):
  """Fill flags with flagfile written by the main process."""
  tf.logging.info("Waiting for flagfile to appear at {}..."
                  .format(flagfile))
  wait_for_path(flagfile)
  tf.logging.info("flagfile found.")

  # `flags` module opens `flagfile` with `open`, which does not work on
  # google cloud storage etc.
  _, flagfile_temp = tempfile.mkstemp()
  tf.gfile.Copy(flagfile, flagfile_temp, overwrite=True)

  flags.FLAGS([__file__, "--flagfile", flagfile_temp])
  tf.gfile.Remove(flagfile_temp)


def write_alive_file(cache_paths):
  """Write file to signal that generation process started correctly."""
  wait_for_path(cache_paths.cache_root)

  log_msg("Signaling that I am alive.")
  with tf.gfile.Open(cache_paths.subproc_alive, "w") as f:
    f.write("Generation subproc has started.")

  @atexit.register
  def remove_alive_file():
    try:
      tf.gfile.Remove(cache_paths.subproc_alive)
    except tf.errors.NotFoundError:
      return  # Main thread has already deleted the entire cache dir.


def main(_):
  # Note: The async process must execute the following two steps in the
  #       following order BEFORE doing anything else:
  #       1) Write the alive file
  #       2) Wait for the flagfile to be written.
  global _log_file
  cache_paths = rconst.Paths(
      data_dir=flags.FLAGS.data_dir, cache_id=flags.FLAGS.cache_id)
  write_alive_file(cache_paths=cache_paths)

  flagfile = os.path.join(cache_paths.cache_root, rconst.FLAGFILE)
  _parse_flagfile(flagfile)

  redirect_logs = flags.FLAGS.redirect_logs

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

    with mlperf_helper.LOGGER(
        enable=flags.FLAGS.output_ml_perf_compliance_logging):
      mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
      _generation_loop(
          num_workers=flags.FLAGS.num_workers,
          cache_paths=cache_paths,
          num_readers=flags.FLAGS.num_readers,
          num_neg=flags.FLAGS.num_neg,
          num_train_positives=flags.FLAGS.num_train_positives,
          num_items=flags.FLAGS.num_items,
          num_users=flags.FLAGS.num_users,
          epochs_per_cycle=flags.FLAGS.epochs_per_cycle,
          num_cycles=flags.FLAGS.num_cycles,
          train_batch_size=flags.FLAGS.train_batch_size,
          eval_batch_size=flags.FLAGS.eval_batch_size,
          deterministic=flags.FLAGS.seed is not None,
          match_mlperf=flags.FLAGS.ml_perf,
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
  """Construct flags for the server."""
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
  flags.DEFINE_integer(name="num_users", default=None,
                       help="The number of unique users. Used for evaluation.")
  flags.DEFINE_integer(name="epochs_per_cycle", default=1,
                       help="The number of epochs of training data to produce"
                            "at a time.")
  flags.DEFINE_integer(name="num_cycles", default=None,
                       help="The number of cycles to produce training data "
                            "for.")
  flags.DEFINE_integer(name="train_batch_size", default=None,
                       help="The batch size with which training TFRecords will "
                            "be chunked.")
  flags.DEFINE_integer(name="eval_batch_size", default=None,
                       help="The batch size with which evaluation TFRecords "
                            "will be chunked.")
  flags.DEFINE_boolean(name="redirect_logs", default=False,
                       help="Catch logs and write them to a file. "
                            "(Useful if this is run as a subprocess)")
  flags.DEFINE_boolean(name="use_tf_logging", default=False,
                       help="Use tf.logging instead of log file.")
  flags.DEFINE_integer(name="seed", default=None,
                       help="NumPy random seed to set at startup. If not "
                            "specified, a seed will not be set.")
  flags.DEFINE_boolean(name="ml_perf", default=None,
                       help="Match MLPerf. See ncf_main.py for details.")
  flags.DEFINE_bool(name="output_ml_perf_compliance_logging", default=None,
                    help="Output the MLPerf compliance logging. See "
                         "ncf_main.py for details.")

  flags.mark_flags_as_required(["data_dir", "cache_id"])

if __name__ == "__main__":
  define_flags()
  absl_app.run(main)
