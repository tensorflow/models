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
"""Run training data GRPC server."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
from concurrent import futures
import multiprocessing
import os
import pickle
import signal
import sys
import time
import typing

from absl import app as absl_app
from absl import logging as absl_logging
from absl import flags
import grpc
import numpy as np
import  tensorflow as tf

from official.datasets import movielens
from official.recommendation.data_server import prepare
from official.recommendation.data_server import server_command_pb2
from official.recommendation.data_server import server_command_pb2_grpc


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

    negatives = prepare.sample_with_exclusion(
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


def fischer_yates_subsample(batch_size, buffer_size):
  # type: (int, int) -> np.ndarray
  """Generate the first k elements of the domain [0, n).

  For ease of notation and to follow binomial coefficient conventions:
    k = batch_size
    n = buffer_size
    0 < k <= n

  This function subsamples a range without replacement using either a dense or
  sparse algorithm. The cutoff is set to a heuristic value of 0.25; any value of
  O(0.1) is reasonable.

  Dense subsampling:
    If k is O(n), one should simply call np.choice() on an array of [0, n).
    Under the hood NumPy shuffles an array of size n and samples the first k
    points.

  Sparse subsampling:
    if k << n, enumerating a vector of size n is unnecessary. Fortunately
    Fischer-Yates shuffling is well suited for partial shuffling. NumPy does
    not implement Fischer-Yates for k << n because NumPy does not have a set
    data structure, though it has been proposed.

    In this case, partial Fischer-Yates is simply a special case of sampling
    with exclusion where the exclusion set is the empty set.
  """

  if batch_size / buffer_size >= 0.25:
    return np.random.choice(range(buffer_size), size=batch_size, replace=False)

  return np.array(prepare.sample_with_exclusion(
      num_items=buffer_size, positive_set=set(), n=batch_size,
      replacement=False))


class TrainData(server_command_pb2_grpc.TrainDataServicer):
  """GRPC servicer which serves training data batches."""

  def __init__(self, pool, shard_dir, num_neg, num_items, spillover):
    # type: (multiprocessing.Pool, str, int, int, bool) -> None
    super(TrainData, self).__init__()
    self.pool = pool  # type: multiprocessing.Pool
    self._lock_manager = multiprocessing.Manager()

    self.should_stop = False

    if not tf.gfile.Exists(shard_dir):
      raise ValueError("shard_dir `{}` does not exist.".format(shard_dir))
    shards = tf.gfile.ListDirectory(shard_dir)
    invalid_shards = [i for i in shards if not i.endswith(".pickle")
                      and not tf.gfile.IsDirectory(os.path.join(shard_dir, i))]
    if invalid_shards:
      raise ValueError("Invalid shard(s): {}".format(", ".join(invalid_shards)))
    self.shards = [os.path.join(shard_dir, i) for i in shards
                   if not tf.gfile.IsDirectory(os.path.join(shard_dir, i))]

    self._map_fn = functools.partial(_process_shard, num_neg=num_neg,
                                     num_items=num_items)

    self._mapper = None
    self._mapper_exhausted = True
    self._buffer_arrays = [
        np.zeros((0,), dtype=np.int32),   # Users
        np.zeros((0,), dtype=np.uint16),  # Items
        np.zeros((0,), dtype=np.int8),    # Labels
    ]
    self._shuffle_buffer_size = None
    self._overfill_factor = 2
    self._buffer_lock = self._lock_manager.Lock()

    self.spillover = spillover

  def Alive(self, request, context):
    return server_command_pb2.Ack(success=True)

  def Enqueue(self, request, context):
    if ((self._buffer_arrays[0].shape[0] > 0 and not self.spillover)
        or not self._mapper_exhausted):
      raise OSError("Previous epochs have not been consumed.")

    num_epochs = request.value
    shard_queue = []
    for _ in range(num_epochs):
      shard_queue.extend(self.shards)
    self._mapper = self.pool.imap(self._map_fn, shard_queue)
    self._mapper_exhausted = False
    self._shuffle_buffer_size = request.shuffle_buffer_size
    if not self._shuffle_buffer_size:
      raise ValueError("Shuffle buffer size not specified.")

    return server_command_pb2.Ack(success=True)

  def get_subsample_indicies(self, k, n=None, shuffle=True):
    # type: (int, int, bool) -> np.ndarray
    if shuffle:
      n = n or self._shuffle_buffer_size + k - 1

      result = self.pool.apply(fischer_yates_subsample,
                               kwds=dict(batch_size=k, buffer_size=n))
      return result
    return np.arange(k)

  def GetBatch(self, request, context):
    max_batch_size = request.max_batch_size
    shuffle = request.shuffle
    subsample_indices = self.get_subsample_indicies(
        k=max_batch_size, shuffle=shuffle)

    with self._buffer_lock:
      buffer_size = self._buffer_arrays[0].shape[0]
      if (self.spillover and self._mapper_exhausted and
          buffer_size < max_batch_size):
        return server_command_pb2.Batch(users=b"", items=b"", labels=b"")

      if (buffer_size < self._shuffle_buffer_size + max_batch_size - 1
          and not self._mapper_exhausted):
        secondary_buffer = []
        new_buffer_size = buffer_size
        while (new_buffer_size < self._shuffle_buffer_size *
               self._overfill_factor and not self._mapper_exhausted):
          try:
            shard = self._mapper.next()
            secondary_buffer.append(shard)
            new_buffer_size += shard[0].shape[0]
          except StopIteration:
            self._mapper_exhausted = True

        if secondary_buffer:
          self._buffer_arrays = [
              np.concatenate([self._buffer_arrays[i]] +
                             [j[i] for j in secondary_buffer], axis=0)
              for i in range(3)
          ]

      buffer_size = self._buffer_arrays[0].shape[0]

      if buffer_size >= self._shuffle_buffer_size + max_batch_size - 1:
        pass  # common case is computed outside of the lock.
      elif buffer_size > max_batch_size:
        subsample_indices = self.get_subsample_indicies(
            k=max_batch_size, n=buffer_size, shuffle=shuffle)
      else:
        subsample_indices = np.arange(buffer_size)

      batch_size = subsample_indices.shape[0]
      output = [
          self._buffer_arrays[i][subsample_indices].copy() for i in range(3)
      ]
      high_indices = subsample_indices[
          np.argwhere(subsample_indices >= batch_size)[:, 0]]
      low_indices = subsample_indices[
          np.argwhere(subsample_indices < batch_size)[:, 0]]
      low_index_conjugate = np.arange(batch_size)
      low_index_conjugate[low_indices] = -1
      low_index_conjugate = low_index_conjugate[
          np.argwhere(low_index_conjugate >= 0)[:, 0]]
      for i in range(3):
        self._buffer_arrays[i][high_indices] = \
          self._buffer_arrays[i][low_index_conjugate]
        self._buffer_arrays[i] = self._buffer_arrays[i][batch_size:]

    n = output[0].shape[0]
    print("Serving batch: n = {}".format(n))

    return server_command_pb2.Batch(
        users=bytes(memoryview(output[0])),
        items=bytes(memoryview(output[1])),
        labels=bytes(memoryview(output[2]))
    )

  def ShutdownServer(self, request, context):
    response = server_command_pb2.Ack()
    self.should_stop = True
    response.success = True
    return response


def sigint_handler(signal, frame):
  absl_logging.info("Shutting down worker.")


def init_worker():
  signal.signal(signal.SIGINT, sigint_handler)


def run_server(port, num_workers, shard_dir, num_neg, num_items, spillover):
  # type: (int, int, str, int, int, bool) -> None
  """Bring up GRPC server.

  Args:
    port: The GRPC port to which requests should be made.
    num_workers: The number of workers (server threads and pool workers) to use.
    shard_dir: The filepath of where the training data shards are stored.
    num_neg: The number of negative examples to generate for each postive.
    num_items: The cardinality of the item set.
    spillover: Whether to return a partial final batch (False) or allow it to
      spill over into the next training cycle (True).
  """

  with contextlib.closing(multiprocessing.Pool(
      processes=num_workers, initializer=init_worker)) as pool:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_workers))
    servicer = TrainData(pool=pool, shard_dir=shard_dir,
                         num_neg=num_neg, num_items=num_items,
                         spillover=spillover)
    server_command_pb2_grpc.add_TrainDataServicer_to_server(servicer, server)

    absl_logging.info("Starting train data server on port {}".format(port))
    server.add_insecure_port("[::]:{}".format(port))
    server.start()

    try:
      while True:
        time.sleep(1)
        if servicer.should_stop:
          break
        sys.stdout.flush()
        sys.stderr.flush()
    except KeyboardInterrupt:
      pass
    finally:
      pool.terminate()
      server.stop(0)


def define_flags():
  """Construct flags for the server.

  This function does not use offical.utils.flags, as these flags are not meant
  to be used by humans. Rather, they should be passed as part of a subprocess
  call.
  """
  flags.DEFINE_integer(name="port", default=46293,
                       help="GRPC port for training data server.")
  flags.DEFINE_integer(name="num_workers", default=multiprocessing.cpu_count(),
                       help="Number of parallel requests and size of the "
                            "negative generation worker pool.")
  flags.DEFINE_string(name="shard_dir", default=None,
                      help="Location of the sharded test positives.")
  flags.DEFINE_integer(name="num_neg", default=None,
                       help="The Number of negative instances to pair with a "
                            "positive instance.")
  flags.DEFINE_integer(name="num_items", default=None,
                       help="Number of items from which to select negatives.")
  flags.DEFINE_boolean(
      name="spillover", default=True,
      help="If a complete batch cannot be provided, return an empty batch and "
           "start the next epoch from a non-empty buffer. This guarantees "
           "fixed batch sizes.")
  flags.mark_flags_as_required(["shard_dir", "num_neg", "num_items"])


def main(_):
  port = flags.FLAGS.port
  num_workers = flags.FLAGS.num_workers
  shard_dir = flags.FLAGS.shard_dir
  num_neg = flags.FLAGS.num_neg
  num_items = flags.FLAGS.num_items
  spillover = flags.FLAGS.spillover
  log_dir = os.path.join(shard_dir, "logs")
  tf.gfile.MakeDirs(log_dir)

  # This server is generally run in a subprocess.
  print("Redirecting stdout and stderr to files in {}".format(log_dir))
  stdout = open(os.path.join(log_dir, "stdout.log"), "wt")
  stderr = open(os.path.join(log_dir, "stderr.log"), "wt")
  try:
    sys.stdout = stdout
    sys.stderr = stderr
    run_server(port=port, num_workers=num_workers, shard_dir=shard_dir,
               num_neg=num_neg, num_items=num_items, spillover=spillover)
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout.close()
    stderr.close()


if __name__ == "__main__":
  absl_logging.set_verbosity(absl_logging.INFO)
  define_flags()
  absl_app.run(main)
