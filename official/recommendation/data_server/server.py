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
import grpc
from concurrent import futures
import multiprocessing
import os
import pickle
import signal
import sys
import time

from absl import app as absl_app
from absl import logging as absl_logging
from absl import flags
import numpy as np
import  tensorflow as tf

from official.datasets import movielens
from official.recommendation.data_server import prepare
from official.recommendation.data_server import server_command_pb2
from official.recommendation.data_server import server_command_pb2_grpc


def _process_shard(shard_path, num_items, num_neg):
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

    # TODO(robieta): include option to prevent eval pts from becoming negatives
    negatives = prepare.construct_false_negatives(
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
  if batch_size / buffer_size >= 0.25:
    return np.random.choice(range(buffer_size), size=batch_size, replace=False)

  p = 1 - batch_size / buffer_size
  sample_size = int(batch_size * (1 / p) * 1.2)
  indicies = set(np.random.randint(low=0, high=buffer_size, size=(sample_size,)))
  while len(indicies) < batch_size:
    indicies = set(np.random.randint(low=0, high=buffer_size, size=(sample_size,)))
  return np.random.choice(list(indicies), size=batch_size, replace=False)


class TrainData(server_command_pb2_grpc.TrainDataServicer):
  def __init__(self, pool, shard_dir, num_neg, num_items):
    super(TrainData, self).__init__()
    self.pool = pool  # type: multiprocessing.Pool
    self._lock_manager = multiprocessing.Manager()

    self.should_stop = False

    if not tf.gfile.Exists(shard_dir):
      raise ValueError("shard_dir `{}` does not exist.".format(shard_dir))
    shards = tf.gfile.ListDirectory(shard_dir)
    invalid_shards = [i for i in shards if not i.endswith(".pickle") and not tf.gfile.IsDirectory(os.path.join(shard_dir, i))]
    if invalid_shards:
      raise ValueError("Invalid shard(s): {}".format(", ".join(invalid_shards)))
    self.shards = [os.path.join(shard_dir, i) for i in shards if not tf.gfile.IsDirectory(os.path.join(shard_dir, i))]

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

  def Alive(self, request, context):
    return server_command_pb2.Ack(success=True)

  def Enqueue(self, request, context):
    if self._buffer_arrays[0].shape[0] > 0 or not self._mapper_exhausted:
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
    if shuffle:
      n = n or self._shuffle_buffer_size + k - 1

      result = self.pool.apply(fischer_yates_subsample, kwds=dict(batch_size=k, buffer_size=n))
      return result
    return np.arange(k)

  def GetBatch(self, request, context):
    max_batch_size = request.max_batch_size
    shuffle = request.shuffle
    subsample_indicies = self.get_subsample_indicies(
        k=max_batch_size, shuffle=shuffle)

    with self._buffer_lock:
      buffer_size = self._buffer_arrays[0].shape[0]
      if buffer_size < self._shuffle_buffer_size + max_batch_size - 1 and not self._mapper_exhausted:
        secondary_buffer = []
        new_buffer_size = buffer_size
        while (new_buffer_size < self._shuffle_buffer_size * self._overfill_factor
               and not self._mapper_exhausted):
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
        subsample_indicies = self.get_subsample_indicies(
            k=max_batch_size, n=buffer_size, shuffle=shuffle)
      else:
        subsample_indicies = np.arange(buffer_size)

      batch_size = subsample_indicies.shape[0]
      output = [
        self._buffer_arrays[i][subsample_indicies].copy() for i in range(3)
      ]
      high_indicies = subsample_indicies[np.argwhere(subsample_indicies >= batch_size)[:, 0]]
      low_indicies = subsample_indicies[np.argwhere(subsample_indicies < batch_size)[:, 0]]
      low_index_conjugate = np.arange(batch_size)
      low_index_conjugate[low_indicies] = -1
      low_index_conjugate = low_index_conjugate[np.argwhere(low_index_conjugate >= 0)[:, 0]]
      for i in range(3):
        self._buffer_arrays[i][high_indicies] = self._buffer_arrays[i][low_index_conjugate]
        self._buffer_arrays[i] = self._buffer_arrays[i][batch_size:]

    n = output[0].shape[0]
    print("Serving batch: n = {}".format(n))

    response = server_command_pb2.Batch()
    response.users = bytes(memoryview(output[0]))
    response.items = bytes(memoryview(output[1]))
    response.labels = bytes(memoryview(output[2]))
    return response

  def ShutdownServer(self, request, context):
    response = server_command_pb2.Ack()
    self.should_stop = True
    response.success = True
    return response


def sigint_handler(_signal, frame):
  absl_logging.info("Shutting down worker.")


def init_worker():
  signal.signal(signal.SIGINT, sigint_handler)


def run_server(port, num_workers, shard_dir, num_neg, num_items):
  with contextlib.closing(multiprocessing.Pool(
      processes=num_workers, initializer=init_worker)) as pool:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_workers))
    servicer = TrainData(pool=pool, shard_dir=shard_dir,
                         num_neg=num_neg, num_items=num_items)
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
  flags.mark_flags_as_required(["shard_dir", "num_neg", "num_items"])


def main(_):
  port = flags.FLAGS.port
  num_workers = flags.FLAGS.num_workers
  shard_dir = flags.FLAGS.shard_dir
  num_neg = flags.FLAGS.num_neg
  num_items = flags.FLAGS.num_items
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
               num_neg=num_neg, num_items=num_items)
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout.close()
    stderr.close()


if __name__ == "__main__":
  absl_logging.set_verbosity(absl_logging.INFO)
  define_flags()
  absl_app.run(main)
