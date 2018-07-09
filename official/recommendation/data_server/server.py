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


class TrainData(server_command_pb2_grpc.TrainDataServicer):
  def __init__(self, pool, shard_dir, num_neg, num_items):
    super(TrainData, self).__init__()
    self.pool = pool  # type: multiprocessing.Pool
    self.should_stop = False

    if not tf.gfile.Exists(shard_dir):
      raise ValueError("shard_dir `{}` does not exist.".format(shard_dir))
    shards = tf.gfile.ListDirectory(shard_dir)
    invalid_shards = [i for i in shards if not i.endswith(".pickle")]
    if invalid_shards:
      raise ValueError("Invalid shard(s): {}".format(", ".join(invalid_shards)))
    self.shards = [os.path.join(shard_dir, i) for i in shards]

    self._map_fn = functools.partial(_process_shard, num_neg=num_neg,
                                     num_items=num_items)
    self._shard_queue = []

  def Alive(self, request, context):
    return server_command_pb2.Ack(success=True)

  def Enqueue(self, request, context):
    num_epochs = request.value
    self._shard_queue = []
    for _ in range(num_epochs):
      self._shard_queue.extend(self.shards)

    return server_command_pb2.Ack(success=True)

  def GetBatch(self, request, context):
    if not self._shard_queue:
      context.abort(1, "No shards enqueued.")
    batch_shard = self._shard_queue.pop()
    batch = self.pool.apply(self._map_fn, args=[batch_shard])

    users, items, labels = batch
    if request.shuffle:
      shuffle_ind = np.random.permutation(users.shape[0])
      users = users[shuffle_ind]
      items = items[shuffle_ind]
      labels = labels[shuffle_ind]

    response = server_command_pb2.Batch()
    response.users = bytes(memoryview(users))
    response.items = bytes(memoryview(items))
    response.labels = bytes(memoryview(labels))
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
  run_server(port=port, num_workers=num_workers, shard_dir=shard_dir,
             num_neg=num_neg, num_items=num_items)


if __name__ == "__main__":
  absl_logging.set_verbosity(absl_logging.INFO)
  define_flags()
  absl_app.run(main)
