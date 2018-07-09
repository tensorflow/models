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

import atexit
import multiprocessing
import os
import signal
import subprocess
import time
import typing

import grpc
import numpy as np
import six
import tensorflow as tf

from official.datasets import movielens
from official.recommendation.data_server import prepare
from official.recommendation.data_server import server_command_pb2
from official.recommendation.data_server import server_command_pb2_grpc


_PORT = 46293
_CHANNEL = grpc.insecure_channel("localhost:{}".format(_PORT))
_STUB = server_command_pb2_grpc.TrainDataStub(_CHANNEL)
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "server.py")


def alive():
  try:
    response = _STUB.Alive(server_command_pb2.Check())
    return response.success
  except grpc.RpcError:
    return False


def enqueue(num_epochs=1):
  _STUB.Enqueue(server_command_pb2.NumEpochs(value=num_epochs))


def get_batch(shuffle=True):
  response = _STUB.GetBatch(server_command_pb2.BatchRequest(shuffle=shuffle))
  users = np.frombuffer(response.users, dtype=np.int32)
  items = np.frombuffer(response.items, dtype=np.uint16)
  labels = np.frombuffer(response.labels, dtype=np.int8)
  return users, items, labels


def shutdown():
  _STUB.ShutdownServer(server_command_pb2.Shutdown())


def _shutdown_thorough(proc):
  try:
    proc.send_signal(signal.SIGINT)
    time.sleep(1)
  finally:
    tf.logging.info("Shutting down train data GRPC server.")
    proc.terminate()


def initialize(dataset, data_dir, num_neg):
  ncf_dataset = prepare.run(dataset=dataset, data_dir=data_dir)
  server_env = os.environ.copy()
  server_env["CUDA_VISIBLE_DEVICES"] = ""
  python = "python3" if six.PY3 else "python2"

  # By limiting the number of workers we guarantee that the worker
  # pool underlying the GRPC server doesn't starve other processes.
  num_workers = int(multiprocessing.cpu_count() * 0.75)

  server_args = [python, _SERVER_PATH,
                 "--shard_dir", ncf_dataset.train_shard_dir,
                 "--num_neg", str(num_neg),
                 "--num_items", str(ncf_dataset.num_items),
                 "--num_workers", str(num_workers)]
  proc = subprocess.Popen(args=server_args, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          shell=False, env=server_env)

  atexit.register(_shutdown_thorough, proc=proc)

  while True:
    tf.logging.info("Checking for train data GRPC server...")
    if alive():
      break
    time.sleep(1)
  tf.logging.info("Train data GRPC server is up.")

  return ncf_dataset


def get_input_fn(training, ncf_dataset, batch_size, num_epochs=1):
  # type: (bool, prepare.NCFDataset, int) -> typing.Callable
  def input_fn():
    if training:
      if not alive():
        raise OSError("No train data GRPC server.")
      enqueue(num_epochs=num_epochs)

      def train_generator():
        users, items, labels = get_batch(shuffle=True)
        users = np.expand_dims(users, axis=1)
        items = np.expand_dims(items, axis=1)
        labels = np.expand_dims(labels, axis=1)
        n = labels.shape[0]
        n_batches = int(np.ceil(n / batch_size))
        for i in range(n_batches):
          yield ({
            movielens.USER_COLUMN: users[i * batch_size: (i+1) * batch_size, :],
            movielens.ITEM_COLUMN: items[i * batch_size: (i+1) * batch_size, :],
          }, labels[i * batch_size: (i+1) * batch_size, :])

      output_types = ({
        movielens.USER_COLUMN: tf.int32,
        movielens.ITEM_COLUMN: tf.uint16,
      }, tf.int8)

      output_shapes = ({
        movielens.USER_COLUMN: tf.TensorShape([None, 1]),
        movielens.ITEM_COLUMN: tf.TensorShape([None, 1]),
      }, tf.TensorShape([None, 1]))

      dataset = tf.data.Dataset.range(num_epochs * ncf_dataset.num_train_shards)
      interleave = tf.contrib.data.parallel_interleave(
          lambda _: tf.data.Dataset.from_generator(
              train_generator, output_types=output_types,
              output_shapes=output_shapes),
          cycle_length=multiprocessing.cpu_count(),
          block_length=10,
          sloppy=True,
      )
      dataset = dataset.apply(interleave)

    else:
      # Using Dataset.from_generator() rather than
      # Dataset.from_tensor_slices().batch() has two benefits:
      #   1) The test data does not have to be serialized into the TensorFlow
      #      graph.
      #   2) Batching using NumPy slices is significantly faster than
      #      using Dataset.batch()
      #
      # Overall this results in a ~50x increase in throughput.
      def _pred_generator():
        users = np.expand_dims(ncf_dataset.test_data[0][movielens.USER_COLUMN],
                               axis=1)
        items = np.expand_dims(ncf_dataset.test_data[0][movielens.ITEM_COLUMN],
                               axis=1)
        n = users.shape[0]
        n_batches = int(np.ceil(n / batch_size))
        for i in range(n_batches):
          yield {
            movielens.USER_COLUMN: users[i * batch_size: (i+1) * batch_size, :],
            movielens.ITEM_COLUMN: items[i * batch_size: (i+1) * batch_size, :],
          }

      output_types = {movielens.USER_COLUMN: tf.int32,
                      movielens.ITEM_COLUMN: tf.uint16}
      output_shapes = {movielens.USER_COLUMN: tf.TensorShape([None, 1]),
                       movielens.ITEM_COLUMN: tf.TensorShape([None, 1])}

      dataset = tf.data.Dataset.from_generator(
          _pred_generator, output_types=output_types,
          output_shapes=output_shapes)

    return dataset.prefetch(32)

  return input_fn
