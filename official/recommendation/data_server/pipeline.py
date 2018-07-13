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
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "server.py")


def make_stub():
  channel = grpc.insecure_channel("localhost:{}".format(_PORT))
  return server_command_pb2_grpc.TrainDataStub(channel)


def alive():
  try:
    response = make_stub().Alive(server_command_pb2.Check())
    return response.success
  except grpc.RpcError:
    return False


def enqueue(num_epochs=1, shuffle_buffer_size=1024**2):
  make_stub().Enqueue(server_command_pb2.NumEpochs(
      value=num_epochs, shuffle_buffer_size=shuffle_buffer_size))


def get_batch(shuffle=True):
  response = make_stub().GetBatch(server_command_pb2.BatchRequest(shuffle=shuffle))
  users = np.frombuffer(response.users, dtype=np.int32)
  items = np.frombuffer(response.items, dtype=np.uint16)
  labels = np.frombuffer(response.labels, dtype=np.int8)
  return users, items, labels


def shutdown():
  make_stub().ShutdownServer(server_command_pb2.Shutdown())


def _shutdown_thorough(proc):
  try:
    proc.send_signal(signal.SIGINT)
    time.sleep(1)
  finally:
    tf.logging.info("Shutting down train data GRPC server.")
    proc.terminate()


def initialize(dataset, data_dir, num_neg, num_data_readers=None):
  ncf_dataset = prepare.run(dataset=dataset, data_dir=data_dir,
                            num_data_readers=num_data_readers, num_neg=num_neg)
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
      enqueue(num_epochs=num_epochs, shuffle_buffer_size=10 * 1024 ** 2)

      def make_reader(_):
        # reader_dataset = tf.data.Dataset.range(num_epochs * ncf_dataset.shards_per_reader)
        reader_dataset = tf.contrib.data.Counter()
        def _rpc_fn(_):
          rpc_op = tf.contrib.rpc.rpc(
              address="localhost:{}".format(_PORT),
              method="/official.recommendation.data_server.TrainData/GetBatch",
              request=server_command_pb2.BatchRequest(shuffle=True, max_batch_size=batch_size).SerializeToString(),
              protocol="grpc",
          )
          def _decode_proto(shard_bytes):
            message = server_command_pb2.Batch.FromString(shard_bytes)
            users = np.frombuffer(message.users, dtype=np.int32)
            items = np.frombuffer(message.items, dtype=np.uint16)
            labels = np.frombuffer(message.labels, dtype=np.int8)
            users, items, labels = [np.expand_dims(i, axis=1) for i in [users, items, labels]]
            if users.shape[0] == 0:
              raise StopIteration
            # import json
            # metadata = json.loads(message.metadata_json.decode("utf-8"))
            # print(users.shape, items.shape, labels.shape, metadata)
            return users, items, labels

          decoded_shard = tf.py_func(_decode_proto, inp=[rpc_op], Tout=(np.int32, np.uint16, np.int8))
          return {
            movielens.USER_COLUMN: tf.reshape(decoded_shard[0], (-1, 1)),
            movielens.ITEM_COLUMN: tf.reshape(decoded_shard[1], (-1, 1)),
          }, tf.reshape(decoded_shard[2], (-1, 1))

        reader_dataset = reader_dataset.map(_rpc_fn, num_parallel_calls=2)

        return reader_dataset

      dataset = tf.data.Dataset.range(ncf_dataset.num_data_readers)
      interleave = tf.contrib.data.parallel_interleave(
          make_reader,
          cycle_length=ncf_dataset.num_data_readers,
          block_length=4,
          sloppy=True,
          prefetch_input_elements=4,
      )
      dataset = dataset.apply(interleave)  #.prefetch(32)


      # with tf.Session().as_default() as sess:
      #   shard = dataset.make_one_shot_iterator().get_next()
      #   count = 0
      #   st = time.time()
      #   while True:
      #     try:
      #       result = sess.run(shard)
      #       count += 1
      #     except tf.errors.OutOfRangeError:
      #       break
      #     # print(result)
      #     # print(result[0].shape, result[1].shape, result[2].shape)
      #     if count % 25 == 0:
      #       print(count / (time.time() - st))
      #
      # import sys
      # sys.exit()
      #
      # def train_generator():
      #   users, items, labels = get_batch(shuffle=True)
      #   users = np.expand_dims(users, axis=1)
      #   items = np.expand_dims(items, axis=1)
      #   labels = np.expand_dims(labels, axis=1)
      #   n = labels.shape[0]
      #   n_batches = int(np.ceil(n / batch_size))
      #   for i in range(n_batches):
      #     yield ({
      #       movielens.USER_COLUMN: users[i * batch_size: (i+1) * batch_size, :],
      #       movielens.ITEM_COLUMN: items[i * batch_size: (i+1) * batch_size, :],
      #     }, labels[i * batch_size: (i+1) * batch_size, :])
      #
      # output_types = ({
      #   movielens.USER_COLUMN: tf.int32,
      #   movielens.ITEM_COLUMN: tf.uint16,
      # }, tf.int8)
      #
      # output_shapes = ({
      #   movielens.USER_COLUMN: tf.TensorShape([None, 1]),
      #   movielens.ITEM_COLUMN: tf.TensorShape([None, 1]),
      # }, tf.TensorShape([None, 1]))
      #
      # dataset = tf.data.Dataset.range(num_epochs * ncf_dataset.num_train_shards)
      # interleave = tf.contrib.data.parallel_interleave(
      #     lambda _: tf.data.Dataset.from_generator(
      #         train_generator, output_types=output_types,
      #         output_shapes=output_shapes),
      #     cycle_length=multiprocessing.cpu_count(),
      #     block_length=10,
      #     sloppy=True,
      # )
      # dataset = dataset.apply(interleave)

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
