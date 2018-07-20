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
"""Construct training data server, and access it through GRPC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import multiprocessing
import os
import signal
import socket
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
_REMOTE_ADDRESS = "{}:{}".format(socket.gethostbyname(socket.gethostname()), _PORT)
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "server.py")

_SHUFFLE_BUFFER_SIZE = 60 * 1024 ** 2


def make_stub():
  channel = grpc.insecure_channel(_REMOTE_ADDRESS)
  return server_command_pb2_grpc.TrainDataStub(channel)


def alive():
  # type: () -> bool
  try:
    response = make_stub().Alive(server_command_pb2.Check())
    return response.success
  except grpc.RpcError:
    return False


def enqueue(num_epochs=1, shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE):
  make_stub().Enqueue(server_command_pb2.NumEpochs(
      value=num_epochs, shuffle_buffer_size=shuffle_buffer_size))


def shutdown():
  make_stub().ShutdownServer(server_command_pb2.Shutdown())


def _shutdown_thorough(proc):
  try:
    proc.send_signal(signal.SIGINT)
    time.sleep(1)
  finally:
    tf.logging.info("Shutting down train data GRPC server.")
    proc.terminate()


def initialize(dataset, data_dir, num_neg, num_data_readers=None, debug=False):
  # type: (str, str, int, int, bool) -> prepare.NCFDataset
  """Load data, create a data server, and return the dataset container object.

  The main role of this function is to create a GRPC server in a subprocess
  which can serve training data using the tf.contrib.rpc.rpc() op. This pattern
  allows the server to operate asynchronously and avoids a variety of issues
  which spring up when attempting to perform significant data preprocessing
  outside of TensorFlow on the same thread as the training thread. (Such as the
  GIL.)

  Args:
    dataset: The name of the dataset to be used for training.
    data_dir: The directory in which the data should be written.
    num_neg: The number of false negatives per positive example to be
      generated during training.
    num_data_readers: The number of tf.data.Dataset instances to be interleaved
      during training. If `None`, a sensible value will be chosen.
    debug: A boolean indicating extra steps should be performed to assist with
      testing.

  returns:
    An NCFDataset object describing the processed data.
  """
  ncf_dataset = prepare.run(dataset=dataset, data_dir=data_dir,
                            num_data_readers=num_data_readers, num_neg=num_neg,
                            debug=debug)
  server_env = os.environ.copy()

  # The data server uses TensorFlow for tf.gfile, but it does not need GPU
  # resources and by default will try to allocate GPU memory. This would cause
  # contention with the main training process.
  server_env["CUDA_VISIBLE_DEVICES"] = ""

  python = "python3" if six.PY3 else "python2"

  # By limiting the number of workers we guarantee that the worker
  # pool underlying the GRPC server doesn't starve other processes.
  num_workers = int(multiprocessing.cpu_count() * 0.75)

  server_args = [
      python, _SERVER_PATH,
      "--shard_dir", ncf_dataset.train_shard_dir,
      "--num_neg", str(num_neg),
      "--num_items", str(ncf_dataset.num_items),
      "--num_workers", str(num_workers),
      "--spillover", "True"  # This allows the training input function to
                             # guarantee batch size and significantly improves
                             # performance. (~5% increase in examples/sec)
  ]
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


def get_input_fn(training, ncf_dataset, batch_size, num_epochs=1, shuffle=None):
  """Construct input function for Estimator.

  Overview:
    Training input_fn:

       ___________________________________
      | Training Data Server (subprocess) |
       ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
      ^  ^  ^
      |  |  |
    ...........
    .  GRPC   .
    ...........
      |  |  |    ________________
      |  |  ┗-->| Reader Dataset |   ‾‾|
      |  |       ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾      |
      |  |       ________________      |   ____________
      |  ┗----->| Reader Dataset |     |- | Interleave | -> batches
      .          ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾      |   ‾‾‾‾‾‾‾‾‾‾‾‾
      .             ...                |
                                     __|

    Eval input_fn:
       ___________________
      | Generator  Dataset| -> batches
       ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

  Args:
    training: Boolean of whether the input_fn is for training (True) or
      eval (False).
    ncf_dataset: NCFDataset instance. This MUST be the same one created by
      official.recommendation.data_server.pipeline.initialize().
    batch_size: The size of batches that the input_fn's Dataset should yield.
    num_epochs: The number of epochs to include in the dataset. If
      num_epochs > 1, new false negatives will be generated for each epoch.
    shuffle: Whether to shuffle the training data. Shuffling is performed
      by the data server, not the Dataset class.

  Returns:
    An input_fn which accepts no arguments and returns a tf.data.Dataset.
  """
  # type: (bool, prepare.NCFDataset, int, int, bool) -> typing.Callable

  def input_fn(params):
    # type: (dict) -> tf.data.Dataset
    """The input function for an NCF Estimator."""

    batch_size = params["batch_size"]

    feature_map = {
      movielens.USER_COLUMN: tf.FixedLenFeature([batch_size], dtype=tf.int64),
      movielens.ITEM_COLUMN: tf.FixedLenFeature([batch_size], dtype=tf.int64),
      "labels": tf.FixedLenFeature([batch_size], dtype=tf.int64),
    }

    def _deserialize(examples_serialized):
      features = tf.parse_single_example(examples_serialized, feature_map)
      return features[movielens.USER_COLUMN], features[movielens.ITEM_COLUMN], features["labels"]


    if training:
      if not alive():
        raise OSError("No train data GRPC server.")
      enqueue(num_epochs=num_epochs, shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE)
      calls_per_reader = int(np.ceil(ncf_dataset.num_train_pts / batch_size / ncf_dataset.num_data_readers))

      def make_reader(_):
        reader_dataset = tf.data.Dataset.range(calls_per_reader)
        def _rpc_fn(_):
          """Construct RPC op to request training data."""
          rpc_op = tf.contrib.rpc.rpc(
              address=_REMOTE_ADDRESS,
              method="/official.recommendation.data_server.TrainData/GetBatch",
              request=server_command_pb2.BatchRequest(
                  shuffle=shuffle, max_batch_size=batch_size
              ).SerializeToString(),
              protocol="grpc",
          )


          return rpc_op

        def _decode(rpc_bytes):
          decoded_shard = _deserialize(rpc_bytes)

          decoded_users, decoded_items, decoded_labels = [
              tf.reshape(decoded_shard[i], (batch_size, 1)) for i in range(3)
          ]

          return {
              movielens.USER_COLUMN: decoded_users,
              movielens.ITEM_COLUMN: decoded_items,
          }, tf.reshape(decoded_shard[2], (batch_size, 1))

        reader_dataset = reader_dataset.map(
            _rpc_fn,
            num_parallel_calls=1
        ).filter(lambda x: tf.not_equal(x, "")).map(_decode)

        return reader_dataset

      dataset = tf.data.Dataset.range(ncf_dataset.num_data_readers)
      interleave = tf.contrib.data.parallel_interleave(
          make_reader,
          cycle_length=ncf_dataset.num_data_readers,
          block_length=4,
          sloppy=True,
          prefetch_input_elements=4,
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
        """Yield slices of the validation data, and pad on the last batch."""
        users = np.expand_dims(ncf_dataset.test_data[0][movielens.USER_COLUMN],
                               axis=1)
        items = np.expand_dims(ncf_dataset.test_data[0][movielens.ITEM_COLUMN],
                               axis=1)
        n = users.shape[0]
        n_batches = int(np.ceil(n / batch_size))
        for i in range(n_batches):
          batch_users = users[i * batch_size: (i+1) * batch_size, :]
          batch_items = items[i * batch_size: (i+1) * batch_size, :]
          num_in_batch = batch_users.shape[0]
          delta = batch_size - num_in_batch
          if delta:
            batch_users = np.pad(batch_users, ((0, delta), (0, 0)), "constant")
            batch_items = np.pad(batch_items, ((0, delta), (0, 0)), "constant")

          yield {
              movielens.USER_COLUMN: batch_users,
              movielens.ITEM_COLUMN: batch_items,
              "n": num_in_batch,
          }

      output_types = {movielens.USER_COLUMN: tf.int32,
                      movielens.ITEM_COLUMN: tf.uint16,
                      "n": tf.int64}
      output_shapes = {movielens.USER_COLUMN: tf.TensorShape([batch_size, 1]),
                       movielens.ITEM_COLUMN: tf.TensorShape([batch_size, 1]),
                       "n": tf.TensorShape(None)}

      dataset = tf.data.Dataset.from_generator(
          _pred_generator, output_types=output_types,
          output_shapes=output_shapes)

    return dataset.prefetch(32)

  return input_fn
