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
"""Preprocessing step to create, read, write tf.Examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random

import tensorflow as tf  # pylint: disable=g-bad-import-order

import coords
import features as features_lib
import numpy as np
import sgf_wrapper

TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)


# Constructing tf.Examples
def _one_hot(board_size, index):
  onehot = np.zeros([board_size * board_size + 1], dtype=np.float32)
  onehot[index] = 1
  return onehot


def make_tf_example(features, pi, value):
  """Make tf examples.

  Args:
    features: [N, N, FEATURE_DIM] nparray of uint8
    pi: [N * N + 1] nparray of float32
    value: float

  Returns:
    tf example.

  """
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'x': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[features.tostring()])),
              'pi': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[pi.tostring()])),
              'outcome': tf.train.Feature(
                  float_list=tf.train.FloatList(value=[value]))
          }))


def write_tf_examples(filename, tf_examples, serialize=True):
  """Write tf.Example to files.

  Args:
    filename: Where to write tf.records
    tf_examples: An iterable of tf.Example
    serialize: whether to serialize the examples.
  """
  with tf.python_io.TFRecordWriter(
      filename, options=TF_RECORD_CONFIG) as writer:
    for ex in tf_examples:
      if serialize:
        writer.write(ex.SerializeToString())
      else:
        writer.write(ex)


# Read tf.Example from files
def _batch_parse_tf_example(board_size, batch_size, example_batch):
  """Parse tf examples.

  Args:
    board_size: the go board size
    batch_size: the batch size
    example_batch: a batch of tf.Example

  Returns:
    A tuple (feature_tensor, dict of output tensors)
  """
  features = {
      'x': tf.FixedLenFeature([], tf.string),
      'pi': tf.FixedLenFeature([], tf.string),
      'outcome': tf.FixedLenFeature([], tf.float32),
  }
  parsed = tf.parse_example(example_batch, features)
  x = tf.decode_raw(parsed['x'], tf.uint8)
  x = tf.cast(x, tf.float32)
  x = tf.reshape(x, [batch_size, board_size, board_size,
                     features_lib.NEW_FEATURES_PLANES])
  pi = tf.decode_raw(parsed['pi'], tf.float32)
  pi = tf.reshape(pi, [batch_size, board_size * board_size + 1])
  outcome = parsed['outcome']
  outcome.set_shape([batch_size])
  return (x, {'pi_tensor': pi, 'value_tensor': outcome})


def read_tf_records(
    shuffle_buffer_size, batch_size, tf_records, num_repeats=None,
    shuffle_records=True, shuffle_examples=True, filter_amount=1.0):
  """Read tf records.

  Args:
    shuffle_buffer_size: how big of a buffer to fill before shuffling
    batch_size: batch size to return
    tf_records: a list of tf_record filenames
    num_repeats: how many times the data should be read (default: infinite)
    shuffle_records: whether to shuffle the order of files read
    shuffle_examples: whether to shuffle the tf.Examples
    filter_amount: what fraction of records to keep

  Returns:
    a tf dataset of batched tensors
  """
  if shuffle_records:
    random.shuffle(tf_records)
  record_list = tf.data.Dataset.from_tensor_slices(tf_records)

  # compression_type here must agree with write_tf_examples
  # cycle_length = how many tfrecord files are read in parallel
  # block_length = how many tf.Examples are read from each file before
  #   moving to the next file
  # The idea is to shuffle both the order of the files being read,
  # and the examples being read from the files.
  dataset = record_list.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'),
      cycle_length=64, block_length=16)
  dataset = dataset.filter(
      lambda x: tf.less(tf.random_uniform([1]), filter_amount)[0])
  # TODO(amj): apply py_func for transforms here.
  if num_repeats is not None:
    dataset = dataset.repeat(num_repeats)
  else:
    dataset = dataset.repeat()
  if shuffle_examples:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
  dataset = dataset.batch(batch_size)
  return dataset


def get_input_tensors(params, batch_size, tf_records, num_repeats=None,
                      shuffle_records=True, shuffle_examples=True,
                      filter_amount=0.05):
  """Read tf.Records and prepare them for ingestion by dualnet.

  Args:
    params: An object of hyperparameters
    batch_size: batch size to return
    tf_records: a list of tf_record filenames
    num_repeats: how many times the data should be read (default: infinite)
    shuffle_records: whether to shuffle the order of files read
    shuffle_examples: whether to shuffle the tf.Examples
    filter_amount: what fraction of records to keep

  Returns:
    A dict of tensors (see return value of batch_parse_tf_example)
  """
  shuffle_buffer_size = params.shuffle_buffer_size
  dataset = read_tf_records(
      shuffle_buffer_size, batch_size, tf_records, num_repeats=num_repeats,
      shuffle_records=shuffle_records, shuffle_examples=shuffle_examples,
      filter_amount=filter_amount)
  dataset = dataset.filter(lambda t: tf.equal(tf.shape(t)[0], batch_size))
  def batch_parse_tf_example(batch_size, dataset):
    return _batch_parse_tf_example(params.board_size, batch_size, dataset)
  dataset = dataset.map(functools.partial(
      batch_parse_tf_example, batch_size))
  return dataset.make_one_shot_iterator().get_next()


# End-to-end utility functions
def make_dataset_from_selfplay(data_extracts, params):
  """Make an iterable of tf.Examples.

  Args:
    data_extracts: An iterable of (position, pi, result) tuples
    params: An object of hyperparameters

  Returns:
    An iterable of tf.Examples.
  """
  board_size = params.board_size
  tf_examples = (make_tf_example(features_lib.extract_features(
      board_size, pos), pi, result) for pos, pi, result in data_extracts)
  return tf_examples


def make_dataset_from_sgf(board_size, sgf_filename, tf_record):
  pwcs = sgf_wrapper.replay_sgf_file(board_size, sgf_filename)
  def make_tf_example_from_pwc(pwcs):
    return _make_tf_example_from_pwc(board_size, pwcs)
  tf_examples = map(make_tf_example_from_pwc, pwcs)
  write_tf_examples(tf_record, tf_examples)


def _make_tf_example_from_pwc(board_size, position_w_context):
  features = features_lib.extract_features(
      board_size, position_w_context.position)
  pi = _one_hot(board_size, coords.to_flat(
      board_size, position_w_context.next_move))
  value = position_w_context.result
  return make_tf_example(features, pi, value)


def shuffle_tf_examples(shuffle_buffer_size, gather_size, records_to_shuffle):
  """Read through tf.Record and yield shuffled, but unparsed tf.Examples.

  Args:
    shuffle_buffer_size: the size for shuffle buffer
    gather_size: The number of tf.Examples to be gathered together
    records_to_shuffle: A list of filenames

  Yields:
    An iterator yielding lists of bytes, which are serialized tf.Examples.
  """
  dataset = read_tf_records(shuffle_buffer_size, gather_size,
                            records_to_shuffle, num_repeats=1)
  batch = dataset.make_one_shot_iterator().get_next()
  sess = tf.Session()
  while True:
    try:
      result = sess.run(batch)
      yield list(result)
    except tf.errors.OutOfRangeError:
      break
