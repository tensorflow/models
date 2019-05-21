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
"""Convenience functions for managing dataset file buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import multiprocessing
import os
import tempfile
import uuid

import numpy as np
import six

import tensorflow as tf


class _GarbageCollector(object):
  """Deletes temporary buffer files at exit.

  Certain tasks (such as NCF Recommendation) require writing buffers to
  temporary files. (Which may be local or distributed.) It is not generally safe
  to delete these files during operation, but they should be cleaned up. This
  class keeps track of temporary files created, and deletes them at exit.
  """
  def __init__(self):
    self.temp_buffers = []

  def register(self, filepath):
    self.temp_buffers.append(filepath)

  def purge(self):
    try:
      for i in self.temp_buffers:
        if tf.gfile.Exists(i):
          tf.gfile.Remove(i)
          tf.logging.info("Buffer file {} removed".format(i))
    except Exception as e:
      tf.logging.error("Failed to cleanup buffer files: {}".format(e))


_GARBAGE_COLLECTOR = _GarbageCollector()
atexit.register(_GARBAGE_COLLECTOR.purge)

_ROWS_PER_CORE = 50000


def write_to_temp_buffer(dataframe, buffer_folder, columns):
  if buffer_folder is None:
    _, buffer_path = tempfile.mkstemp()
  else:
    tf.gfile.MakeDirs(buffer_folder)
    buffer_path = os.path.join(buffer_folder, str(uuid.uuid4()))
  _GARBAGE_COLLECTOR.register(buffer_path)

  return write_to_buffer(dataframe, buffer_path, columns)


def iter_shard_dataframe(df, rows_per_core=1000):
  """Two way shard of a dataframe.

  This function evenly shards a dataframe so that it can be mapped efficiently.
  It yields a list of dataframes with length equal to the number of CPU cores,
  with each dataframe having rows_per_core rows. (Except for the last batch
  which may have fewer rows in the dataframes.) Passing vectorized inputs to
  a multiprocessing pool is much more effecient than iterating through a
  dataframe in serial and passing a list of inputs to the pool.

  Args:
    df: Pandas dataframe to be sharded.
    rows_per_core: Number of rows in each shard.

  Returns:
    A list of dataframe shards.
  """
  n = len(df)
  num_cores = min([multiprocessing.cpu_count(), n])

  num_blocks = int(np.ceil(n / num_cores / rows_per_core))
  max_batch_size = num_cores * rows_per_core
  for i in range(num_blocks):
    min_index = i * max_batch_size
    max_index = min([(i + 1) * max_batch_size, n])
    df_shard = df[min_index:max_index]
    n_shard = len(df_shard)
    boundaries = np.linspace(0, n_shard, num_cores + 1, dtype=np.int64)
    yield [df_shard[boundaries[j]:boundaries[j+1]] for j in range(num_cores)]


def _shard_dict_to_examples(shard_dict):
  """Converts a dict of arrays into a list of example bytes."""
  n = [i for i in shard_dict.values()][0].shape[0]
  feature_list = [{} for _ in range(n)]
  for column, values in shard_dict.items():
    if len(values.shape) == 1:
      values = np.reshape(values, values.shape + (1,))

    if values.dtype.kind == "i":
      feature_map = lambda x: tf.train.Feature(
          int64_list=tf.train.Int64List(value=x))
    elif values.dtype.kind == "f":
      feature_map = lambda x: tf.train.Feature(
          float_list=tf.train.FloatList(value=x))
    else:
      raise ValueError("Invalid dtype")
    for i in range(n):
      feature_list[i][column] = feature_map(values[i])
  examples = [
      tf.train.Example(features=tf.train.Features(feature=example_features))
      for example_features in feature_list
  ]

  return [e.SerializeToString() for e in examples]


def _serialize_shards(df_shards, columns, pool, writer):
  """Map sharded dataframes to bytes, and write them to a buffer.

  Args:
    df_shards: A list of pandas dataframes. (Should be of similar size)
    columns: The dataframe columns to be serialized.
    pool: A multiprocessing pool to serialize in parallel.
    writer: A TFRecordWriter to write the serialized shards.
  """
  # Pandas does not store columns of arrays as nd arrays. stack remedies this.
  map_inputs = [{c: np.stack(shard[c].values, axis=0) for c in columns}
                for shard in df_shards]

  # Failure within pools is very irksome. Thus, it is better to thoroughly check
  # inputs in the main process.
  for inp in map_inputs:
    # Check that all fields have the same number of rows.
    assert len(set([v.shape[0] for v in inp.values()])) == 1
    for val in inp.values():
      assert hasattr(val, "dtype")
      assert hasattr(val.dtype, "kind")
      assert val.dtype.kind in ("i", "f")
      assert len(val.shape) in (1, 2)
  shard_bytes = pool.map(_shard_dict_to_examples, map_inputs)
  for s in shard_bytes:
    for example in s:
      writer.write(example)

def write_to_buffer(dataframe, buffer_path, columns, expected_size=None):
  """Write a dataframe to a binary file for a dataset to consume.

  Args:
    dataframe: The pandas dataframe to be serialized.
    buffer_path: The path where the serialized results will be written.
    columns: The dataframe columns to be serialized.
    expected_size: The size in bytes of the serialized results. This is used to
      lazily construct the buffer.

  Returns:
    The path of the buffer.
  """
  if tf.gfile.Exists(buffer_path) and tf.gfile.Stat(buffer_path).length > 0:
    actual_size = tf.gfile.Stat(buffer_path).length
    if expected_size == actual_size:
      return buffer_path
    tf.logging.warning(
        "Existing buffer {} has size {}. Expected size {}. Deleting and "
        "rebuilding buffer.".format(buffer_path, actual_size, expected_size))
    tf.gfile.Remove(buffer_path)

  if dataframe is None:
    raise ValueError(
        "dataframe was None but a valid existing buffer was not found.")

  tf.gfile.MakeDirs(os.path.split(buffer_path)[0])

  tf.logging.info("Constructing TFRecordDataset buffer: {}".format(buffer_path))

  count = 0
  pool = multiprocessing.Pool(multiprocessing.cpu_count())
  try:
    with tf.python_io.TFRecordWriter(buffer_path) as writer:
      for df_shards in iter_shard_dataframe(df=dataframe,
                                            rows_per_core=_ROWS_PER_CORE):
        _serialize_shards(df_shards, columns, pool, writer)
        count += sum([len(s) for s in df_shards])
        tf.logging.info("{}/{} examples written."
                        .format(str(count).ljust(8), len(dataframe)))
  finally:
    pool.terminate()

  tf.logging.info("Buffer write complete.")
  return buffer_path
