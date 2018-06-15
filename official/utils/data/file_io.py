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

import multiprocessing
import os
import tempfile
import uuid

import numpy as np
import six

import tensorflow as tf


class _GarbageCollector(object):
  def __init__(self):
    self.temp_buffers = []

  def register(self, filepath):
    self.temp_buffers.append(filepath)

  def __del__(self):
    for i in self.temp_buffers:
      if tf.gfile.Exists(i):
        tf.gfile.Remove(i)

_GARBAGE_COLLECTOR = _GarbageCollector()

# More powerful machines benefit from larger scale maps.
_DISK_WRITE_THRESHOLD = int(125000  * multiprocessing.cpu_count())


def write_to_temp_buffer(dataframe, buffer_folder, columns):
  if buffer_folder is None:
    _, buffer_path = tempfile.mkstemp()
  else:
    tf.gfile.MakeDirs(buffer_folder)
    buffer_path = os.path.join(buffer_folder, str(uuid.uuid4()))
  _GARBAGE_COLLECTOR.register(buffer_path)
  write_to_buffer(dataframe, buffer_path, columns)
  return buffer_path


def _to_bytes(key_value_list):
  features = {}
  for key, value in key_value_list:
    if isinstance(value, int):
      features[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, float):
      features[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    elif isinstance(value, np.ndarray) and value.dtype.kind == "i":
      value = value.astype(np.int64)
      assert len(value.shape) == 1
      features[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=value))
    elif isinstance(value, (six.text_type, six.binary_type)):
      if not isinstance(value, six.binary_type):
        value = value.encode("utf-8")
      features[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[value]))
    else:
      raise ValueError("Unknown type.")
  example = tf.train.Example(features=tf.train.Features(feature=features))
  example_bytes = example.SerializeToString()
  return example_bytes


def _write_to_disk(key_value_list, pool, writer):
    serialized_rows = pool.map(_to_bytes, key_value_list)
    for example_bytes in serialized_rows:
      writer.write(example_bytes)


def write_to_buffer(dataframe, buffer_path, columns, expected_size=None):
  """Write a dataframe to a binary file for a dataset to consume."""
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

  with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    with tf.python_io.TFRecordWriter(buffer_path) as writer:
      data_rows = []
      count = 0
      for row in dataframe.itertuples():
        data_rows.append([(key, getattr(row, key)) for key in columns])
        count += 1
        # It is necessary to periodically process the accumulated data to avoid
        # excessive memory consumption.
        if len(data_rows) == _DISK_WRITE_THRESHOLD:
          _write_to_disk(data_rows, pool, writer)
          data_rows = []
          tf.logging.info("{}/{} examples written."
                          .format(str(count).ljust(8), len(dataframe)))
      _write_to_disk(data_rows, pool, writer)
      tf.logging.info("Buffer write complete.")

  return buffer_path
