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

import os

import numpy as np
import six

import tensorflow as tf

def write_to_buffer(dataframe, buffer_path, columns, expected_size=None):
  """Write a dataframe to a binary file for a dataset to consume."""
  if tf.gfile.Exists(buffer_path):
    actual_size = tf.gfile.Stat(buffer_path).length
    if expected_size == actual_size:
      return
    tf.logging.warning(
        "Existing buffer {} has size {}. Expected size {}. Deleting and "
        "rebuilding buffer.".format(buffer_path, actual_size, expected_size))
    tf.gfile.Remove(buffer_path)

  if dataframe is None:
    raise ValueError(
        "dataframe was None but a valid existing buffer was not found.")

  tf.gfile.MakeDirs(os.path.split(buffer_path)[0])

  tf.logging.info("Constructing {}".format(buffer_path))
  with tf.python_io.TFRecordWriter(buffer_path) as writer:
    for row in dataframe.itertuples():
      i = getattr(row, "Index")
      features = {}
      for key in columns:
        value = getattr(row, key)
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
      writer.write(example.SerializeToString())
      if (i + 1) % 50000 == 0:
        tf.logging.info(
            "{}/{} examples written.".format(str(i+1).ljust(8), len(dataframe)))


