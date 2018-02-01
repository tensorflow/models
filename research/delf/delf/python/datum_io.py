# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Python interface for DatumProto.

DatumProto is protocol buffer used to serialize tensor with arbitrary shape.
Please refer to datum.proto for details.

Support read and write of DatumProto from/to numpy array and file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from delf import datum_pb2
import numpy as np
import tensorflow as tf


def ArrayToDatum(arr):
  """Converts numpy array to DatumProto.

  Args:
    arr: Numpy array of arbitrary shape.

  Returns:
    datum: DatumProto object.
  """
  datum = datum_pb2.DatumProto()
  datum.float_list.value.extend(arr.astype(float).flat)
  datum.shape.dim.extend(arr.shape)
  return datum


def DatumToArray(datum):
  """Converts data saved in DatumProto to numpy array.

  Args:
    datum: DatumProto object.

  Returns:
    Numpy array of arbitrary shape.
  """
  return np.array(datum.float_list.value).astype(float).reshape(datum.shape.dim)


def SerializeToString(arr):
  """Converts numpy array to serialized DatumProto.

  Args:
    arr: Numpy array of arbitrary shape.

  Returns:
    Serialized DatumProto string.
  """
  datum = ArrayToDatum(arr)
  return datum.SerializeToString()


def ParseFromString(string):
  """Converts serialized DatumProto string to numpy array.

  Args:
    string: Serialized DatumProto string.

  Returns:
    Numpy array.
  """
  datum = datum_pb2.DatumProto()
  datum.ParseFromString(string)
  return DatumToArray(datum)


def ReadFromFile(file_path):
  """Helper function to load data from a DatumProto format in a file.

  Args:
    file_path: Path to file containing data.

  Returns:
    data: Numpy array.
  """
  with tf.gfile.FastGFile(file_path, 'r') as f:
    return ParseFromString(f.read())


def WriteToFile(data, file_path):
  """Helper function to write data to a file in DatumProto format.

  Args:
    data: Numpy array.
    file_path: Path to file that will be written.
  """
  serialized_data = SerializeToString(data)
  with tf.gfile.FastGFile(file_path, 'w') as f:
    f.write(serialized_data)
