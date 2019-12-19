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

Support read and write of DatumProto from/to NumPy array and file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from delf import datum_pb2


def ArrayToDatum(arr):
  """Converts NumPy array to DatumProto.

  Supports arrays of types:
    - float16 (it is converted into a float32 in DatumProto)
    - float32
    - float64 (it is converted into a float32 in DatumProto)
    - uint8 (it is converted into a uint32 in DatumProto)
    - uint16 (it is converted into a uint32 in DatumProto)
    - uint32
    - uint64 (it is converted into a uint32 in DatumProto)

  Args:
    arr: NumPy array of arbitrary shape.

  Returns:
    datum: DatumProto object.

  Raises:
    ValueError: If array type is unsupported.
  """
  datum = datum_pb2.DatumProto()
  if arr.dtype in ('float16', 'float32', 'float64'):
    datum.float_list.value.extend(arr.astype('float32').flat)
  elif arr.dtype in ('uint8', 'uint16', 'uint32', 'uint64'):
    datum.uint32_list.value.extend(arr.astype('uint32').flat)
  else:
    raise ValueError('Unsupported array type: %s' % arr.dtype)

  datum.shape.dim.extend(arr.shape)
  return datum


def ArraysToDatumPair(arr_1, arr_2):
  """Converts numpy arrays to DatumPairProto.

  Supports same formats as `ArrayToDatum`, see documentation therein.

  Args:
    arr_1: NumPy array of arbitrary shape.
    arr_2: NumPy array of arbitrary shape.

  Returns:
    datum_pair: DatumPairProto object.
  """
  datum_pair = datum_pb2.DatumPairProto()
  datum_pair.first.CopyFrom(ArrayToDatum(arr_1))
  datum_pair.second.CopyFrom(ArrayToDatum(arr_2))

  return datum_pair


def DatumToArray(datum):
  """Converts data saved in DatumProto to NumPy array.

  Args:
    datum: DatumProto object.

  Returns:
    NumPy array of arbitrary shape.
  """
  if datum.HasField('float_list'):
    return np.array(datum.float_list.value).astype('float32').reshape(
        datum.shape.dim)
  elif datum.HasField('uint32_list'):
    return np.array(datum.uint32_list.value).astype('uint32').reshape(
        datum.shape.dim)
  else:
    raise ValueError('Input DatumProto does not have float_list or uint32_list')


def DatumPairToArrays(datum_pair):
  """Converts data saved in DatumPairProto to NumPy arrays.

  Args:
    datum_pair: DatumPairProto object.

  Returns:
    Two NumPy arrays of arbitrary shape.
  """
  first_datum = DatumToArray(datum_pair.first)
  second_datum = DatumToArray(datum_pair.second)
  return first_datum, second_datum


def SerializeToString(arr):
  """Converts NumPy array to serialized DatumProto.

  Args:
    arr: NumPy array of arbitrary shape.

  Returns:
    Serialized DatumProto string.
  """
  datum = ArrayToDatum(arr)
  return datum.SerializeToString()


def SerializePairToString(arr_1, arr_2):
  """Converts pair of NumPy arrays to serialized DatumPairProto.

  Args:
    arr_1: NumPy array of arbitrary shape.
    arr_2: NumPy array of arbitrary shape.

  Returns:
    Serialized DatumPairProto string.
  """
  datum_pair = ArraysToDatumPair(arr_1, arr_2)
  return datum_pair.SerializeToString()


def ParseFromString(string):
  """Converts serialized DatumProto string to NumPy array.

  Args:
    string: Serialized DatumProto string.

  Returns:
    NumPy array.
  """
  datum = datum_pb2.DatumProto()
  datum.ParseFromString(string)
  return DatumToArray(datum)


def ParsePairFromString(string):
  """Converts serialized DatumPairProto string to NumPy arrays.

  Args:
    string: Serialized DatumProto string.

  Returns:
    Two NumPy arrays.
  """
  datum_pair = datum_pb2.DatumPairProto()
  datum_pair.ParseFromString(string)
  return DatumPairToArrays(datum_pair)


def ReadFromFile(file_path):
  """Helper function to load data from a DatumProto format in a file.

  Args:
    file_path: Path to file containing data.

  Returns:
    data: NumPy array.
  """
  with tf.gfile.GFile(file_path, 'rb') as f:
    return ParseFromString(f.read())


def ReadPairFromFile(file_path):
  """Helper function to load data from a DatumPairProto format in a file.

  Args:
    file_path: Path to file containing data.

  Returns:
    Two NumPy arrays.
  """
  with tf.gfile.GFile(file_path, 'rb') as f:
    return ParsePairFromString(f.read())


def WriteToFile(data, file_path):
  """Helper function to write data to a file in DatumProto format.

  Args:
    data: NumPy array.
    file_path: Path to file that will be written.
  """
  serialized_data = SerializeToString(data)
  with tf.gfile.GFile(file_path, 'w') as f:
    f.write(serialized_data)


def WritePairToFile(arr_1, arr_2, file_path):
  """Helper function to write pair of arrays to a file in DatumPairProto format.

  Args:
    arr_1: NumPy array of arbitrary shape.
    arr_2: NumPy array of arbitrary shape.
    file_path: Path to file that will be written.
  """
  serialized_data = SerializePairToString(arr_1, arr_2)
  with tf.gfile.GFile(file_path, 'w') as f:
    f.write(serialized_data)
