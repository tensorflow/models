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

"""A set of utils for dealing with nested lists and tuples of Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest

def map_nested(map_fn, nested):
  """Executes map_fn on every element in a (potentially) nested structure.

  Args:
    map_fn: A callable to execute on each element in 'nested'.
    nested: A potentially nested combination of sequence objects. Sequence
      objects include tuples, lists, namedtuples, and all subclasses of
      collections.Sequence except strings. See nest.is_sequence for details.
      For example [1, ('hello', 4.3)] is a nested structure containing elements
      1, 'hello', and 4.3.
  Returns:
    out_structure: A potentially nested combination of sequence objects with the
      same structure as the 'nested' input argument. out_structure
      contains the result of applying map_fn to each element in 'nested'. For
      example map_nested(lambda x: x+1, [1, (3, 4.3)]) returns [2, (4, 5.3)].
  """
  out = map(map_fn, nest.flatten(nested))
  return nest.pack_sequence_as(nested, out)


def tile_tensors(tensors, multiples):
  """Tiles a set of Tensors.

  Args:
    tensors: A potentially nested tuple or list of Tensors with rank
      greater than or equal to the length of 'multiples'. The Tensors do not
      need to have the same rank, but their rank must not be dynamic.
    multiples: A python list of ints indicating how to tile each Tensor
      in 'tensors'. Similar to the 'multiples' argument to tf.tile.
  Returns:
    tiled_tensors: A potentially nested tuple or list of Tensors with the same
      structure as the 'tensors' input argument. Contains the result of
      applying tf.tile to each Tensor in 'tensors'. When the rank of a Tensor
      in 'tensors' is greater than the length of multiples, multiples is padded
      at the end with 1s. For example when tiling a 4-dimensional Tensor with
      multiples [3, 4], multiples would be padded to [3, 4, 1, 1] before tiling.
  """
  def tile_fn(x):
    return tf.tile(x, multiples + [1]*(x.shape.ndims - len(multiples)))

  return map_nested(tile_fn, tensors)


def gather_tensors(tensors, indices):
  """Performs a tf.gather operation on a set of Tensors.

  Args:
    tensors: A potentially nested tuple or list of Tensors.
    indices: The indices to use for the gather operation.
  Returns:
    gathered_tensors: A potentially nested tuple or list of Tensors with the
      same structure as the 'tensors' input argument. Contains the result of
      applying tf.gather(x, indices) on each element x in 'tensors'.
  """
  return map_nested(lambda x: tf.gather(x, indices), tensors)


def tas_for_tensors(tensors, length):
  """Unstacks a set of Tensors into TensorArrays.

  Args:
    tensors: A potentially nested tuple or list of Tensors with length in the
      first dimension greater than or equal to the 'length' input argument.
    length: The desired length of the TensorArrays.
  Returns:
    tensorarrays: A potentially nested tuple or list of TensorArrays with the
      same structure as 'tensors'. Contains the result of unstacking each Tensor
      in 'tensors'.
  """
  def map_fn(x):
    ta = tf.TensorArray(x.dtype, length, name=x.name.split(':')[0] + '_ta')
    return ta.unstack(x[:length, :])
  return map_nested(map_fn, tensors)


def read_tas(tas, index):
  """Performs a read operation on a set of TensorArrays.

  Args:
    tas: A potentially nested tuple or list of TensorArrays with length greater
      than 'index'.
    index: The location to read from.
  Returns:
    read_tensors: A potentially nested tuple or list of Tensors with the same
      structure as the 'tas' input argument. Contains the result of
      performing a read operation at 'index' on each TensorArray in 'tas'.
  """
  return map_nested(lambda ta: ta.read(index), tas)
