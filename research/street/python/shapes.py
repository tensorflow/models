# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Shape manipulation functions.

rotate_dimensions: prepares for a rotating transpose by returning a rotated
  list of dimension indices.
transposing_reshape: allows a dimension to be factorized, with one of the pieces
  transferred to another dimension, or to transpose factors within a single
  dimension.
tensor_dim: gets a shape dimension as a constant integer if known otherwise a
  runtime usable tensor value.
tensor_shape: returns the full shape of a tensor as the tensor_dim.
"""
import tensorflow as tf


def rotate_dimensions(num_dims, src_dim, dest_dim):
  """Returns a list of dimension indices that will rotate src_dim to dest_dim.

  src_dim is moved to dest_dim, with all intervening dimensions shifted towards
  the hole left by src_dim. Eg:
  num_dims = 4, src_dim=3, dest_dim=1
  Returned list=[0, 3, 1, 2]
  For a tensor with dims=[5, 4, 3, 2] a transpose would yield [5, 2, 4, 3].
  Args:
    num_dims: The number of dimensions to handle.
    src_dim:  The dimension to move.
    dest_dim: The dimension to move src_dim to.

  Returns:
    A list of rotated dimension indices.
  """
  # List of dimensions for transpose.
  dim_list = range(num_dims)
  # Shuffle src_dim to dest_dim by swapping to shuffle up the other dims.
  step = 1 if dest_dim > src_dim else -1
  for x in xrange(src_dim, dest_dim, step):
    dim_list[x], dim_list[x + step] = dim_list[x + step], dim_list[x]
  return dim_list


def transposing_reshape(tensor,
                        src_dim,
                        part_a,
                        part_b,
                        dest_dim_a,
                        dest_dim_b,
                        name=None):
  """Splits src_dim and sends one of the pieces to another dim.

  Terminology:
  A matrix is often described as 'row-major' or 'column-major', which doesn't
  help if you can't remember which is the row index and which is the column,
  even if you know what 'major' means, so here is a simpler explanation of it:
  When TF stores a tensor of size [d0, d1, d2, d3] indexed by [i0, i1, i2, i3],
  the memory address of an element is calculated using:
  ((i0 * d1 + i1) * d2 + i2) * d3 + i3, so, d0 is the MOST SIGNIFICANT dimension
  and d3 the LEAST SIGNIFICANT, just like in the decimal number 1234, 1 is the
  most significant digit and 4 the least significant. In both cases the most
  significant is multiplied by the largest number to determine its 'value'.
  Furthermore, if we reshape the tensor to [d0'=d0, d1'=d1 x d2, d2'=d3], then
  the MOST SIGNIFICANT part of d1' is d1 and the LEAST SIGNIFICANT part of d1'
  is d2.

  Action:
  transposing_reshape splits src_dim into factors [part_a, part_b], and sends
  the most significant part (of size  part_a) to be the most significant part of
  dest_dim_a*(Exception: see NOTE 2), and the least significant part (of size
  part_b) to be the most significant part of dest_dim_b.
  This is basically a combination of reshape, rotating transpose, reshape.
  NOTE1: At least one of dest_dim_a and dest_dim_b must equal src_dim, ie one of
  the parts always stays put, so src_dim is never totally destroyed and the
  output number of dimensions is always the same as the input.
  NOTE2: If dest_dim_a == dest_dim_b == src_dim, then parts a and b are simply
  transposed within src_dim to become part_b x part_a, so the most significant
  part becomes the least significant part and vice versa. Thus if you really
  wanted to make one of the parts the least significant side of the destiantion,
  the destination dimension can be internally transposed with a second call to
  transposing_reshape.
  NOTE3: One of part_a and part_b may be -1 to allow src_dim to be of unknown
  size with one known-size factor. Otherwise part_a * part_b must equal the size
  of src_dim.
  NOTE4: The reshape preserves as many known-at-graph-build-time dimension sizes
  as are available.

  Example:
  Input dims=[5, 2, 6, 2]
  tensor=[[[[0, 1][2, 3][4, 5][6, 7][8, 9][10, 11]]
           [[12, 13][14, 15][16, 17][18, 19][20, 21][22, 23]]
          [[[24, 25]...
  src_dim=2, part_a=2, part_b=3, dest_dim_a=3, dest_dim_b=2
  output dims =[5, 2, 3, 4]
  output tensor=[[[[0, 1, 6, 7][2, 3, 8, 9][4, 5, 10, 11]]
                  [[12, 13, 18, 19][14, 15, 20, 21][16, 17, 22, 23]]]
                 [[[24, 26, 28]...
  Example2:
  Input dims=[phrases, words, letters]=[2, 6, x]
  tensor=[[[the][cat][sat][on][the][mat]]
         [[a][stitch][in][time][saves][nine]]]
  We can factorize the 6 words into 3x2 = [[the][cat]][[sat][on]][[the][mat]]
  or 2x3=[[the][cat][sat]][[on][the][mat]] and
  src_dim=1, part_a=3, part_b=2, dest_dim_a=1, dest_dim_b=1
  would yield:
  [[[the][sat][the][cat][on][mat]]
   [[a][in][saves][stitch][time][nine]]], but
  src_dim=1, part_a=2, part_b=3, dest_dim_a=1, dest_dim_b=1
  would yield:
  [[[the][on][cat][the][sat][mat]]
   [[a][time][stitch][saves][in][nine]]], and
  src_dim=1, part_a=2, part_b=3, dest_dim_a=0, dest_dim_b=1
  would yield:
  [[[the][cat][sat]]
   [[a][stitch][in]]
   [[on][the][mat]]
   [[time][saves][nine]]]
  Now remember that the words above represent any least-significant subset of
  the input dimensions.

  Args:
    tensor:     A tensor to reshape.
    src_dim:    The dimension to split.
    part_a:     The first factor of the split.
    part_b:     The second factor of the split.
    dest_dim_a: The dimension to move part_a of src_dim to.
    dest_dim_b: The dimension to move part_b of src_dim to.
    name:       Optional base name for all the ops.

  Returns:
    Reshaped tensor.

  Raises:
    ValueError: If the args are invalid.
  """
  if dest_dim_a != src_dim and dest_dim_b != src_dim:
    raise ValueError(
        'At least one of dest_dim_a, dest_dim_b must equal src_dim!')
  if part_a == 0 or part_b == 0:
    raise ValueError('Zero not allowed for part_a or part_b!')
  if part_a < 0 and part_b < 0:
    raise ValueError('At least one of part_a and part_b must be positive!')
  if not name:
    name = 'transposing_reshape'
  prev_shape = tensor_shape(tensor)
  expanded = tf.reshape(
      tensor,
      prev_shape[:src_dim] + [part_a, part_b] + prev_shape[src_dim + 1:],
      name=name + '_reshape_in')
  dest = dest_dim_b
  if dest_dim_a != src_dim:
    # We are just moving part_a to dest_dim_a.
    dest = dest_dim_a
  else:
    # We are moving part_b to dest_dim_b.
    src_dim += 1
  dim_list = rotate_dimensions(len(expanded.get_shape()), src_dim, dest)
  expanded = tf.transpose(expanded, dim_list, name=name + '_rot_transpose')
  # Reshape identity except dest,dest+1, which get merged.
  ex_shape = tensor_shape(expanded)
  combined = ex_shape[dest] * ex_shape[dest + 1]
  return tf.reshape(
      expanded,
      ex_shape[:dest] + [combined] + ex_shape[dest + 2:],
      name=name + '_reshape_out')


def tensor_dim(tensor, dim):
  """Returns int dimension if known at a graph build time else a tensor.

  If the size of the dim of tensor is known at graph building time, then that
  known value is returned, otherwise (instead of None), a Tensor that will give
  the size of the dimension when the graph is run. The return value will be
  accepted by tf.reshape in multiple (or even all) dimensions, even when the
  sizes are not known at graph building time, unlike -1, which can only be used
  in one dimension. It is a bad idea to use tf.shape all the time, as some ops
  demand a known (at graph build time) size. This function therefore returns
  the best available, most useful dimension size.
  Args:
    tensor: Input tensor.
    dim:    Dimension to find the size of.

  Returns:
    An integer if shape is known at build time, otherwise a tensor of int32.
  """
  result = tensor.get_shape().as_list()[dim]
  if result is None:
    result = tf.shape(tensor)[dim]
  return result


def tensor_shape(tensor):
  """Returns a heterogeneous list of tensor_dim for the tensor.

  See tensor_dim for a more detailed explanation.
  Args:
    tensor: Input tensor.

  Returns:
    A heterogeneous list of integers and int32 tensors.
  """
  result = []
  for d in xrange(len(tensor.get_shape())):
    result.append(tensor_dim(tensor, d))
  return result
