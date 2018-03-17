# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Utils used to manipulate tensor shapes."""

import tensorflow as tf

from object_detection.utils import static_shape


def _is_tensor(t):
  """Returns a boolean indicating whether the input is a tensor.

  Args:
    t: the input to be tested.

  Returns:
    a boolean that indicates whether t is a tensor.
  """
  return isinstance(t, (tf.Tensor, tf.SparseTensor, tf.Variable))


def _set_dim_0(t, d0):
  """Sets the 0-th dimension of the input tensor.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    d0: an integer indicating the 0-th dimension of the input tensor.

  Returns:
    the tensor t with the 0-th dimension set.
  """
  t_shape = t.get_shape().as_list()
  t_shape[0] = d0
  t.set_shape(t_shape)
  return t


def pad_tensor(t, length):
  """Pads the input tensor with 0s along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after padding, assuming length <= t.shape[0].

  Returns:
    padded_t: the padded tensor, whose first dimension is length. If the length
      is an integer, the first dimension of padded_t is set to length
      statically.
  """
  t_rank = tf.rank(t)
  t_shape = tf.shape(t)
  t_d0 = t_shape[0]
  pad_d0 = tf.expand_dims(length - t_d0, 0)
  pad_shape = tf.cond(
      tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
      lambda: tf.expand_dims(length - t_d0, 0))
  padded_t = tf.concat([t, tf.zeros(pad_shape, dtype=t.dtype)], 0)
  if not _is_tensor(length):
    padded_t = _set_dim_0(padded_t, length)
  return padded_t


def clip_tensor(t, length):
  """Clips the input tensor along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after clipping, assuming length <= t.shape[0].

  Returns:
    clipped_t: the clipped tensor, whose first dimension is length. If the
      length is an integer, the first dimension of clipped_t is set to length
      statically.
  """
  clipped_t = tf.gather(t, tf.range(length))
  if not _is_tensor(length):
    clipped_t = _set_dim_0(clipped_t, length)
  return clipped_t


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after processing.

  Returns:
    processed_t: the processed tensor, whose first dimension is length. If the
      length is an integer, the first dimension of the processed tensor is set
      to length statically.
  """
  processed_t = tf.cond(
      tf.greater(tf.shape(t)[0], length),
      lambda: clip_tensor(t, length),
      lambda: pad_tensor(t, length))
  if not _is_tensor(length):
    processed_t = _set_dim_0(processed_t, length)
  return processed_t


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')


def check_min_image_dim(min_dim, image_tensor):
  """Checks that the image width/height are greater than some number.

  This function is used to check that the width and height of an image are above
  a certain value. If the image shape is static, this function will perform the
  check at graph construction time. Otherwise, if the image shape varies, an
  Assertion control dependency will be added to the graph.

  Args:
    min_dim: The minimum number of pixels along the width and height of the
             image.
    image_tensor: The image tensor to check size for.

  Returns:
    If `image_tensor` has dynamic size, return `image_tensor` with a Assert
    control dependency. Otherwise returns image_tensor.

  Raises:
    ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
  """
  image_shape = image_tensor.get_shape()
  image_height = static_shape.get_height(image_shape)
  image_width = static_shape.get_width(image_shape)
  if image_height is None or image_width is None:
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                       tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
        ['image size must be >= {} in both height and width.'.format(min_dim)])
    with tf.control_dependencies([shape_assert]):
      return tf.identity(image_tensor)

  if image_height < min_dim or image_width < min_dim:
    raise ValueError(
        'image size must be >= %d in both height and width; image dim = %d,%d' %
        (min_dim, image_height, image_width))

  return image_tensor


def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a, shape_b)


def assert_shape_equal_along_first_dimension(shape_a, shape_b):
  """Asserts that shape_a and shape_b are the same along the 0th-dimension.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
    if shape_a[0] != shape_b[0]:
      raise ValueError('Unequal first dimension {}, {}'.format(
          shape_a[0], shape_b[0]))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a[0], shape_b[0])

