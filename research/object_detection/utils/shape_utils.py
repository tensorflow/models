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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.utils import static_shape


get_dim_as_int = static_shape.get_dim_as_int


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

  # Computing the padding statically makes the operation work with XLA.
  rank = len(t.get_shape())
  paddings = [[0 for _ in range(2)] for _ in range(rank)]
  t_d0 = tf.shape(t)[0]

  if isinstance(length, int) or len(length.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
    paddings[0][1] = length - t_d0
  else:
    paddings[0][1] = length[0] - t_d0

  return tf.pad(t, paddings)


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
  return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


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


def assert_box_normalized(boxes, maximum_normalized_coordinate=1.1):
  """Asserts the input box tensor is normalized.

  Args:
    boxes: a tensor of shape [N, 4] where N is the number of boxes.
    maximum_normalized_coordinate: Maximum coordinate value to be considered
      as normalized, default to 1.1.

  Returns:
    a tf.Assert op which fails when the input box tensor is not normalized.

  Raises:
    ValueError: When the input box tensor is not normalized.
  """
  box_minimum = tf.reduce_min(boxes)
  box_maximum = tf.reduce_max(boxes)
  return tf.Assert(
      tf.logical_and(
          tf.less_equal(box_maximum, maximum_normalized_coordinate),
          tf.greater_equal(box_minimum, 0)),
      [boxes])


def flatten_dimensions(inputs, first, last):
  """Flattens `K-d` tensor along [first, last) dimensions.

  Converts `inputs` with shape [D0, D1, ..., D(K-1)] into a tensor of shape
  [D0, D1, ..., D(first) * D(first+1) * ... * D(last-1), D(last), ..., D(K-1)].

  Example:
  `inputs` is a tensor with initial shape [10, 5, 20, 20, 3].
  new_tensor = flatten_dimensions(inputs, first=1, last=3)
  new_tensor.shape -> [10, 100, 20, 3].

  Args:
    inputs: a tensor with shape [D0, D1, ..., D(K-1)].
    first: first value for the range of dimensions to flatten.
    last: last value for the range of dimensions to flatten. Note that the last
      dimension itself is excluded.

  Returns:
    a tensor with shape
    [D0, D1, ..., D(first) * D(first + 1) * ... * D(last - 1), D(last), ...,
     D(K-1)].

  Raises:
    ValueError: if first and last arguments are incorrect.
  """
  if first >= inputs.shape.ndims or last > inputs.shape.ndims:
    raise ValueError('`first` and `last` must be less than inputs.shape.ndims. '
                     'found {} and {} respectively while ndims is {}'.format(
                         first, last, inputs.shape.ndims))
  shape = combined_static_and_dynamic_shape(inputs)
  flattened_dim_prod = tf.reduce_prod(shape[first:last],
                                      keepdims=True)
  new_shape = tf.concat([shape[:first], flattened_dim_prod,
                         shape[last:]], axis=0)
  return tf.reshape(inputs, new_shape)


def flatten_first_n_dimensions(inputs, n):
  """Flattens `K-d` tensor along first n dimension to be a `(K-n+1)-d` tensor.

  Converts `inputs` with shape [D0, D1, ..., D(K-1)] into a tensor of shape
  [D0 * D1 * ... * D(n-1), D(n), ... D(K-1)].

  Example:
  `inputs` is a tensor with initial shape [10, 5, 20, 20, 3].
  new_tensor = flatten_first_n_dimensions(inputs, 2)
  new_tensor.shape -> [50, 20, 20, 3].

  Args:
    inputs: a tensor with shape [D0, D1, ..., D(K-1)].
    n: The number of dimensions to flatten.

  Returns:
    a tensor with shape [D0 * D1 * ... * D(n-1), D(n), ... D(K-1)].
  """
  return flatten_dimensions(inputs, first=0, last=n)


def expand_first_dimension(inputs, dims):
  """Expands `K-d` tensor along first dimension to be a `(K+n-1)-d` tensor.

  Converts `inputs` with shape [D0, D1, ..., D(K-1)] into a tensor of shape
  [dims[0], dims[1], ..., dims[-1], D1, ..., D(k-1)].

  Example:
  `inputs` is a tensor with shape [50, 20, 20, 3].
  new_tensor = expand_first_dimension(inputs, [10, 5]).
  new_tensor.shape -> [10, 5, 20, 20, 3].

  Args:
    inputs: a tensor with shape [D0, D1, ..., D(K-1)].
    dims: List with new dimensions to expand first axis into. The length of
      `dims` is typically 2 or larger.

  Returns:
    a tensor with shape [dims[0], dims[1], ..., dims[-1], D1, ..., D(k-1)].
  """
  inputs_shape = combined_static_and_dynamic_shape(inputs)
  expanded_shape = tf.stack(dims + inputs_shape[1:])

  # Verify that it is possible to expand the first axis of inputs.
  assert_op = tf.assert_equal(
      inputs_shape[0], tf.reduce_prod(tf.stack(dims)),
      message=('First dimension of `inputs` cannot be expanded into provided '
               '`dims`'))

  with tf.control_dependencies([assert_op]):
    inputs_reshaped = tf.reshape(inputs, expanded_shape)

  return inputs_reshaped


def resize_images_and_return_shapes(inputs, image_resizer_fn):
  """Resizes images using the given function and returns their true shapes.

  Args:
    inputs: a float32 Tensor representing a batch of inputs of shape
      [batch_size, height, width, channels].
    image_resizer_fn: a function which takes in a single image and outputs
      a resized image and its original shape.

  Returns:
    resized_inputs: The inputs resized according to image_resizer_fn.
    true_image_shapes: A integer tensor of shape [batch_size, 3]
      representing the height, width and number of channels in inputs.
  """

  if inputs.dtype is not tf.float32:
    raise ValueError('`resize_images_and_return_shapes` expects a'
                     ' tf.float32 tensor')

  # TODO(jonathanhuang): revisit whether to always use batch size as
  # the number of parallel iterations vs allow for dynamic batching.
  outputs = static_or_dynamic_map_fn(
      image_resizer_fn,
      elems=inputs,
      dtype=[tf.float32, tf.int32])
  resized_inputs = outputs[0]
  true_image_shapes = outputs[1]

  return resized_inputs, true_image_shapes
