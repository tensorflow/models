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
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_shape[index])
  return combined_shape
