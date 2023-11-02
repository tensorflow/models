# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions."""

import collections
from typing import List, Optional, Union

import tensorflow as tf, tf_keras


def resolve_shape(
    tensor: tf.Tensor,
    resolve_batch_size: bool = True) -> List[Union[tf.Tensor, int]]:
  """Fully resolves the shape of the tensor.

  Args:
    tensor: The tensor for which to resolve the shape.
    resolve_batch_size: If True, fully resolve the batch size. If False,
      return the batch size if it is statically known and -1 otherwise. This
      can be more efficient when converting a model to TFLite.

  Returns:
    A list containing the static dimension where possible and the dynamic
    dimension otherwise.
  """
  with tf.name_scope('resolve_shape'):
    shape = tensor.get_shape().as_list()
    if None in shape:
      shape_dynamic = tf.shape(tensor)
      if shape[0] is None:
        shape[0] = shape_dynamic[0] if resolve_batch_size else -1
      for i in range(1, len(shape)):
        if shape[i] is None:
          shape[i] = shape_dynamic[i]
    return shape


def set_shape_dim(tensor: tf.Tensor, index: int, size: int) -> None:
  """Set value of index-th element of tensor shape to size."""
  shape = tensor.get_shape().as_list()
  if len(shape) <= index:
    raise ValueError(
        'Tensor rank must be at least %d. Got %d' % (index + 1, len(shape)))
  shape[index] = size
  tensor.set_shape(shape)


def truncate_or_pad(input_tensor: tf.Tensor,
                    new_size: int,
                    axis: int = 1,
                    constant_value: Union[int, float] = 0) -> tf.Tensor:
  """Truncate or zeros pad the axis of input tensor to new size."""
  rank = len(input_tensor.shape)

  if rank <= axis:
    raise ValueError(
        'Tensor rank must be at least %d. Got %d' % (axis + 1, rank))

  orig_size = tf.shape(input_tensor)[axis]

  def _new_size(dim):
    if dim == axis:
      return new_size
    n = tf.shape(input_tensor)[dim]
    return -1 if n is None else n

  def _truncate():
    begin = [0] * rank
    size = [_new_size(dim) for dim in range(rank)]
    return tf.slice(input_tensor, begin, size)

  def _pad():
    padding = [[0, 0] for _ in range(rank)]
    padding[axis][1] = new_size - orig_size
    return tf.pad(input_tensor, padding, constant_values=constant_value)

  output = tf.cond(orig_size >= new_size, _truncate, _pad)
  if isinstance(new_size, int):
    set_shape_dim(output, axis, new_size)
  return output


def rotate_rboxes90(rboxes: tf.Tensor,
                    image_width: int,
                    image_height: int,
                    rotation_count: int = 1) -> tf.Tensor:
  """Rotate oriented rectangles counter-clockwise by multiples of 90 degrees."""
  image_width = tf.cast(image_width, dtype=tf.float32)
  image_height = tf.cast(image_height, dtype=tf.float32)

  rotation_count = rotation_count % 4
  x, y, w, h, angle = tf.split(rboxes, 5, axis=1)

  if rotation_count == 0:
    return rboxes
  elif rotation_count == 1:
    angle = tf.where(angle < -90.0, angle + 270, angle - 90)
    return tf.concat([y, image_width - x - 1, w, h, angle], axis=1)
  elif rotation_count == 2:
    angle = tf.where(angle < 0.0, angle + 180, angle - 180)
    return tf.concat([image_width - x - 1, image_height - y - 1, w, h, angle],
                     axis=1)
  else:
    angle = tf.where(angle > 90.0, angle - 270, angle + 90)
    return tf.concat([image_height - y - 1, x, w, h, angle], axis=1)


def normalize_image_to_range(image: tf.Tensor,
                             original_minval: int = 0,
                             original_maxval: int = 255,
                             target_minval: float = -1.0,
                             target_maxval: float = 1.0) -> tf.Tensor:
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to the [target_minval, target_maxval] range.

  Args:
    image: A tensor of shape [height, width, channels]. Input will be converted
      to float32 type before normalization.
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    A float tensor with the same shape as the input image.
  """
  if image.dtype is not tf.float32:
    image = tf.cast(image, dtype=tf.float32)

  original_minval = float(original_minval)
  original_maxval = float(original_maxval)
  target_minval = float(target_minval)
  target_maxval = float(target_maxval)
  image = tf.cast(image, dtype=tf.float32)
  image = tf.subtract(image, original_minval)
  image = tf.multiply(image, (target_maxval - target_minval) /
                      (original_maxval - original_minval))
  image = tf.add(image, target_minval)

  return image


def get_padding_mask_from_valid_lengths(
    valid_lengths: tf.Tensor,
    max_length: Optional[int] = None,
    dtype: tf.dtypes.DType = tf.bool) -> tf.Tensor:
  """Gets a 2D mask of the padded region from valid lengths.

  Args:
    valid_lengths: A 1D int tensor containing the valid length of each row.
    max_length: (optional, int) The maximum length of each row. If `None`, the
      maximum value in `valid_lengths` will be used.
    dtype: The output dtype.

  Returns:
    2D padded region mask.
  """
  with tf.name_scope('get_padding_mask_from_valid_lengths'):
    if max_length is None:
      max_length = tf.reduce_max(valid_lengths)
    padding_mask = tf.logical_not(tf.sequence_mask(valid_lengths, max_length))

    return tf.cast(padding_mask, dtype=dtype)


def get_transformer_attention_bias(padding_mask: tf.Tensor) -> tf.Tensor:
  """Gets attention bias.

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_attention_heads, max_length, max_length].
  The tensor is zero at non-padded locations, and -1e9 (negative infinity) at
  padded locations.

  Args:
    padding_mask: A [batch_size, max_length] float tensor, the padding mask.

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, max_length].
  """
  with tf.name_scope('attention_bias'):
    # Uses -1e9 to represent -infinity. We do not actually use -Inf, since we
    # want to be able to multiply these values by zero to get zero.
    # (-Inf * 0 = NaN)
    attention_bias = padding_mask * -1e9
    attention_bias = tf.expand_dims(
        tf.expand_dims(attention_bias, axis=1), axis=1)

  return attention_bias


class DisjointSet:
  """A disjoint set implementation."""

  def __init__(self, num_elements: int):
    self._num_elements = num_elements
    self._parent = list(range(num_elements))

  def find(self, item: int) -> int:
    if self._parent[item] == item:
      return item
    else:
      self._parent[item] = self.find(self._parent[item])
      return self._parent[item]

  def union(self, i1: int, i2: int) -> None:
    r1 = self.find(i1)
    r2 = self.find(i2)
    self._parent[r1] = r2

  def to_group(self) -> List[List[int]]:
    """Return the grouping results.

    Returns:
        A list of integer lists. Each list represents the IDs belonging to the
      same group.
    """
    groups = collections.defaultdict(list)
    for i in range(self._num_elements):
      r = self.find(i)
      groups[r].append(i)
    return list(groups.values())
