# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

LARGE_NUM = 1. / tf.keras.backend.epsilon()


def _smallest_positive_root(a, b, c) -> tf.Tensor:
  """
    Returns the smallest positive root of a quadratic equation.
    This implements the fixed version in
    https://github.com/princeton-vl/CornerNet.
  """
  
  discriminant = b ** 2 - 4 * a * c
  discriminant_sqrt = tf.sqrt(discriminant)
  
  root1 = (-b - discriminant_sqrt) / (2 * a)
  root2 = (-b + discriminant_sqrt) / (2 * a)
  
  return tf.where(tf.less(discriminant, 0), tf.cast(LARGE_NUM, b.dtype),
                  (-b + discriminant_sqrt) / (2))


def gaussian_radius(det_size, min_overlap=0.7) -> int:
  """
    Given a bounding box size, returns a lower bound on how far apart the
    corners of another bounding box can lie while still maintaining the given
    minimum overlap, or IoU. Modified from implementation found in
    https://github.com/tensorflow/models/blob/master/research/object_detection/core/target_assigner.py.

    Params:
        det_size (tuple): tuple of integers representing height and width
        min_overlap (tf.float32): minimum IoU desired
    Returns:
        int representing desired gaussian radius
    """
  height, width = det_size[0], det_size[1]
  
  # Case where detected box is offset from ground truth and no box completely
  # contains the other.
  
  a1 = 1
  b1 = -(height + width)
  c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
  r1 = _smallest_positive_root(a1, b1, c1)
  
  # Case where detection is smaller than ground truth and completely contained
  # in it.
  
  a2 = 4
  b2 = -2 * (height + width)
  c2 = (1 - min_overlap) * width * height
  r2 = _smallest_positive_root(a2, b2, c2)
  
  # Case where ground truth is smaller than detection and completely contained
  # in it.
  
  a3 = 4 * min_overlap
  b3 = 2 * min_overlap * (height + width)
  c3 = (min_overlap - 1) * width * height
  r3 = _smallest_positive_root(a3, b3, c3)
  # TODO discuss whether to return scalar or tensor
  
  return tf.reduce_min([r1, r2, r3], axis=0)


def _gaussian_penalty(radius: int, dtype=tf.float32) -> tf.Tensor:
  """
  This represents the penalty reduction around a point.
  Params:
      radius (int): integer for radius of penalty reduction
      type (tf.dtypes.DType): datatype of returned tensor
  Returns:
      tf.Tensor of shape (2 * radius + 1, 2 * radius + 1).
  """
  width = 2 * radius + 1
  sigma = tf.cast(radius / 3, dtype=dtype)
  
  range_width = tf.range(width)
  range_width = tf.cast(range_width - tf.expand_dims(radius, axis=-1),
                        dtype=dtype)
  
  x = tf.expand_dims(range_width, axis=-1)
  y = tf.expand_dims(range_width, axis=-2)
  
  exponent = ((-1 * (x ** 2) - (y ** 2)) / (2 * sigma ** 2))
  return tf.math.exp(exponent)


@tf.function
def cartesian_product(*tensors, repeat=1):
  """
  Equivalent of itertools.product except for TensorFlow tensors.

  Example:
    cartesian_product(tf.range(3), tf.range(4))

    array([[0, 0],
       [0, 1],
       [0, 2],
       [0, 3],
       [1, 0],
       [1, 1],
       [1, 2],
       [1, 3],
       [2, 0],
       [2, 1],
       [2, 2],
       [2, 3]], dtype=int32)>

  Params:
    tensors (list[tf.Tensor]): a list of 1D tensors to compute the product of
    repeat (int): number of times to repeat the tensors
      (https://docs.python.org/3/library/itertools.html#itertools.product)

  Returns:
    An nD tensor where n is the number of tensors
  """
  tensors = tensors * repeat
  return tf.reshape(tf.transpose(tf.stack(tf.meshgrid(*tensors, indexing='ij')),
                                 [*[i + 1 for i in range(len(tensors))], 0]),
                    (-1, len(tensors)))


def write_all(ta, index, values):
  for i in range(tf.shape(values)[0]):
    ta = ta.write(index + i, values[i, ...])
  return ta, index + i


@tf.function
def draw_gaussian(hm_shape, blob, dtype, scaling_factor=1):
  """ Draws an instance of a 2D gaussian on a heatmap.
  
  A heatmap with shape hm_shape and of type dtype is generated with 
  a gaussian with a given center, radius, and scaling factor

  Args:
    hm_shape: A `list` of `Tensor` of shape [3] that gives the height, width, 
      and number of channels in the heatmap
    blob: A `Tensor` of shape [4] that gives the channel number, x, y, and
      radius for the desired gaussian to be drawn onto
    dtype: The desired type of the heatmap
    scaling_factor: A `int` that can be used to scale the magnitude of the 
      gaussian
  Returns:
    A `Tensor` with shape hm_shape and type dtype with a 2D gaussian
  """
  gaussian_heatmap = tf.zeros(shape=hm_shape, dtype=dtype)
  
  blob = tf.cast(blob, tf.int32)
  obj_class, x, y, radius = blob[0], blob[1], blob[2], blob[3]
  
  height, width = hm_shape[0], hm_shape[1]
  
  left = tf.math.minimum(x, radius)
  right = tf.math.minimum(width - x, radius + 1)
  top = tf.math.minimum(y, radius)
  bottom = tf.math.minimum(height - y, radius + 1)
  
  gaussian = _gaussian_penalty(radius=radius, dtype=dtype)
  gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  gaussian = tf.reshape(gaussian, [-1])
  
  heatmap_indices = cartesian_product(
      tf.range(y - top, y + bottom), tf.range(x - left, x + right), [obj_class])
  gaussian_heatmap = tf.tensor_scatter_nd_update(
      gaussian_heatmap, heatmap_indices, gaussian * scaling_factor)
  
  return gaussian_heatmap
