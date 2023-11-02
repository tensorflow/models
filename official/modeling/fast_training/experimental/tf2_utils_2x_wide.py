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

"""Stacking model horizontally."""

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras


def expand_vector(v: np.ndarray) -> np.ndarray:
  """Expands a vector with batch dimensions.

  Equivalent to expand_1_axis(v, epsilon=0.0, axis=-1)

  Args:
    v: A vector with shape [..., a].

  Returns:
    A vector with shape [..., 2 * a].
  """
  return np.repeat(v, 2, axis=-1)


def expand_1_axis(w: np.ndarray,
                  epsilon: float,
                  axis: int) -> np.ndarray:
  """Expands either the first dimension or the last dimension of w.

  If `axis = 0`, the following constraint will be satisfied:
  matmul(x, w) ==
      matmul(expand_vector(x), expand_1_axis(w, epsilon=0.1, axis=0))

  If `axis = -1`, the following constraint will be satisfied if `epsilon = 0.0`:
  expand_vector(matmul(x, w)) ==
      2 * matmul(x, expand_1_axis(w, epsilon=0.0, axis=-1))

  Args:
    w: Numpy array of shape [a_0, a_1, ..., a_i-1, a_i].
    epsilon: Symmetric Noise added to expanded tensor.
    axis: Must be either 0 or -1.

  Returns:
    Expanded numpy array.
  """
  assert axis in (0, -1), (
      "Only support expanding the first or the last dimension. "
      "Got: {}".format(axis))

  rank = len(w.shape)

  d_w = np.random.normal(np.zeros_like(w), np.fabs(w) * epsilon, w.shape)
  d_w = np.repeat(d_w, 2, axis=axis)

  sign_flip = np.array([1, -1])
  for _ in range(rank - 1):
    sign_flip = np.expand_dims(sign_flip, axis=-1 if axis == 0 else 0)
  sign_flip = np.tile(sign_flip,
                      [w.shape[0]] + [1] * (rank - 2) + [w.shape[-1]])

  d_w *= sign_flip
  w_expand = (np.repeat(w, 2, axis=axis) + d_w) / 2
  return w_expand


def expand_2_axes(w: np.ndarray,
                  epsilon: float) -> np.ndarray:
  """Expands the first dimension and the last dimension of w.

  The following constraint will be satisfied:
  expand_vector(matmul(x, w)) == matmul(expand_vector(x), expand_2_axes(w))

  Args:
    w: Numpy array of shape [a_0, a_1, ..., a_i-1, a_i].
    epsilon: Symmetric Noise added to expanded tensor.

  Returns:
    Expanded numpy array.
  """
  rank = len(w.shape)

  d_w = np.random.normal(np.zeros_like(w), np.fabs(w) * epsilon, w.shape)
  d_w = np.repeat(np.repeat(d_w, 2, axis=0), 2, axis=-1)

  sign_flip = np.array([1, -1])
  for _ in range(rank - 1):
    sign_flip = np.expand_dims(sign_flip, axis=-1)
  sign_flip = np.tile(sign_flip,
                      [w.shape[0]] + [1] * (rank - 2) + [w.shape[-1] * 2])
  d_w *= sign_flip

  w_expand = (np.repeat(np.repeat(w, 2, axis=0), 2, axis=-1) + d_w) / 2
  return w_expand


def var_to_var(var_from: tf.Variable,
               var_to: tf.Variable,
               epsilon: float):
  """Expands a variable to another variable.

  Assume the shape of `var_from` is (a, b, ..., y, z), the shape of `var_to`
  can be (a, ..., z * 2), (a * 2, ..., z * 2), (a * 2, ..., z)

  If the shape of `var_to` is (a, ..., 2 * z):
    For any x, tf.matmul(x, var_to) ~= expand_vector(tf.matmul(x, var_from)) / 2
    Not that there will be noise added to the left hand side, if epsilon != 0.
  If the shape of `var_to` is (2 * a, ..., z):
    For any x, tf.matmul(expand_vector(x), var_to) == tf.matmul(x, var_from)
  If the shape of `var_to` is (2 * a, ..., 2 * z):
    For any x, tf.matmul(expand_vector(x), var_to) ==
        expand_vector(tf.matmul(expand_vector(x), var_from))

  Args:
    var_from: input variable to expand.
    var_to: output variable.
    epsilon: the noise ratio that will be added, when splitting `var_from`.
  """
  shape_from = var_from.shape
  shape_to = var_to.shape

  if shape_from == shape_to:
    var_to.assign(var_from)

  elif len(shape_from) == 1 and len(shape_to) == 1:
    var_to.assign(expand_vector(var_from.numpy()))

  elif shape_from[0] * 2 == shape_to[0] and shape_from[-1] == shape_to[-1]:
    var_to.assign(expand_1_axis(var_from.numpy(), epsilon=epsilon, axis=0))

  elif shape_from[0] == shape_to[0] and shape_from[-1] * 2 == shape_to[-1]:
    var_to.assign(expand_1_axis(var_from.numpy(), epsilon=epsilon, axis=-1))

  elif shape_from[0] * 2 == shape_to[0] and shape_from[-1] * 2 == shape_to[-1]:
    var_to.assign(expand_2_axes(var_from.numpy(), epsilon=epsilon))

  else:
    raise ValueError("Shape not supported, {}, {}".format(shape_from, shape_to))


def model_to_model_2x_wide(model_from: tf.Module,
                           model_to: tf.Module,
                           epsilon: float = 0.1):
  """Expands a model to a wider version.

  Also makes sure that the output of the model is not changed after expanding.
  For example:
  ```
  model_narrow = tf_keras.Sequential()
  model_narrow.add(tf_keras.Input(shape=(3,)))
  model_narrow.add(tf_keras.layers.Dense(4))
  model_narrow.add(tf_keras.layers.Dense(1))

  model_wide = tf_keras.Sequential()
  model_wide.add(tf_keras.Input(shape=(6,)))
  model_wide.add(tf_keras.layers.Dense(8))
  model_wide.add(tf_keras.layers.Dense(1))

  model_to_model_2x_wide(model_narrow, model_wide)
  assert model_narrow([[1, 2, 3]]) == model_wide([[1, 1, 2, 2, 3, 3]])
  ```

  We assume that `model_from` and `model_to` has the same architecture and only
  widths of them differ.

  Args:
    model_from: input model to expand.
    model_to: output model whose variables will be assigned expanded values
      according to `model_from`.
    epsilon: the noise ratio that will be added, when splitting `var_from`.
  """
  for w_from, w_to in zip(model_from.trainable_variables,
                          model_to.trainable_variables):
    logging.info("expanding %s %s to %s %s",
                 w_from.name, w_from.shape, w_to.name, w_to.shape)
    var_to_var(w_from, w_to, epsilon=epsilon)
