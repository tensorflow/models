# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Regularizers that group other regularizers for residual connections.

An Elementwise operation between two tensors (addition, multiplication, maximum
etc) imposes a constraint of equality of the shapes of the constituents. For
example, if A, B are convolutions, and another op in the network
receives A + B as input, it means that the i-th output of A is tied to the i-th
output of B. Only if the i-th output was regularized away by the reguarizer in
both A and B can we discard the i-th activation in both.

Therefore we group the i-th output of A and the i-th output of B in a group
LASSO, a group for each i. The grouping methods can vary, and this file offers
several variants.

Residual connections, in ResNet or in RNNs, are examples where this kind of
grouping is needed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import generic_regularizers


DEFAULT_THRESHOLD = 0.01


class MaxGroupingRegularizer(generic_regularizers.OpRegularizer):
  """A regularizer that groups others by taking their maximum."""

  def __init__(self, regularizers_to_group):
    """Creates an instance.

    Args:
      regularizers_to_group: A list of generic_regularizers.OpRegularizer
        objects.Their regularization_vector (alive_vector) are expected to be of
        the same length.

    Raises:
      ValueError: regularizers_to_group is not of length 2 (TODO:
        support arbitrary length if needed.
    """
    _raise_if_length_is_not2(regularizers_to_group)
    self._regularization_vector = tf.maximum(
        regularizers_to_group[0].regularization_vector,
        regularizers_to_group[1].regularization_vector)
    self._alive_vector = tf.logical_or(regularizers_to_group[0].alive_vector,
                                       regularizers_to_group[1].alive_vector)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class L2GroupingRegularizer(generic_regularizers.OpRegularizer):
  r"""A regularizer that groups others by taking their L2 norm.

  R_j = sqrt((\sum_i r_{ij}^2))

  Where r_i is the i-th regularization vector, r_{ij} is its j-th element, and
  R_j is the j-th element of the resulting regularization vector.
  """

  def __init__(self, regularizers_to_group, threshold=DEFAULT_THRESHOLD):
    """Creates an instance.

    Args:
      regularizers_to_group: A list of generic_regularizers.OpRegularizer
        objects.Their regularization_vector (alive_vector) are expected to be of
        the same length.
      threshold: A float. An group of activations will be considered alive if
        its L2 norm is greater than `threshold`.

    Raises:
      ValueError: regularizers_to_group is not of length 2 (TODO:
        support arbitrary length if needed.
    """
    _raise_if_length_is_not2(regularizers_to_group)
    self._regularization_vector = tf.sqrt((
        lazy_square(regularizers_to_group[0].regularization_vector) +
        lazy_square(regularizers_to_group[1].regularization_vector)))
    self._alive_vector = self._regularization_vector > threshold

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


def _raise_if_length_is_not2(regularizers_to_group):
  if len(regularizers_to_group) != 2:
    raise ValueError('Currently only groups of size 2 are supported.')


def lazy_square(tensor):
  """Computes the square of a tensor in a lazy way.

  This function is lazy in the following sense, for:
    tensor = tf.sqrt(input)
  will return input (and not tf.square(tensor)).

  Args:
    tensor: A `Tensor` of floats to compute the square of.

  Returns:
    The squre of the input tensor.
  """
  if tensor.op.type == 'Sqrt':
    return tensor.op.inputs[0]
  else:
    return tf.square(tensor)
