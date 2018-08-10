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
"""A regularizer for convolutions, based on group-lasso.

All the weights that are related to a single output are grouped into one LASSO
group (https://arxiv.org/pdf/1611.06321.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from morph_net.framework import generic_regularizers


class ConvGroupLassoRegularizer(generic_regularizers.OpRegularizer):
  """A regularizer for convolutions, based on group-lasso.

  Supported ops: Conv2D and Conv2DBackpropInput (transposed Conv2D).
  are supported. The grouping is done according to the formula:

  (1 - l1_fraction) * L2(weights) / sqrt(dim) + l1_fraction * L1(weights) / dim,

  where `dim` is the number of weights associated with an activation, L2 and L1
  are the respective norms, and l1_fraction controls the balance between L1 and
  L2 grouping. The paper cited above experiments with 0.0 and 0.5 for
  l1_fraction.
  """

  def __init__(self, op, threshold, l1_fraction=0.0):
    """Creates an instance.

    Args:
      op: A tf.Operation object of type Conv2D or Conv2DBackpropInput.
      threshold: A float. When the norm of the group associated with an
        activation is below the threshold, it will be considered dead.
      l1_fraction: A float, controls the balance between L1 and L2 grouping
        (see above).

    Raises:
      ValueError: `op` is not of type 'Conv2D' or 'Conv2DBackpropInput', or
        l1_fraction is outside interval [0.0, 1.0].
    """
    if op.type not in ('Conv2D', 'Conv2DBackpropInput'):
      raise ValueError('The given op is not Conv2D or Conv2DBackpropInput.')
    if l1_fraction < 0.0 or l1_fraction > 1.0:
      raise ValueError(
          'l1_fraction should be in [0.0, 1.0], not %e.' % l1_fraction)

    self._threshold = threshold
    conv_weights = op.inputs[1]
    # For a Conv2D (Conv2DBackpropInput) the output dimension of the weight
    # matrix is 3 (2). We thus reduce over all other dimensions.

    l2_norm = tf.sqrt(
        tf.reduce_mean(tf.square(conv_weights), axis=_get_reduce_dims(op)))
    if l1_fraction > 0.0:
      l1_norm = tf.reduce_mean(tf.abs(conv_weights), axis=_get_reduce_dims(op))
      norm = l1_fraction * l1_norm + (1.0 - l1_fraction) * l2_norm
    else:
      norm = l2_norm
    # Sanity check: Output dimension of 'op' should match that of 'norm':
    assert op.outputs[0].shape.ndims == 4
    assert norm.shape.ndims == 1
    op.outputs[0].shape.dims[3].assert_is_compatible_with(norm.shape.dims[0])
    self._regularization_vector = norm
    self._alive_vector = norm > threshold

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class ConvGroupLassoRegularizerFactory(object):
  """A class for creating a ConvGroupLassoRegularizer for convolutions."""

  def __init__(self, threshold, l1_fraction=0.0):
    """Creates an instance.

    Args:
      threshold: A float scalar, will be used as a threshold for all
        ConvGroupLassoRegularizer-s created by this class.
      l1_fraction: A float scalar, will be passed as l1_fraction to all
        ConvGroupLassoRegularizer-s created by this class.
    """
    self._threshold = threshold
    self._l1_fraction = l1_fraction

  def create_regularizer(self, op, opreg_manager=None):
    """Creates a ConvGroupLassoRegularizer for `op`.

    Args:
      op: A tf.Operation of type 'Conv2D'.
      opreg_manager: unused

    Returns:
      a ConvGroupLassoRegularizer that corresponds to `op`.
    """
    del opreg_manager  # unused
    return ConvGroupLassoRegularizer(op, self._threshold, self._l1_fraction)


def _get_reduce_dims(op):
  """Returns the reduction dimensions for grouping weights of various ops."""
  type_to_dims = {'Conv2D': (0, 1, 2), 'Conv2DBackpropInput': (0, 1, 3)}
  try:
    return type_to_dims[op.type]
  except KeyError:
    raise ValueError('Reduce dims are unknown for op type %s' % op.type)
