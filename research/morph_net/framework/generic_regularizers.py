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
"""Interface for MorphNet regularizers framework.

A subclasses of Regularizer represent a regularizer that targets a certain
quantity: Number of flops, model size, number of activations etc. The
Regularizer interface has two methods:

1. `get_regularization_term`, which returns a regularization term that should be
   included in the total loss to target the quantity.

2. `get_cost`, the quantity itself (for example, the number of flops). This is
   useful for display in TensorBoard, and later, to to provide feedback for
   automatically tuning the coefficient that multplies the regularization term,
   until the cost reaches (or goes below) its target value.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class OpRegularizer(object):
  """An interface for Op Regularizers.

  An OpRegularizer object corresponds to a tf.Operation, and provides
  a regularizer for the output of the op (we assume that the op has one output
  of interest in the context of MorphNet).
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def regularization_vector(self):
    """Returns a vector of floats, with regularizers.

    The length of the vector is the number of "output activations" (call them
    neurons, nodes, filters etc) of the op. For a convolutional network, it's
    the number of filters (aka "depth"). For a fully-connected layer, it's
    usually the second (and last) dimension - assuming the first one is the
    batch size.
    """
    pass

  @abc.abstractproperty
  def alive_vector(self):
    """Returns a vector of booleans, indicating which activations are alive.

    (call them activations, neurons, nodes, filters etc). This vector is of the
    same length as the regularization_vector.
    """
    pass


class NetworkRegularizer(object):
  """An interface for Network Regularizers."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_regularization_term(self, ops=None):
    """Compute the regularization term.

    Args:
      ops: A list of tf.Operation objects. If specified, only the regularization
        term associated with the ops in `ops` will be returned. Otherwise, all
        relevant ops in the default TensorFlow graph will be included.

    Returns:
      A tf.Tensor scalar of floating point type that evaluates to the
      regularization term (that should be added to the total loss, with a
      suitable coefficient)
    """
    pass

  @abc.abstractmethod
  def get_cost(self, ops=None):
    """Calculates the cost targeted by the Regularizer.

    Args:
      ops: A list of tf.Operation objects. If specified, only the cost
        pertaining to the ops in the `ops` will be returned. Otherwise, all
        relevant ops in the default TensorFlow graph will be included.

    Returns:
      A tf.Tensor scalar that evaluates to the cost.
    """
    pass


def dimensions_are_compatible(op_regularizer):
  """Checks if op_regularizer's alive_vector matches regularization_vector."""
  return op_regularizer.alive_vector.shape.with_rank(1).dims[
      0].is_compatible_with(
          op_regularizer.regularization_vector.shape.with_rank(1).dims[0])
