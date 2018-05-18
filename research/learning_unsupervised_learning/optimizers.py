# Copyright 2018 Google, Inc. All Rights Reserved.
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



"""Optimizers for use in unrolled optimization.

These optimizers contain a compute_updates function and its own ability to keep
track of internal state.
These functions can be used with a tf.while_loop to perform multiple training
steps per sess.run.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import tensorflow as tf
import sonnet as snt

from learning_unsupervised_learning import utils

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class UnrollableOptimizer(snt.AbstractModule):
  """Interface for optimizers that can be used in unrolled computation.
  apply_gradients is derrived from compute_update and assign_state.
  """

  def __init__(self, *args, **kwargs):
    super(UnrollableOptimizer, self).__init__(*args, **kwargs)
    self()

  @abc.abstractmethod
  def compute_updates(self, xs, gs, state=None):
    """Compute next step updates for a given variable list and state.

    Args:
      xs: list of tensors
        The "variables" to perform an update on.
        Note these must match the same order for which get_state was originally
        called.
      gs: list of tensors
        Gradients of `xs` with respect to some loss.
      state: Any
        Optimizer specific state to keep track of accumulators such as momentum
        terms
    """
    raise NotImplementedError()

  def _build(self):
    pass

  @abc.abstractmethod
  def get_state(self, var_list):
    """Get the state value associated with a list of tf.Variables.

    This state is commonly going to be a NamedTuple that contains some
    mapping between variables and the state associated with those variables.
    This state could be a moving momentum variable tracked by the optimizer.

    Args:
        var_list: list of tf.Variable
    Returns:
      state: Any
        Optimizer specific state
    """
    raise NotImplementedError()

  def assign_state(self, state):
    """Assigns the state to the optimizers internal variables.

    Args:
      state: Any
    Returns:
      op: tf.Operation
        The operation that performs the assignment.
    """
    raise NotImplementedError()

  def apply_gradients(self, grad_vars):
    gradients, variables = zip(*grad_vars)
    state = self.get_state(variables)
    new_vars, new_state = self.compute_updates(variables, gradients, state)
    assign_op = self.assign_state(new_state)
    op = utils.assign_variables(variables, new_vars)
    return tf.group(assign_op, op, name="apply_gradients")


class UnrollableGradientDescentRollingOptimizer(UnrollableOptimizer):

  def __init__(self,
               learning_rate,
               name="UnrollableGradientDescentRollingOptimizer"):
    self.learning_rate = learning_rate
    super(UnrollableGradientDescentRollingOptimizer, self).__init__(name=name)


  def compute_updates(self, xs, gs, learning_rates, state):
    new_vars = []
    for x, g, lr in utils.eqzip(xs, gs, learning_rates):
      if lr is None:
        lr = self.learning_rate
      if g is not None:
        new_vars.append((x * (1 - lr) - g * lr))
      else:
        new_vars.append(x)
    return new_vars, state

  def get_state(self, var_list):
    return tf.constant(0.0)

  def assign_state(self, state, var_list=None):
    return tf.no_op()
