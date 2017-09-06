# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""A trainable ADAM optimizer that learns its internal variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from learned_optimizer.optimizer import trainable_optimizer as opt
from learned_optimizer.optimizer import utils


class TrainableAdam(opt.TrainableOptimizer):
  """Adam optimizer with learnable scalar parameters.

  See Kingma et. al., 2014 for algorithm (http://arxiv.org/abs/1412.6980).
  """

  def __init__(self,
               learning_rate=1e-3,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               **kwargs):
    """Initializes the TrainableAdam optimizer with the given initial values.

    Args:
      learning_rate: The learning rate (default: 1e-3).
      beta1: The exponential decay rate for the 1st moment estimates.
      beta2: The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      **kwargs: Any additional keyword arguments for TrainableOptimizer.

    Raises:
      ValueError: if the learning rate or epsilon is not positive
      ValueError: if beta1 or beta2 is not in (0, 1).
    """
    if learning_rate <= 0:
      raise ValueError("Learning rate must be positive.")
    if epsilon <= 0:
      raise ValueError("Epsilon must be positive.")
    if not 0 < beta1 < 1 or not 0 < beta2 < 1:
      raise ValueError("Beta values must be between 0 and 1, exclusive.")

    self._reuse_vars = False

    with tf.variable_scope(opt.OPTIMIZER_SCOPE):
      def inv_sigmoid(x):
        return np.log(x / (1.0 - x))

      self.log_learning_rate = tf.get_variable(
          "log_learning_rate",
          shape=[],
          initializer=tf.constant_initializer(np.log(learning_rate)))
      self.beta1_logit = tf.get_variable(
          "beta1_logit",
          shape=[],
          initializer=tf.constant_initializer(inv_sigmoid(beta1)))
      self.beta2_logit = tf.get_variable(
          "beta2_logit",
          shape=[],
          initializer=tf.constant_initializer(inv_sigmoid(beta2)))
      self.log_epsilon = tf.get_variable(
          "log_epsilon",
          shape=[],
          initializer=tf.constant_initializer(np.log(epsilon)))

    # Key names are derived from Algorithm 1 described in
    # https://arxiv.org/pdf/1412.6980.pdf
    state_keys = ["m", "v", "t"]
    super(TrainableAdam, self).__init__("Adam", state_keys, **kwargs)

  def _initialize_state(self, var):
    """Returns a dictionary mapping names of state variables to their values."""
    vectorized_shape = var.get_shape().num_elements(), 1

    return {key: tf.zeros(vectorized_shape) for key in self.state_keys}

  def _compute_update(self, param, grad, state):
    """Calculates the new internal state and parameters.

    If the gradient is sparse, updates the appropriate slices in the internal
    state and stacks the update tensor.

    Args:
      param: A tensor of parameters.
      grad: A tensor of gradients with the same shape as param.
      state: A dictionary containing any state for the optimizer.

    Returns:
      updated_param: The updated parameters.
      updated_state: The updated state variables in a dictionary.
    """

    with tf.variable_scope(opt.OPTIMIZER_SCOPE) as scope:

      if self._reuse_vars:
        scope.reuse_variables()
      else:
        self._reuse_vars = True

      (grad_values, first_moment, second_moment, timestep, grad_indices
      ) = self._extract_gradients_and_internal_state(
          grad, state, tf.shape(param))

      beta1 = tf.nn.sigmoid(self.beta1_logit)
      beta2 = tf.nn.sigmoid(self.beta2_logit)
      epsilon = tf.exp(self.log_epsilon) + 1e-10
      learning_rate = tf.exp(self.log_learning_rate)

      old_grad_shape = tf.shape(grad_values)
      grad_values = tf.reshape(grad_values, [-1, 1])

      new_timestep = timestep + 1
      new_first_moment = self._update_adam_estimate(
          first_moment, grad_values, beta1)
      new_second_moment = self._debias_adam_estimate(
          second_moment, tf.square(grad_values), beta2)

      debiased_first_moment = self._debias_adam_estimate(
          new_first_moment, beta1, new_timestep)
      debiased_second_moment = self._debias_adam_estimate(
          new_second_moment, beta2, new_timestep)

      # Propagating through the square root of 0 is very bad for stability.
      update = (learning_rate * debiased_first_moment /
                (tf.sqrt(debiased_second_moment + 1e-10) + epsilon))

      update = tf.reshape(update, old_grad_shape)

      if grad_indices is not None:
        param_shape = tf.shape(param)
        update = utils.stack_tensor(
            update, grad_indices, param, param_shape[:1])
        new_first_moment = utils.update_slices(
            new_first_moment, grad_indices, state["m"], param_shape)
        new_second_moment = utils.update_slices(
            new_second_moment, grad_indices, state["v"], param_shape)
        new_timestep = utils.update_slices(
            new_timestep, grad_indices, state["t"], param_shape)

      new_param = param - update

      # collect the update and new state
      new_state = {
          "m": new_first_moment,
          "v": new_second_moment,
          "t": new_timestep
      }

    return new_param, new_state

  def _update_adam_estimate(self, estimate, value, beta):
    """Returns a beta-weighted average of estimate and value."""
    return (beta * estimate) + ((1 - beta) * value)

  def _debias_adam_estimate(self, estimate, beta, t_step):
    """Returns a debiased estimate based on beta and the timestep."""
    return estimate / (1 - tf.pow(beta, t_step))

  def _extract_gradients_and_internal_state(self, grad, state, param_shape):
    """Extracts the gradients and relevant internal state.

    If the gradient is sparse, extracts the appropriate slices from the state.

    Args:
      grad: The current gradient.
      state: The current state.
      param_shape: The shape of the parameter (used if gradient is sparse).

    Returns:
      grad_values: The gradient value tensor.
      first_moment: The first moment tensor (internal state).
      second_moment: The second moment tensor (internal state).
      timestep: The current timestep (internal state).
      grad_indices: The indices for the gradient tensor, if sparse.
          None otherwise.
    """
    grad_values = grad
    grad_indices = None
    first_moment = state["m"]
    second_moment = state["v"]
    timestep = state["t"]

    if isinstance(grad, tf.IndexedSlices):
      grad_indices, grad_values = utils.accumulate_sparse_gradients(grad)
      first_moment = utils.slice_tensor(
          first_moment, grad_indices, param_shape)
      second_moment = utils.slice_tensor(
          second_moment, grad_indices, param_shape)
      timestep = utils.slice_tensor(timestep, grad_indices, param_shape)

    return grad_values, first_moment, second_moment, timestep, grad_indices

