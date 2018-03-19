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

"""Collection of trainable optimizers for meta-optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from learned_optimizer.optimizer import utils
from learned_optimizer.optimizer import trainable_optimizer as opt


# Default was 1e-3
tf.app.flags.DEFINE_float("crnn_rnn_readout_scale", 0.5,
                          """The initialization scale for the RNN readouts.""")
tf.app.flags.DEFINE_float("crnn_default_decay_var_init", 2.2,
                          """The default initializer value for any decay/
                             momentum style variables and constants.
                             sigmoid(2.2) ~ 0.9, sigmoid(-2.2) ~ 0.01.""")

FLAGS = tf.flags.FLAGS


class CoordinatewiseRNN(opt.TrainableOptimizer):
  """RNN that operates on each coordinate of the problem independently."""

  def __init__(self,
               cell_sizes,
               cell_cls,
               init_lr_range=(1., 1.),
               dynamic_output_scale=True,
               learnable_decay=True,
               zero_init_lr_weights=False,
               **kwargs):
    """Initializes the RNN per-parameter optimizer.

    Args:
      cell_sizes: List of hidden state sizes for each RNN cell in the network
      cell_cls: tf.contrib.rnn class for specifying the RNN cell type
      init_lr_range: the range in which to initialize the learning rates.
      dynamic_output_scale: whether to learn weights that dynamically modulate
          the output scale (default: True)
      learnable_decay: whether to learn weights that dynamically modulate the
          input scale via RMS style decay (default: True)
      zero_init_lr_weights: whether to initialize the lr weights to zero
      **kwargs: args passed to TrainableOptimizer's constructor

    Raises:
      ValueError: If the init lr range is not of length 2.
      ValueError: If the init lr range is not a valid range (min > max).
    """
    if len(init_lr_range) != 2:
      raise ValueError(
          "Initial LR range must be len 2, was {}".format(len(init_lr_range)))
    if init_lr_range[0] > init_lr_range[1]:
      raise ValueError("Initial LR range min is greater than max.")
    self.init_lr_range = init_lr_range

    self.zero_init_lr_weights = zero_init_lr_weights
    self.reuse_vars = False

    # create the RNN cell
    with tf.variable_scope(opt.OPTIMIZER_SCOPE):
      self.component_cells = [cell_cls(sz) for sz in cell_sizes]
      self.cell = tf.contrib.rnn.MultiRNNCell(self.component_cells)

      # random normal initialization scaled by the output size
      scale_factor = FLAGS.crnn_rnn_readout_scale / math.sqrt(cell_sizes[-1])
      scaled_init = tf.random_normal_initializer(0., scale_factor)

      # weights for projecting the hidden state to a parameter update
      self.update_weights = tf.get_variable("update_weights",
                                            shape=(cell_sizes[-1], 1),
                                            initializer=scaled_init)

      self._initialize_decay(learnable_decay, (cell_sizes[-1], 1), scaled_init)

      self._initialize_lr(dynamic_output_scale, (cell_sizes[-1], 1),
                          scaled_init)

      state_size = sum([sum(state_size) for state_size in self.cell.state_size])
      self._init_vector = tf.get_variable(
          "init_vector", shape=[1, state_size],
          initializer=tf.random_uniform_initializer(-1., 1.))

    state_keys = ["rms", "rnn", "learning_rate", "decay"]
    super(CoordinatewiseRNN, self).__init__("cRNN", state_keys, **kwargs)

  def _initialize_decay(
      self, learnable_decay, weights_tensor_shape, scaled_init):
    """Initializes the decay weights and bias variables or tensors.

    Args:
      learnable_decay: Whether to use learnable decay.
      weights_tensor_shape: The shape the weight tensor should take.
      scaled_init: The scaled initialization for the weights tensor.
    """
    if learnable_decay:

      # weights for projecting the hidden state to the RMS decay term
      self.decay_weights = tf.get_variable("decay_weights",
                                           shape=weights_tensor_shape,
                                           initializer=scaled_init)
      self.decay_bias = tf.get_variable(
          "decay_bias", shape=(1,),
          initializer=tf.constant_initializer(
              FLAGS.crnn_default_decay_var_init))
    else:
      self.decay_weights = tf.zeros_like(self.update_weights)
      self.decay_bias = tf.constant(FLAGS.crnn_default_decay_var_init)

  def _initialize_lr(
      self, dynamic_output_scale, weights_tensor_shape, scaled_init):
    """Initializes the learning rate weights and bias variables or tensors.

    Args:
      dynamic_output_scale: Whether to use a dynamic output scale.
      weights_tensor_shape: The shape the weight tensor should take.
      scaled_init: The scaled initialization for the weights tensor.
    """
    if dynamic_output_scale:
      zero_init = tf.constant_initializer(0.)
      wt_init = zero_init if self.zero_init_lr_weights else scaled_init
      self.lr_weights = tf.get_variable("learning_rate_weights",
                                        shape=weights_tensor_shape,
                                        initializer=wt_init)
      self.lr_bias = tf.get_variable("learning_rate_bias", shape=(1,),
                                     initializer=zero_init)
    else:
      self.lr_weights = tf.zeros_like(self.update_weights)
      self.lr_bias = tf.zeros([1, 1])

  def _initialize_state(self, var):
    """Return a dictionary mapping names of state variables to their values."""
    vectorized_shape = [var.get_shape().num_elements(), 1]

    min_lr = self.init_lr_range[0]
    max_lr = self.init_lr_range[1]
    if min_lr == max_lr:
      init_lr = tf.constant(min_lr, shape=vectorized_shape)
    else:
      actual_vals = tf.random_uniform(vectorized_shape,
                                      np.log(min_lr),
                                      np.log(max_lr))
      init_lr = tf.exp(actual_vals)

    ones = tf.ones(vectorized_shape)
    rnn_init = ones * self._init_vector

    return {
        "rms": tf.ones(vectorized_shape),
        "learning_rate": init_lr,
        "rnn": rnn_init,
        "decay": tf.ones(vectorized_shape),
    }

  def _compute_update(self, param, grad, state):
    """Update parameters given the gradient and state.

    Args:
      param: tensor of parameters
      grad: tensor of gradients with the same shape as param
      state: a dictionary containing any state for the optimizer

    Returns:
      updated_param: updated parameters
      updated_state: updated state variables in a dictionary
    """

    with tf.variable_scope(opt.OPTIMIZER_SCOPE) as scope:

      if self.reuse_vars:
        scope.reuse_variables()
      else:
        self.reuse_vars = True

      param_shape = tf.shape(param)

      (grad_values, decay_state, rms_state, rnn_state, learning_rate_state,
       grad_indices) = self._extract_gradients_and_internal_state(
           grad, state, param_shape)

      # Vectorize and scale the gradients.
      grad_scaled, rms = utils.rms_scaling(grad_values, decay_state, rms_state)

      # Apply the RNN update.
      rnn_state_tuples = self._unpack_rnn_state_into_tuples(rnn_state)
      rnn_output, rnn_state_tuples = self.cell(grad_scaled, rnn_state_tuples)
      rnn_state = self._pack_tuples_into_rnn_state(rnn_state_tuples)

      # Compute the update direction (a linear projection of the RNN output).
      delta = utils.project(rnn_output, self.update_weights)

      # The updated decay is an affine projection of the hidden state
      decay = utils.project(rnn_output, self.decay_weights,
                            bias=self.decay_bias, activation=tf.nn.sigmoid)

      # Compute the change in learning rate (an affine projection of the RNN
      # state, passed through a 2x sigmoid, so the change is bounded).
      learning_rate_change = 2. * utils.project(rnn_output, self.lr_weights,
                                                bias=self.lr_bias,
                                                activation=tf.nn.sigmoid)

      # Update the learning rate.
      new_learning_rate = learning_rate_change * learning_rate_state

      # Apply the update to the parameters.
      update = tf.reshape(new_learning_rate * delta, tf.shape(grad_values))

      if isinstance(grad, tf.IndexedSlices):
        update = utils.stack_tensor(update, grad_indices, param,
                                    param_shape[:1])
        rms = utils.update_slices(rms, grad_indices, state["rms"], param_shape)
        new_learning_rate = utils.update_slices(new_learning_rate, grad_indices,
                                                state["learning_rate"],
                                                param_shape)
        rnn_state = utils.update_slices(rnn_state, grad_indices, state["rnn"],
                                        param_shape)
        decay = utils.update_slices(decay, grad_indices, state["decay"],
                                    param_shape)

      new_param = param - update

      # Collect the update and new state.
      new_state = {
          "rms": rms,
          "learning_rate": new_learning_rate,
          "rnn": rnn_state,
          "decay": decay,
      }

    return new_param, new_state

  def _extract_gradients_and_internal_state(self, grad, state, param_shape):
    """Extracts the gradients and relevant internal state.

    If the gradient is sparse, extracts the appropriate slices from the state.

    Args:
      grad: The current gradient.
      state: The current state.
      param_shape: The shape of the parameter (used if gradient is sparse).

    Returns:
      grad_values: The gradient value tensor.
      decay_state: The current decay state.
      rms_state: The current rms state.
      rnn_state: The current state of the internal rnns.
      learning_rate_state: The current learning rate state.
      grad_indices: The indices for the gradient tensor, if sparse.
          None otherwise.
    """
    if isinstance(grad, tf.IndexedSlices):
      grad_indices, grad_values = utils.accumulate_sparse_gradients(grad)
      decay_state = utils.slice_tensor(state["decay"], grad_indices,
                                       param_shape)
      rms_state = utils.slice_tensor(state["rms"], grad_indices, param_shape)
      rnn_state = utils.slice_tensor(state["rnn"], grad_indices, param_shape)
      learning_rate_state = utils.slice_tensor(state["learning_rate"],
                                               grad_indices, param_shape)
      decay_state.set_shape([None, 1])
      rms_state.set_shape([None, 1])
    else:
      grad_values = grad
      grad_indices = None

      decay_state = state["decay"]
      rms_state = state["rms"]
      rnn_state = state["rnn"]
      learning_rate_state = state["learning_rate"]
    return (grad_values, decay_state, rms_state, rnn_state, learning_rate_state,
            grad_indices)

  def _unpack_rnn_state_into_tuples(self, rnn_state):
    """Creates state tuples from the rnn state vector."""
    rnn_state_tuples = []
    cur_state_pos = 0
    for cell in self.component_cells:
      total_state_size = sum(cell.state_size)
      cur_state = tf.slice(rnn_state, [0, cur_state_pos],
                           [-1, total_state_size])
      cur_state_tuple = tf.split(value=cur_state, num_or_size_splits=2,
                                 axis=1)
      rnn_state_tuples.append(cur_state_tuple)
      cur_state_pos += total_state_size
    return rnn_state_tuples

  def _pack_tuples_into_rnn_state(self, rnn_state_tuples):
    """Creates a single state vector concatenated along column axis."""
    rnn_state = None
    for new_state_tuple in rnn_state_tuples:
      new_c, new_h = new_state_tuple
      if rnn_state is None:
        rnn_state = tf.concat([new_c, new_h], axis=1)
      else:
        rnn_state = tf.concat([rnn_state, tf.concat([new_c, new_h], 1)], axis=1)
    return rnn_state

