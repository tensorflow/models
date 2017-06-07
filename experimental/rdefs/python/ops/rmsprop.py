# Copyright 2016 Google Inc. All Rights Reserved.
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
"""RMSProp for score function gradients and IndexedSlices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf


def _gradients_per_example(loss, variable):
  """Returns per-example gradients.

  Args:
    loss: A [n_samples, batch_size] shape tensor
    variable: A variable to optimize of shape var_shape

  Returns:
    grad: A tensor of shape [n_samples, *var_shape]
  """
  grad_list = [tf.gradients(loss_sample, variable)[0] for loss_sample in
               tf.unpack(loss)]
  if isinstance(grad_list[0], tf.IndexedSlices):
    grad = tf.pack([g.values for g in grad_list])
    grad = tf.IndexedSlices(values=grad, indices=grad_list[0].indices)
  else:
    grad = tf.pack(grad_list)
  return grad


def _cov(a, b):
  """Calculates covariance between a and b."""
  v = (a - tf.reduce_mean(a, 0)) * (b - tf.reduce_mean(b, 0))
  return tf.reduce_mean(v, 0)


def _var(a):
  """Returns the variance across the sample dimension."""
  _, var = tf.nn.moments(a, [0])
  return var


def _update_mean_square(mean_square, variable):
  """Update mean square for a variable."""
  if isinstance(variable, tf.IndexedSlices):
    square_sum = tf.reduce_sum(tf.square(variable.values), 0)
    mean_square_lookup = tf.nn.embedding_lookup(mean_square, variable.indices)
    moving_mean_square = 0.9 * mean_square_lookup + 0.1 * square_sum
    return tf.scatter_update(mean_square, variable.indices, moving_mean_square)
  else:
    square_sum = tf.reduce_sum(tf.square(variable), 0)
    moving_mean_square = 0.9 * mean_square + 0.1 * square_sum
  return tf.assign(mean_square, square_sum)


def _get_mean_square(variable):
  with tf.variable_scope('optimizer_state'):
    mean_square = tf.get_variable(name=variable.name[:-2],
                                  shape=variable.get_shape(),
                                  initializer=tf.ones_initializer,
                                  dtype=variable.dtype.base_dtype)
  return mean_square


def _control_variate(grad, learning_signal):
  if isinstance(grad, tf.IndexedSlices):
    grad = grad.values
  cov = _cov(grad * learning_signal, grad)
  var = _var(grad)
  return cov / var


def _rmsprop_maximize(learning_rate, learning_signal, log_prob, variable,
                      clip_min=None, clip_max=None):
  """Builds rmsprop maximization ops for a single variable."""
  grad = _gradients_per_example(log_prob, variable)
  if learning_signal.get_shape().ndims == 2:
    # if we have multiple samples of latent variables, need to broadcast
    # grad of shape [n_samples_latents, batch_size, n_timesteps, z_dim]
    # with learning_signal of shape [n_samples_latents, batch_size]:
    learning_signal = tf.expand_dims(tf.expand_dims(learning_signal, 2), 2)
  control_variate = _control_variate(grad, learning_signal)
  mean_square = _get_mean_square(variable)
  update_mean_square = _update_mean_square(mean_square, grad)
  variance_reduced_learning_sig = learning_signal - control_variate
  update_name = variable.name[:-2] + '/score_function_grad_estimator'
  if isinstance(grad, tf.IndexedSlices):
    mean_square_lookup = tf.nn.embedding_lookup(mean_square, grad.indices)
    mean_square_lookup = tf.expand_dims(mean_square_lookup, 0)

    update_per_sample = (grad.values / tf.sqrt(mean_square_lookup)
                         * variance_reduced_learning_sig)
    update = tf.reduce_mean(update_per_sample, 0, name=update_name)
    step = learning_rate * update
    if clip_min is None and clip_max is None:
      apply_step = tf.scatter_add(variable, grad.indices, step)
    else:
      var_lookup = tf.nn.embedding_lookup(variable, grad.indices)
      new_var = var_lookup + step
      new_var_clipped = tf.clip_by_value(
          new_var, clip_value_min=clip_min, clip_value_max=clip_max)
      apply_step = tf.scatter_update(variable, grad.indices, new_var)
  else:
    update_per_sample = (grad / tf.sqrt(mean_square)
                         * variance_reduced_learning_sig)
    update = tf.reduce_mean(update_per_sample, 0,
                            name=update_name)
    step = learning_rate * update
    if clip_min is None and clip_max is None:
      apply_step = tf.assign(variable, variable + step)
    else:
      new_var = variable + step
      new_var_clipped = tf.clip_by_value(
          new_var, clip_value_min=clip_min, clip_value_max=clip_max)
      apply_step = tf.assign(variable, new_var_clipped)
  # add to collection for keeping track of stats
  tf.add_to_collection('non_reparam_variable_grads', update)
  with tf.control_dependencies([update_mean_square]):
    train_op = tf.group(apply_step)
  return train_op


def maximize_with_control_variate(learning_rate, learning_signal, log_prob,
                                  variable_list, global_step=None):
  """Build a covariance control variate with rmsprop updates.

  Args:
    learning_rate: Step size
    learning_signal: Usually the ELBO; the bound we optimize
        Shape [n_samples, batch_size]
    log_prob: log probability of samples of latent variables
    variable_list: List of variables
    global_step: Global step

  Returns:
    train_op: Group of operations that apply an RMSProp update with the
        control variate
  """
  train_ops = []
  for variable in variable_list:
    clip_max, clip_min = (None, None)
    if 'shape_softplus_inv' in variable.name:
      clip_max = sys.float_info.max
      clip_min = 5e-3
    elif 'mean_softplus_inv' in variable.name:
      clip_max = sys.float_info.max
      clip_min = 1e-5
    train_ops.append(_rmsprop_maximize(
        learning_rate, learning_signal, log_prob, variable, clip_max=clip_max,
        clip_min=clip_min))
  if global_step is not None:
    increment_global_step = tf.assign(global_step, global_step + 1)
    with tf.control_dependencies(train_ops):
      train_op = tf.group(increment_global_step)
  else:
    train_op = tf.group(*train_ops)
  return train_op
