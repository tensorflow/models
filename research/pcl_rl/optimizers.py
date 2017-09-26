# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Optimizers mostly for value estimate.

Gradient Descent optimizer
LBFGS optimizer
Best Fit optimizer

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.optimize


def var_size(v):
  return int(np.prod([int(d) for d in v.shape]))


def gradients(loss, var_list):
  grads = tf.gradients(loss, var_list)
  return [g if g is not None else tf.zeros(v.shape)
          for g, v in zip(grads, var_list)]

def flatgrad(loss, var_list):
  grads = gradients(loss, var_list)
  return tf.concat([tf.reshape(grad, [-1])
                    for (v, grad) in zip(var_list, grads)
                    if grad is not None], 0)


def get_flat(var_list):
  return tf.concat([tf.reshape(v, [-1]) for v in var_list], 0)


def set_from_flat(var_list, flat_theta):
  assigns = []
  shapes = [v.shape for v in var_list]
  sizes = [var_size(v) for v in var_list]

  start = 0
  assigns = []
  for (shape, size, v) in zip(shapes, sizes, var_list):
    assigns.append(v.assign(
        tf.reshape(flat_theta[start:start + size], shape)))
    start += size
  assert start == sum(sizes)

  return tf.group(*assigns)


class LbfgsOptimization(object):

  def __init__(self, max_iter=25, mix_frac=1.0):
    self.max_iter = max_iter
    self.mix_frac = mix_frac

  def setup_placeholders(self):
    self.flat_theta = tf.placeholder(tf.float32, [None], 'flat_theta')
    self.intended_values = tf.placeholder(tf.float32, [None], 'intended_values')

  def setup(self, var_list, values, targets, pads,
            inputs, regression_weight):
    self.setup_placeholders()
    self.values = values
    self.targets = targets

    self.raw_loss = (tf.reduce_sum((1 - pads) * tf.square(values - self.intended_values))
                     / tf.reduce_sum(1 - pads))
    self.loss_flat_gradient = flatgrad(self.raw_loss, var_list)

    self.flat_vars = get_flat(var_list)
    self.set_vars = set_from_flat(var_list, self.flat_theta)

  def optimize(self, sess, feed_dict):
    old_theta = sess.run(self.flat_vars)

    old_values, targets = sess.run([self.values, self.targets], feed_dict=feed_dict)
    intended_values = targets * self.mix_frac + old_values * (1 - self.mix_frac)
    feed_dict = dict(feed_dict)
    feed_dict[self.intended_values] = intended_values

    def calc_loss_and_grad(theta):
      sess.run(self.set_vars, feed_dict={self.flat_theta: theta})
      loss, grad = sess.run([self.raw_loss, self.loss_flat_gradient],
                            feed_dict=feed_dict)
      grad = grad.astype('float64')
      return loss, grad

    theta, _, _ = scipy.optimize.fmin_l_bfgs_b(
        calc_loss_and_grad, old_theta, maxiter=self.max_iter)
    sess.run(self.set_vars, feed_dict={self.flat_theta: theta})


class GradOptimization(object):

  def __init__(self, learning_rate=0.001, max_iter=25, mix_frac=1.0):
    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.mix_frac = mix_frac

  def get_optimizer(self):
    return tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                  epsilon=2e-4)

  def setup_placeholders(self):
    self.flat_theta = tf.placeholder(tf.float32, [None], 'flat_theta')
    self.intended_values = tf.placeholder(tf.float32, [None], 'intended_values')

  def setup(self, var_list, values, targets, pads,
            inputs, regression_weight):
    self.setup_placeholders()
    self.values = values
    self.targets = targets

    self.raw_loss = (tf.reduce_sum((1 - pads) * tf.square(values - self.intended_values))
                     / tf.reduce_sum(1 - pads))

    opt = self.get_optimizer()
    params = var_list
    grads = tf.gradients(self.raw_loss, params)
    self.gradient_ops = opt.apply_gradients(zip(grads, params))

  def optimize(self, sess, feed_dict):
    old_values, targets = sess.run([self.values, self.targets], feed_dict=feed_dict)
    intended_values = targets * self.mix_frac + old_values * (1 - self.mix_frac)

    feed_dict = dict(feed_dict)
    feed_dict[self.intended_values] = intended_values

    for _ in xrange(self.max_iter):
      sess.run(self.gradient_ops, feed_dict=feed_dict)


class BestFitOptimization(object):

  def __init__(self, mix_frac=1.0):
    self.mix_frac = mix_frac

  def setup_placeholders(self):
    self.new_regression_weight = tf.placeholder(
        tf.float32, self.regression_weight.shape)

  def setup(self, var_list, values, targets, pads,
            inputs, regression_weight):
    self.values = values
    self.targets = targets

    self.inputs = inputs
    self.regression_weight = regression_weight

    self.setup_placeholders()

    self.update_regression_weight = tf.assign(
        self.regression_weight, self.new_regression_weight)

  def optimize(self, sess, feed_dict):
    reg_input, reg_weight, old_values, targets = sess.run(
        [self.inputs, self.regression_weight, self.values, self.targets],
        feed_dict=feed_dict)

    intended_values = targets * self.mix_frac + old_values * (1 - self.mix_frac)

    # taken from rllab
    reg_coeff = 1e-5
    for _ in range(5):
      best_fit_weight = np.linalg.lstsq(
          reg_input.T.dot(reg_input) +
          reg_coeff * np.identity(reg_input.shape[1]),
          reg_input.T.dot(intended_values))[0]
      if not np.any(np.isnan(best_fit_weight)):
        break
      reg_coeff *= 10

    if len(best_fit_weight.shape) == 1:
      best_fit_weight = np.expand_dims(best_fit_weight, -1)

    sess.run(self.update_regression_weight,
             feed_dict={self.new_regression_weight: best_fit_weight})
