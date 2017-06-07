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
"""Gamma mixture model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from ops import rmsprop
from ops import util

st = tf.contrib.bayesflow.stochastic_tensor
distributions = tf.contrib.distributions


def train(optimizer):
  """Trains a gamma-normal mixture model.

  From http://ajbc.io/resources/bbvi_for_gammas.pdf.

  Args:
    optimizer: string specifying whether to use manual rmsprop or rmsprop
        that works on IndexedSlices

  Returns:
    Means learned using variational inference with the given optimizer
  """
  tf.reset_default_graph()
  np_dtype = np.float64
  np.random.seed(11)
  alpha_0 = np.array(0.1, dtype=np_dtype)
  mu_0 = np.array(5., dtype=np_dtype)
  # number of components
  n_components = 12
  # number of datapoints
  n_data = 100
  mu = np.random.gamma(alpha_0, mu_0 / alpha_0, n_components)
  x = np.random.normal(mu, 1., (n_data, n_components))
  ## set up for inference
  # the number of samples to draw for each parameter
  n_samples = 40
  batch_size = 1
  np.random.seed(123232)
  tf.set_random_seed(25343)
  tf_dtype = tf.float64
  inv_softplus = util.inv_softplus

  lambda_alpha_var = tf.get_variable(
      'lambda_alpha', shape=[1, n_components], dtype=tf_dtype,
      initializer=tf.constant_initializer(value=0.1))
  lambda_mu_var = tf.get_variable(
      'lambda_mu', shape=[1, n_components], dtype=tf_dtype,
      initializer=tf.constant_initializer(value=0.1))

  x_indices = tf.placeholder(shape=[batch_size], dtype=tf.int64)

  if optimizer == 'rmsprop_indexed_slices':
    lambda_alpha = tf.nn.embedding_lookup(lambda_alpha_var, x_indices)
    lambda_mu = tf.nn.embedding_lookup(lambda_mu_var, x_indices)
  elif optimizer == 'rmsprop_manual':
    lambda_alpha = lambda_alpha_var
    lambda_mu = lambda_mu_var

  variational = st.StochasticTensor(distributions.Gamma,
                                    alpha=tf.nn.softplus(lambda_alpha),
                                    beta=(tf.nn.softplus(lambda_alpha)
                                          / tf.nn.softplus(lambda_mu)),
                                    dist_value_type=st.SampleValue(n=n_samples),
                                    validate_args=False)

  # truncate samples (don't sample zero )
  sample_mus = tf.maximum(variational.value(), 1e-300)

  # probability of samples given prior
  prior = distributions.Gamma(alpha=alpha_0,
                              beta=alpha_0/mu_0,
                              validate_args=False)

  p = prior.log_pdf(sample_mus)

  # probability of samples given variational parameters
  q = variational.distribution.log_pdf(sample_mus)

  likelihood = distributions.Normal(mu=tf.expand_dims(sample_mus, 1),
                                    sigma=np.array(1., dtype=np_dtype),
                                    validate_args=False)

  # probability of observations given samples
  x_ph = tf.expand_dims(tf.constant(x, dtype=tf_dtype), 0)
  p += tf.reduce_sum(likelihood.log_pdf(x_ph), 2)

  elbo = p - q

  # run BBVI for a fixed number of iterations
  iteration = tf.Variable(0, trainable=False)
  increment_iteration = tf.assign(iteration, iteration + 1)

  # Robbins-Monro sequence for step size
  rho = tf.pow(tf.cast(iteration, tf_dtype) + 1024., -0.7)

  if optimizer == 'rmsprop_manual':
    # control variates to decrease variance of gradient ;
    # one for each variational parameter
    g_alpha = tf.pack([tf.gradients(q_sample, lambda_alpha)[0]
                       for q_sample in tf.unpack(q)])
    g_mu = tf.pack([tf.gradients(q_sample, lambda_mu)[0]
                    for q_sample in tf.unpack(q)])

    def cov(a, b):
      v = (a - tf.reduce_mean(a, 0)) * (b - tf.reduce_mean(b, 0))
      return tf.reduce_mean(v, 0)

    _, var_g_alpha = tf.nn.moments(g_alpha, [0])
    _, var_g_mu = tf.nn.moments(g_mu, [0])

    cov_alpha = cov(g_alpha * (p - q), g_alpha)
    cov_mu = cov(g_mu * (p - q), g_mu)

    cv_alpha = cov_alpha / var_g_alpha
    cv_mu = cov_mu / var_g_mu

    ms_mu = tf.Variable(tf.ones_like(g_mu), trainable=False)
    ms_alpha = tf.Variable(tf.ones_like(g_alpha), trainable=False)
    def update_ms(ms, var):
      return tf.assign(ms, 0.9 * ms + 0.1 * tf.reduce_sum(tf.square(var), 0))

    update_ms_ops = [update_ms(ms_mu, g_mu), update_ms(ms_alpha, g_alpha)]

    # update each variational parameter with smaple average
    alpha_step = rho * tf.reduce_mean(g_alpha / tf.sqrt(ms_alpha) *
                                      (p - q - cv_alpha), 0)
    update_alpha = tf.assign(lambda_alpha, lambda_alpha + alpha_step)
    mu_step = rho * tf.reduce_mean(g_mu / tf.sqrt(ms_mu) * (p - q - cv_mu), 0)
    update_mu = tf.assign(lambda_mu, lambda_mu + mu_step)
    train_ops = tf.group(update_mu, update_alpha)
  elif optimizer == 'rmsprop_indexed_slices':
    variable_list = [lambda_mu_var, lambda_alpha_var]
    train_ops = rmsprop.maximize_with_control_variate(
        rho, elbo, q, variable_list, iteration)

  # truncate variational parameters
  get_min = lambda var, val: tf.assign(var, tf.maximum(var, inv_softplus(val)))
  get_max = lambda var, val: tf.assign(var, tf.minimum(var, inv_softplus(val)))

  get_min_ops = [get_min(lambda_alpha_var, 0.005), get_min(lambda_mu_var, 1e-5)]
  get_max_ops = [get_max(var, sys.float_info.max)
                 for var in [lambda_mu_var, lambda_alpha_var]]

  truncate_ops = get_min_ops + get_max_ops

  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*truncate_ops)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    fd = {x_indices: [0]}
    print('running variational inference using: %s' % optimizer)
    for i in range(100):
      if i % 10 == 0:
        print('iteration %d\telbo %.3e'
              % (i, np.mean(np.sum(elbo.eval(fd), axis=1))))
      if optimizer == 'rmsprop_manual':
        sess.run(update_ms_ops)
      sess.run(train_ops, fd)
      sess.run(increment_iteration)
    # return the learned variational means
    np_mu = sess.run(tf.nn.softplus(lambda_mu), fd)
  return np_mu
