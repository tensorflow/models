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
"""Classes for models and variational distributions for recurrent DEFs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

from ops import util

st = tf.contrib.bayesflow.stochastic_tensor

distributions = tf.contrib.distributions


class NormalNormalRDEF(object):
  """Class for a recurrent DEF with normal latent variables and normal weights.
  """

  def __init__(self, n_timesteps, batch_size, p_w_mu_sigma, p_w_sigma_sigma,
               p_z_sigma, fixed_p_z_sigma, z_dim, dtype):
    """Initializes the NormalNormalRDEF class.

    Args:
      n_timesteps: int. number of timesteps
      batch_size: int. batch size
      p_w_mu_sigma: float. prior for the weights for the mean of the latent
          variables
      p_w_sigma_sigma: float. prior for the weights for the variance of the
          latent variables
      p_z_sigma: floating point prior for the latent variables
      fixed_p_z_sigma: bool. whether the prior variance is learned
      z_dim: int. dimension of each latent variable
      dtype: dtype
    """

    self.n_timesteps = n_timesteps
    self.batch_size = batch_size
    self.p_w_mu_sigma = p_w_mu_sigma
    self.p_w_sigma_sigma = p_w_sigma_sigma
    self.p_z_sigma = p_z_sigma
    self.fixed_p_z_sigma = fixed_p_z_sigma
    self.z_dim = z_dim
    self.dtype = dtype

  def log_prob(self, params, x):
    """Returns the log joint. log p(x | z, w)p(z)p(w); [batch_size].

    Args:
      params: dict. dictionary of samples of the latent variables.
      x: tensor. minibatch of examples

    Returns:
      The log joint of the NormalNormalRDEF probability model.
    """
    z_1 = params['z_1']
    w_1_mu = params['w_1_mu']
    w_1_sigma = params['w_1_sigma']
    log_p_x_zw, p = util.build_bernoulli_log_likelihood(
        params, x, self.batch_size)
    self.p_x_zw_bernoulli_p = p

    log_p_z, log_p_w_mu, log_p_w_sigma = self.build_recurrent_layer(
        z_1, w_1_mu, w_1_sigma)
    return log_p_x_zw + log_p_z + log_p_w_mu + log_p_w_sigma

  def build_recurrent_layer(self, z, w_mu, w_sigma):
    """Creates a gaussian layer of the recurrent DEF.

    Args:
      z: sampled gaussian latent variables [batch_size, n_timesteps, z_dim]
      w_mu: sampled gaussian stochastic weights [z_dim, z_dim]
      w_sigma: sampled gaussian stochastic weights for stddev
          [z_dim, z_dim]

    Returns:
      log_p_z: log prior of latent variables evaluated at the samples z.
      log_p_w_mu: log density of the weights evaluated at the sampled weights w.
      log_p_w_sigma: log density of weights for stddev.
    """
    # the prior for the weights p(w) has two parts: p(w_mu) and p(w_sigma)
    # prior for the weights for the mean parameter
    p_w_mu = distributions.Normal(
        mu=0., sigma=self.p_w_mu_sigma, validate_args=False)
    log_p_w_mu = tf.reduce_sum(p_w_mu.log_pdf(w_mu))

    if self.fixed_p_z_sigma:
      log_p_w_sigma = 0.0
    else:
      # prior for the weights for the standard deviation
      p_w_sigma = distributions.Normal(mu=0., sigma=self.p_w_sigma_sigma,
                                       validate_args=False)
      log_p_w_sigma = tf.reduce_sum(p_w_sigma.log_pdf(w_sigma))

    # need this for indexing npy-style
    z = z.value()

    # the prior for the latent variable at the first timestep is just 0, 1
    z_t0 = z[:, 0, :]
    p_z_t0 = distributions.Normal(
        mu=0., sigma=self.p_z_sigma, validate_args=False)
    log_p_z_t0 = tf.reduce_sum(p_z_t0.log_pdf(z_t0), 1)

    # the prior for subsequent timesteps is off by one
    mu = tf.batch_matmul(z[:, :self.n_timesteps-1, :],
                         tf.pack([w_mu] * self.batch_size))
    if self.fixed_p_z_sigma:
      sigma = self.p_z_sigma
    else:
      wz = tf.batch_matmul(z[:, :self.n_timesteps-1, :],
                           tf.pack([w_sigma] * self.batch_size))
      sigma = tf.maximum(tf.nn.softplus(wz), 1e-5)
    p_z_t1_to_end = distributions.Normal(mu=mu, sigma=sigma,
                                         validate_args=False)
    log_p_z_t1_to_end = tf.reduce_sum(
        p_z_t1_to_end.log_pdf(z[:, 1:, :]), [1, 2])
    log_p_z = log_p_z_t0 + log_p_z_t1_to_end
    return log_p_z, log_p_w_mu, log_p_w_sigma

  def recurrent_layer_sample(self, w_mu, w_sigma, n_samples_latents):
    """Sample from the model, with learned latent weights.

    Args:
      w_mu: latent weights for the mean parameter. [z_dim, z_dim]
      w_sigma: latent weights for the standard deviation. [z_dim, z_dim]
      n_samples_latents: how many samples of latent variables

    Returns:
      z: samples from the generative process.
    """
    p_z_t0 = distributions.Normal(
        mu=0., sigma=self.p_z_sigma, validate_args=False)
    z_t0 = p_z_t0.sample_n(n=n_samples_latents * self.z_dim)
    z_t0 = tf.reshape(z_t0, [n_samples_latents, self.z_dim])
    def sample_timestep(z_t_prev, w_mu, w_sigma):
      mu_t = tf.matmul(z_t_prev, w_mu)
      if self.fixed_p_z_sigma:
        sigma_t = self.p_z_sigma
      else:
        wz_t = tf.matmul(z_t_prev, w_sigma)
        sigma_t = tf.maximum(tf.nn.softplus(wz_t), 1e-5)
      p_z_t = distributions.Normal(mu=mu_t, sigma=sigma_t, validate_args=False)
      if self.z_dim == 1:
        return p_z_t.sample_n(n=1)[0, :, :]
      else:
        return tf.squeeze(p_z_t.sample_n(n=1))
    z_list = [z_t0]
    for _ in range(self.n_timesteps - 1):
      z_t = sample_timestep(z_list[-1], w_mu, w_sigma)
      z_list.append(z_t)
    z = tf.pack(z_list)  # [n_timesteps, n_samples_latents, z_dim]
    z = tf.transpose(z, perm=[1, 0, 2])  # [n_samples, n_timesteps, z_dim]
    return z

  def likelihood_sample(self, params, z_1, n_samples):
    return util.bernoulli_likelihood_sample(params, z_1, n_samples)


class NormalNormalRDEFVariational(object):
  """Creates the variational family for the recurrent DEF model.

    Variational family:
      q_z_1: gaussian approximate posterior q(z_1) for latents of first layer.
        [n_examples, n_timesteps, z_dim]
      q_w_1_mu: gaussian approximate posterior q(w_1) for mean weights of first
        (recurrent) layer [z_dim, z_dim]
      q_w_1_sigma: gaussian approximate posterior q(w_1) for std weights, first
        (recurrent) layer [z_dim, z_dim]
      q_w_0: gaussian approximate posterior q(w_0) for weights of observation
        layer [z_dim, timestep_dim]
  """

  def __init__(self, x_indexes, n_examples, n_timesteps, z_dim,
               timestep_dim, init_sigma_q_w_mu, init_sigma_q_z,
               init_sigma_q_w_sigma, fixed_p_z_sigma, fixed_q_z_sigma,
               fixed_q_w_mu_sigma, fixed_q_w_sigma_sigma,
               fixed_q_w_0_sigma, init_sigma_q_w_0_sigma, dtype):
    """Initializes the variational family for the NormalNormalRDEF.

    Args:
      x_indexes: tensor. indices of the datapoints.
      n_examples: int. number of examples in the dataset.
      n_timesteps: int. number of timesteps in each datapoint.
      z_dim: int. dimension of latent variables.
      timestep_dim: int. dimension of each timestep.
      init_sigma_q_w_mu: float. initial variance for weights for the means of
          the latent variables.
      init_sigma_q_z: float. initial variance for the variational distribution
          for the latent variables.
      init_sigma_q_w_sigma: float. initial variance for the weights for the
          variance of the latent variables.
      fixed_p_z_sigma: bool. whether to keep the prior over latents fixed.
      fixed_q_z_sigma: bool. whether to train the variance of the variational
          distributions for the latents.
      fixed_q_w_mu_sigma: bool. whether to train the variance of the weights for
          the latent variables.
      fixed_q_w_sigma_sigma: bool. whether to train the variance of the weights
          for the variance of the latent variables.
      fixed_q_w_0_sigma: bool. whether te train the variance of the weights for
          the observations.
      init_sigma_q_w_0_sigma: float. initial variance for the observation
          weights.
      dtype: dtype
    """
    self.x_indexes = x_indexes
    self.n_examples = n_examples
    self.n_timesteps = n_timesteps
    self.z_dim = z_dim
    self.timestep_dim = timestep_dim
    self.init_sigma_q_z = init_sigma_q_z
    self.init_sigma_q_w_mu = init_sigma_q_w_mu
    self.init_sigma_q_w_sigma = init_sigma_q_w_sigma
    self.init_sigma_q_w_0_sigma = init_sigma_q_w_0_sigma
    self.fixed_p_z_sigma = fixed_p_z_sigma
    self.fixed_q_z_sigma = fixed_q_z_sigma
    self.fixed_q_w_mu_sigma = fixed_q_w_mu_sigma
    self.fixed_q_w_sigma_sigma = fixed_q_w_sigma_sigma
    self.fixed_q_w_0_sigma = fixed_q_w_0_sigma
    self.dtype = dtype
    self.build_graph()

  @property
  def sample(self):
    """Returns a dict of samples of the latent variables."""
    return self.params

  def build_graph(self):
    """Builds the graph for the variational family for the NormalNormalRDEF."""
    with tf.variable_scope('q_z_1'):
      z_1 = util.build_gaussian(
          [self.n_examples, self.n_timesteps, self.z_dim],
          init_mu=0., init_sigma=self.init_sigma_q_z, x_indexes=self.x_indexes,
          fixed_sigma=self.fixed_q_z_sigma, place_on_cpu=True, dtype=self.dtype)

    with tf.variable_scope('q_w_1_mu'):
      # half of the weights are for the mean, half for the variance
      w_1_mu = util.build_gaussian([self.z_dim, self.z_dim], init_mu=0.,
                                   init_sigma=self.init_sigma_q_w_mu,
                                   fixed_sigma=self.fixed_q_w_mu_sigma,
                                   dtype=self.dtype)

    if self.fixed_p_z_sigma:
      w_1_sigma = None
    else:
      with tf.variable_scope('q_w_1_sigma'):
        w_1_sigma = util.build_gaussian(
            [self.z_dim, self.z_dim],
            init_mu=0., init_sigma=self.init_sigma_q_w_sigma,
            fixed_sigma=self.fixed_q_w_sigma_sigma,
            dtype=self.dtype)

    with tf.variable_scope('q_w_0'):
      w_0 = util.build_gaussian([self.z_dim, self.timestep_dim], init_mu=0.,
                                init_sigma=self.init_sigma_q_w_0_sigma,
                                fixed_sigma=self.fixed_q_w_0_sigma,
                                dtype=self.dtype)

      self.params = {'w_0': w_0, 'w_1_mu': w_1_mu, 'w_1_sigma': w_1_sigma,
                     'z_1': z_1}

  def log_prob(self, q_samples):
    """Get the log joint of variational family: log(q(z, w_mu, w_sigma, w_0)).

    Args:
      q_samples: dict. samples of latent variables

    Returns:
      log_prob: tensor log-probability summed over dimensions of the variables
    """
    w_0 = q_samples['w_0']
    z_1 = q_samples['z_1']
    w_1_mu = q_samples['w_1_mu']
    w_1_sigma = q_samples['w_1_sigma']

    log_prob = 0.
    # preserve the minibatch dimension [0]
    log_prob += tf.reduce_sum(z_1.distribution.log_pdf(z_1), [1, 2])
    # w_1, w_0 are global, so reduce_sum across all dims
    log_prob += tf.reduce_sum(w_1_mu.distribution.log_pdf(w_1_mu))
    log_prob += tf.reduce_sum(w_0.distribution.log_pdf(w_0))
    if not self.fixed_p_z_sigma:
      log_prob += tf.reduce_sum(w_1_sigma.distribution.log_pdf(w_1_sigma))
    return log_prob


class GammaNormalRDEF(object):
  """Class for a recurrent DEF with normal latent variables and normal weights.
  """

  def __init__(self, n_timesteps, batch_size, p_w_shape_sigma, p_w_mean_sigma,
               p_z_shape, p_z_mean, fixed_p_z_mean, z_dim, n_samples_latents,
               use_bias_observations, dtype):
    """Initializes the NormalNormalRDEF class.

    Args:
      n_timesteps: int. number of timesteps
      batch_size: int. batch size
      p_w_shape_sigma: float. prior for the weights for the mean of the latent
          variables
      p_w_mean_sigma: float. prior for the weights for the shape of the
          latent variables
      p_z_shape: float. prior for shape.
      p_z_mean: floating point prior for the latent variables
      fixed_p_z_mean: bool. whether the prior mean is learned
      z_dim: int. dimension of each latent variable
      n_samples_latents: number of samples of latent variables
      use_bias_observations: whether to use bias terms
      dtype: dtype
    """

    self.n_timesteps = n_timesteps
    self.batch_size = batch_size
    self.p_w_shape_sigma = p_w_shape_sigma
    self.p_w_mean_sigma = p_w_mean_sigma
    self.p_z_shape = p_z_shape
    self.p_z_mean = p_z_mean
    self.fixed_p_z_mean = fixed_p_z_mean
    self.z_dim = z_dim
    self.n_samples_latents = n_samples_latents
    self.use_bias_observations = use_bias_observations
    self.use_bias_latents = False
    self.dtype = dtype

  def log_prob(self, params, x):
    """Returns the log joint. log p(x | z, w)p(z)log p(w); [batch_size].

    Args:
      params: dict. dictionary of samples of the latent variables.
      x: tensor. minibatch of examples

    Returns:
      The log joint of the GammaNormalRDEF probability model.
    """
    z_1 = params['z_1']
    w_1_mean = params['w_1_mean']
    w_1_shape = params['w_1_shape']
    log_p_x_zw, p = util.build_bernoulli_log_likelihood(
        params, x, self.batch_size, n_samples_latents=self.n_samples_latents,
        use_bias_observations=self.use_bias_observations)
    self.p_x_zw_bernoulli_p = p

    log_p_z, log_p_w_shape, log_p_w_mean = self.build_recurrent_layer(
        z_1, w_1_shape, w_1_mean)
    return log_p_x_zw + log_p_z + log_p_w_shape + log_p_w_mean

  def build_recurrent_layer(self, z, w_shape, w_mean):
    """Creates a gaussian layer of the recurrent DEF.

    Args:
      z: sampled gamma latent variables,
          shape [n_samples_latents, batch_size, n_timesteps, z_dim]
      w_shape: single sample of gaussian stochastic weights for shape,
          shape [z_dim, z_dim]
      w_mean: single sample of gaussian stochastic weights for mean,
          shape [z_dim, z_dim]

    Returns:
      log_p_z: log prior of latent variables evaluated at the samples z.
      log_p_w_shape: log density of the weights evaluated at the sampled weights
      log_p_w_mean: log density of weights for stddev.
    """
    # the prior for the weights p(w) has two parts: p(w_shape) and p(w_mean)
    # prior for the weights for the mean parameter
    cast = lambda x: np.array(x, self.dtype)
    p_w_shape = distributions.Normal(mu=cast(0.),
                                     sigma=cast(self.p_w_shape_sigma),
                                     validate_args=False)
    log_p_w_shape = tf.reduce_sum(p_w_shape.log_pdf(w_shape))

    if self.fixed_p_z_mean:
      log_p_w_mean = 0.0
    else:
      # prior for the weights for the standard deviation
      p_w_mean = distributions.Normal(mu=cast(0.),
                                      sigma=cast(self.p_w_mean_sigma),
                                      validate_args=False)
      log_p_w_mean = tf.reduce_sum(p_w_mean.log_pdf(w_mean))

    # need this for indexing npy-style
    z = z.value()

    # the prior for the latent variable at the first timestep is just 0, 1
    z_t0 = z[:, :, 0, :]
    # alpha is shape, beta is inverse scale. we set the scale to be the mean
    # over the shape, so beta = shape / mean.
    p_z_t0 = distributions.Gamma(alpha=cast(self.p_z_shape),
                                 beta=cast(self.p_z_shape / self.p_z_mean),
                                 validate_args=False)
    log_p_z_t0 = tf.reduce_sum(p_z_t0.log_pdf(z_t0), 2)

    # the prior for subsequent timesteps is off by one
    shape = tf.batch_matmul(z[:, :, :self.n_timesteps-1, :],
                            tf.pack([tf.pack([w_shape] * self.batch_size)]
                                    * self.n_samples_latents))
    shape = util.clip_shape(shape)

    if self.fixed_p_z_mean:
      mean = self.p_z_mean
    else:
      wz = tf.batch_matmul(z[:, :, :self.n_timesteps-1, :],
                           tf.pack([tf.pack([w_mean] * self.batch_size)]
                                   * self.n_samples_latents))
      mean = tf.nn.softplus(wz)
      mean = util.clip_mean(mean)
    p_z_t1_to_end = distributions.Gamma(alpha=shape,
                                        beta=shape / mean,
                                        validate_args=False)
    log_p_z_t1_to_end = tf.reduce_sum(
        p_z_t1_to_end.log_pdf(z[:, :, 1:, :]), [2, 3])
    log_p_z = log_p_z_t0 + log_p_z_t1_to_end
    return log_p_z, log_p_w_shape, log_p_w_mean

  def recurrent_layer_sample(self, w_shape, w_mean, n_samples_latents,
                             b_shape=None, b_mean=None):
    """Sample from the model, with learned latent weights.

    Args:
      w_shape: latent weights for the mean parameter. [z_dim, z_dim]
      w_mean: latent weights for the standard deviation. [z_dim, z_dim]
      n_samples_latents: how many samples
      b_shape: bias for shape parameters
      b_mean: bias for mean parameters

    Returns:
      z: samples from the generative process.
    """
    cast = lambda x: np.array(x, self.dtype)
    p_z_t0 = distributions.Gamma(alpha=cast(self.p_z_shape),
                                 beta=cast(self.p_z_shape / self.p_z_mean),
                                 validate_args=False)
    z_t0 = p_z_t0.sample_n(n=n_samples_latents * self.z_dim)

    z_t0 = tf.reshape(z_t0, [n_samples_latents, self.z_dim])

    def sample_timestep(z_t_prev, w_shape, w_mean, b_shape=b_shape,
                        b_mean=b_mean):
      """Sample a single timestep.

      Args:
        z_t_prev: previous timestep latent variable,
            shape [n_samples_latents, z_dim]
        w_shape: latent weights for shape param, shape [z_dim, z_dim]
        w_mean: latent weights for mean param, shape [z_dim, z_dim]
        b_shape: bias for shape parameters
        b_mean: bias for mean parameters

      Returns:
        z_t: A sample of a latent variable for all timesteps
      """
      wz_t = tf.matmul(z_t_prev, w_shape)
      if self.use_bias_latents:
        wz_t += b_shape
      shape_t = tf.nn.softplus(wz_t)
      shape_t = util.clip_shape(shape_t)
      if self.fixed_p_z_mean:
        mean_t = self.p_z_mean
      else:
        wz_t = tf.matmul(z_t_prev, w_mean)
        if self.use_bias_latents:
          wz_t += b_mean
        mean_t = tf.nn.softplus(wz_t)
        mean_t = util.clip_mean(mean_t)
      p_z_t = distributions.Gamma(alpha=shape_t,
                                  beta=shape_t / mean_t,
                                  validate_args=False)
      z_t = p_z_t.sample_n(n=1)[0, :, :]
      return z_t
    z_list = [z_t0]
    for _ in range(self.n_timesteps - 1):
      z_t = sample_timestep(z_list[-1], w_shape, w_mean)
      z_list.append(z_t)
    # pack into shape [n_timesteps, n_samples_latents, z_dim]
    z = tf.pack(z_list)
    # transpose into [n_samples_latents, n_timesteps, z_dim]
    z = tf.transpose(z, perm=[1, 0, 2])
    return z

  def likelihood_sample(self, params, z_1, n_samples):
    return util.bernoulli_likelihood_sample(
        params, z_1, n_samples,
        use_bias_observations=self.use_bias_observations)


class GammaNormalRDEFVariational(object):
  """Creates the variational family for the recurrent DEF model.

    Variational family:
      q_z_1: gaussian approximate posterior q(z_1) for latents of first layer.
        [n_examples, n_timesteps, z_dim]
      q_w_1_shape: gaussian approximate posterior q(w_1) for mean weights of
        (recurrent) layer [z_dim, z_dim]
      q_w_1_mean: gaussian approximate posterior q(w_1) for std weights, first
        (recurrent) layer [z_dim, z_dim]
      q_w_0: gaussian approximate posterior q(w_0) for weights of observation
        layer [z_dim, timestep_dim]
  """

  def __init__(self, x_indexes, n_examples, n_timesteps, z_dim,
               timestep_dim, init_sigma_q_w_shape, init_shape_q_z,
               init_mean_q_z,
               init_sigma_q_w_mean, fixed_p_z_mean, fixed_q_z_mean,
               fixed_q_w_shape_sigma, fixed_q_w_mean_sigma,
               fixed_q_w_0_sigma, init_sigma_q_w_0_sigma, n_samples_latents,
               use_bias_observations,
               dtype):
    """Initializes the variational family for the NormalNormalRDEF.

    Args:
      x_indexes: tensor. indices of the datapoints.
      n_examples: int. number of examples in the dataset.
      n_timesteps: int. number of timesteps in each datapoint.
      z_dim: int. dimension of latent variables.
      timestep_dim: int. dimension of each timestep.
      init_sigma_q_w_shape: float. initial variance for weights for the means of
          the latent variables.
      init_shape_q_z: float. initial variance for the variational distribution
          for the latent variables.
      init_mean_q_z: float. initial mean for latent variables variational.
      init_sigma_q_w_mean: float. initial variance for the weights for the
          variance of the latent variables.
      fixed_p_z_mean: bool. whether to keep the prior over latents fixed.
      fixed_q_z_mean: bool. whether to train the variance of the variational
          distributions for the latents.
      fixed_q_w_shape_sigma: bool. whether to train the variance of the weights
          the latent variables.
      fixed_q_w_mean_sigma: bool. whether to train the variance of the weights
          for the variance of the latent variables.
      fixed_q_w_0_sigma: bool. whether te train the variance of the weights for
          the observations.
      init_sigma_q_w_0_sigma: float. initial variance for the observation
          weights.
      n_samples_latents: number of samples of latent variables to draw
      use_bias_observations: whether to use bias terms
      dtype: dtype
    """
    self.x_indexes = x_indexes
    self.n_examples = n_examples
    self.n_timesteps = n_timesteps
    self.z_dim = z_dim
    self.timestep_dim = timestep_dim
    self.init_mean_q_z = init_mean_q_z
    self.init_shape_q_z = init_shape_q_z
    self.init_sigma_q_w_shape = init_sigma_q_w_shape
    self.init_sigma_q_w_mean = init_sigma_q_w_mean
    self.init_sigma_q_w_0_sigma = init_sigma_q_w_0_sigma
    self.fixed_p_z_mean = fixed_p_z_mean
    self.fixed_q_z_mean = fixed_q_z_mean
    self.fixed_q_w_shape_sigma = fixed_q_w_shape_sigma
    self.fixed_q_w_mean_sigma = fixed_q_w_mean_sigma
    self.fixed_q_w_0_sigma = fixed_q_w_0_sigma
    self.n_samples_latents = n_samples_latents
    self.use_bias_observations = use_bias_observations
    self.dtype = dtype
    with tf.variable_scope('variational'):
      self.build_graph()

  @property
  def sample(self):
    """Returns a dict of samples of the latent variables."""
    return self.params

  @property
  def trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'variational')

  def build_graph(self):
    """Builds the graph for the variational family for the NormalNormalRDEF."""
    with tf.variable_scope('q_z_1'):
      z_1 = util.build_gamma(
          [self.n_examples, self.n_timesteps, self.z_dim],
          init_shape=self.init_shape_q_z,
          init_mean=self.init_mean_q_z,
          x_indexes=self.x_indexes,
          fixed_mean=self.fixed_q_z_mean,
          place_on_cpu=False,
          n_samples=self.n_samples_latents,
          dtype=self.dtype)

    with tf.variable_scope('q_w_1_shape'):
      # half of the weights are for the mean, half for the variance
      w_1_shape = util.build_gaussian([self.z_dim, self.z_dim], init_mu=0.,
                                      init_sigma=self.init_sigma_q_w_shape,
                                      fixed_sigma=self.fixed_q_w_shape_sigma,
                                      dtype=self.dtype)

    if self.fixed_p_z_mean:
      w_1_mean = None
    else:
      with tf.variable_scope('q_w_1_mean'):
        w_1_mean = util.build_gaussian(
            [self.z_dim, self.z_dim],
            init_mu=0., init_sigma=self.init_sigma_q_w_mean,
            fixed_sigma=self.fixed_q_w_mean_sigma,
            dtype=self.dtype)

    with tf.variable_scope('q_w_0'):
      w_0 = util.build_gaussian([self.z_dim, self.timestep_dim], init_mu=0.,
                                init_sigma=self.init_sigma_q_w_0_sigma,
                                fixed_sigma=self.fixed_q_w_0_sigma,
                                dtype=self.dtype)

    self.params = {'w_0': w_0, 'w_1_shape': w_1_shape, 'w_1_mean': w_1_mean,
                   'z_1': z_1}

    if self.use_bias_observations:
      # b_0 = tf.get_variable(
      #     'b_0', [self.timestep_dim], self.dtype, tf.zeros_initializer,
      #     collections=[tf.GraphKeys.VARIABLES, 'reparam_variables'])
      b_0 = util.build_gaussian([self.timestep_dim], init_mu=0.,
                                init_sigma=0.01, fixed_sigma=False,
                                dtype=self.dtype)
      self.params.update({'b_0': b_0})

  def log_prob(self, q_samples):
    """Get the log joint of variational family: log(q(z, w_shape, w_mean, w_0)).

    Args:
      q_samples: dict. samples of latent variables.

    Returns:
      log_prob: tensor log-probability summed over dimensions of the variables
    """
    w_0 = q_samples['w_0']
    z_1 = q_samples['z_1']
    w_1_shape = q_samples['w_1_shape']
    w_1_mean = q_samples['w_1_mean']

    log_prob = 0.
    # preserve the sample and minibatch dimensions [0, 1]
    log_prob += tf.reduce_sum(z_1.distribution.log_pdf(z_1.value()), [2, 3])
    # w_1, w_0 are global, so reduce_sum across all dims
    log_prob += tf.reduce_sum(w_1_shape.distribution.log_pdf(w_1_shape.value()))
    log_prob += tf.reduce_sum(w_0.distribution.log_pdf(w_0.value()))
    if not self.fixed_p_z_mean:
      log_prob += tf.reduce_sum(w_1_mean.distribution.log_pdf(w_1_mean.value()))
    return log_prob
