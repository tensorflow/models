# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# ==============================================================================
import numpy as np
import tensorflow as tf
from utils import linear, log_sum_exp

class Poisson(object):
  """Poisson distributon

  Computes the log probability under the model.

  """
  def __init__(self, log_rates):
    """ Create Poisson distributions with log_rates parameters.

    Args:
      log_rates: a tensor-like list of log rates underlying the Poisson dist.
    """
    self.logr = log_rates

  def logp(self, bin_counts):
    """Compute the log probability for the counts in the bin, under the model.

    Args:
      bin_counts: array-like integer counts

    Returns:
      The log-probability under the Poisson models for each element of
      bin_counts.
    """
    k = tf.to_float(bin_counts)
    # log poisson(k, r) = log(r^k * e^(-r) / k!) = k log(r) - r - log k!
    # log poisson(k, r=exp(x)) = k * x - exp(x) - lgamma(k + 1)
    return k * self.logr - tf.exp(self.logr) - tf.lgamma(k + 1)


def diag_gaussian_log_likelihood(z, mu=0.0, logvar=0.0):
  """Log-likelihood under a Gaussian distribution with diagonal covariance.
    Returns the log-likelihood for each dimension.  One should sum the
    results for the log-likelihood under the full multidimensional model.

  Args:
    z: The value to compute the log-likelihood.
    mu: The mean of the Gaussian
    logvar: The log variance of the Gaussian.

  Returns:
    The log-likelihood under the Gaussian model.
  """

  return -0.5 * (logvar + np.log(2*np.pi) + \
                 tf.square((z-mu)/tf.exp(0.5*logvar)))


def gaussian_pos_log_likelihood(unused_mean, logvar, noise):
  """Gaussian log-likelihood function for a posterior in VAE

  Note: This function is specialized for a posterior distribution, that has the
  form of z = mean + sigma * noise.

  Args:
    unused_mean: ignore
    logvar: The log variance of the distribution
    noise: The noise used in the sampling of the posterior.

  Returns:
    The log-likelihood under the Gaussian model.
  """
  # ln N(z; mean, sigma) = - ln(sigma) - 0.5 ln 2pi - noise^2 / 2
  return - 0.5 * (logvar + np.log(2 * np.pi) + tf.square(noise))


class Gaussian(object):
  """Base class for Gaussian distribution classes."""
  pass


class DiagonalGaussian(Gaussian):
  """Diagonal Gaussian with different constant mean and variances in each
  dimension.
  """

  def __init__(self, batch_size, z_size, mean, logvar):
    """Create a diagonal gaussian distribution.

    Args:
      batch_size: The size of the batch, i.e. 0th dim in 2D tensor of samples.
      z_size: The dimension of the distribution, i.e. 1st dim in 2D tensor.
      mean: The N-D mean of the distribution.
      logvar: The N-D log variance of the diagonal distribution.
    """
    size__xz = [None, z_size]
    self.mean = mean            # bxn already
    self.logvar = logvar        # bxn already
    self.noise = noise = tf.random_normal(tf.shape(logvar))
    self.sample = mean + tf.exp(0.5 * logvar) * noise
    mean.set_shape(size__xz)
    logvar.set_shape(size__xz)
    self.sample.set_shape(size__xz)

  def logp(self, z=None):
    """Compute the log-likelihood under the distribution.

    Args:
      z (optional): value to compute likelihood for, if None, use sample.

    Returns:
      The likelihood of z under the model.
    """
    if z is None:
      z = self.sample

    # This is needed to make sure that the gradients are simple.
    # The value of the function shouldn't change.
    if z == self.sample:
      return gaussian_pos_log_likelihood(self.mean, self.logvar, self.noise)

    return diag_gaussian_log_likelihood(z, self.mean, self.logvar)


class LearnableDiagonalGaussian(Gaussian):
  """Diagonal Gaussian whose mean and variance are learned parameters."""

  def __init__(self, batch_size, z_size, name, mean_init=0.0,
               var_init=1.0, var_min=0.0, var_max=1000000.0):
    """Create a learnable diagonal gaussian distribution.

    Args:
      batch_size: The size of the batch, i.e. 0th dim in 2D tensor of samples.
      z_size: The dimension of the distribution, i.e. 1st dim in 2D tensor.
      name: prefix name for the mean and log TF variables.
      mean_init (optional): The N-D mean initialization of the distribution.
      var_init (optional): The N-D variance initialization of the diagonal
        distribution.
      var_min (optional): The minimum value the learned variance can take in any
        dimension.
      var_max (optional): The maximum value the learned variance can take in any
        dimension.
    """

    size_1xn = [1, z_size]
    size__xn = [None, z_size]
    size_bx1 = tf.stack([batch_size, 1])
    assert var_init > 0.0, "Problems"
    assert var_max >= var_min, "Problems"
    assert var_init >= var_min, "Problems"
    assert var_max >= var_init, "Problems"


    z_mean_1xn = tf.get_variable(name=name+"/mean", shape=size_1xn,
                                 initializer=tf.constant_initializer(mean_init))
    self.mean_bxn = mean_bxn = tf.tile(z_mean_1xn, size_bx1)
    mean_bxn.set_shape(size__xn) # tile loses shape

    log_var_init = np.log(var_init)
    if var_max > var_min:
      var_is_trainable = True
    else:
      var_is_trainable = False

    z_logvar_1xn = \
        tf.get_variable(name=(name+"/logvar"), shape=size_1xn,
                        initializer=tf.constant_initializer(log_var_init),
                        trainable=var_is_trainable)

    if var_is_trainable:
      z_logit_var_1xn = tf.exp(z_logvar_1xn)
      z_var_1xn = tf.nn.sigmoid(z_logit_var_1xn)*(var_max-var_min) + var_min
      z_logvar_1xn = tf.log(z_var_1xn)

    logvar_bxn = tf.tile(z_logvar_1xn, size_bx1)
    self.logvar_bxn = logvar_bxn
    self.noise_bxn = noise_bxn = tf.random_normal(tf.shape(logvar_bxn))
    self.sample_bxn = mean_bxn + tf.exp(0.5 * logvar_bxn) * noise_bxn

  def logp(self, z=None):
    """Compute the log-likelihood under the distribution.

    Args:
      z (optional): value to compute likelihood for, if None, use sample.

    Returns:
      The likelihood of z under the model.
    """
    if z is None:
      z = self.sample

    # This is needed to make sure that the gradients are simple.
    # The value of the function shouldn't change.
    if z == self.sample_bxn:
      return gaussian_pos_log_likelihood(self.mean_bxn, self.logvar_bxn,
                                         self.noise_bxn)

    return diag_gaussian_log_likelihood(z, self.mean_bxn, self.logvar_bxn)

  @property
  def mean(self):
    return self.mean_bxn

  @property
  def logvar(self):
    return self.logvar_bxn

  @property
  def sample(self):
    return self.sample_bxn


class DiagonalGaussianFromInput(Gaussian):
  """Diagonal Gaussian whose mean and variance are conditioned on other
  variables.

  Note: the parameters to convert from input to the learned mean and log
  variance are held in this class.
  """

  def __init__(self, x_bxu, z_size, name, var_min=0.0):
    """Create an input dependent diagonal Gaussian distribution.

    Args:
      x: The input tensor from which the mean and variance are computed,
        via a linear transformation of x.  I.e.
          mu = Wx + b, log(var) = Mx + c
      z_size: The size of the distribution.
      name:  The name to prefix to learned variables.
      var_min (optional): Minimal variance allowed.  This is an additional
        way to control the amount of information getting through the stochastic
        layer.
    """
    size_bxn = tf.stack([tf.shape(x_bxu)[0], z_size])
    self.mean_bxn = mean_bxn = linear(x_bxu, z_size, name=(name+"/mean"))
    logvar_bxn = linear(x_bxu, z_size, name=(name+"/logvar"))
    if var_min > 0.0:
      logvar_bxn = tf.log(tf.exp(logvar_bxn) + var_min)
    self.logvar_bxn = logvar_bxn

    self.noise_bxn = noise_bxn = tf.random_normal(size_bxn)
    self.noise_bxn.set_shape([None, z_size])
    self.sample_bxn = mean_bxn + tf.exp(0.5 * logvar_bxn) * noise_bxn

  def logp(self, z=None):
    """Compute the log-likelihood under the distribution.

    Args:
      z (optional): value to compute likelihood for, if None, use sample.

    Returns:
      The likelihood of z under the model.
    """

    if z is None:
      z = self.sample

    # This is needed to make sure that the gradients are simple.
    # The value of the function shouldn't change.
    if z == self.sample_bxn:
      return gaussian_pos_log_likelihood(self.mean_bxn,
                                         self.logvar_bxn, self.noise_bxn)

    return diag_gaussian_log_likelihood(z, self.mean_bxn, self.logvar_bxn)

  @property
  def mean(self):
    return self.mean_bxn

  @property
  def logvar(self):
    return self.logvar_bxn

  @property
  def sample(self):
    return self.sample_bxn


class GaussianProcess:
  """Base class for Gaussian processes."""
  pass


class LearnableAutoRegressive1Prior(GaussianProcess):
  """AR(1) model where autocorrelation and process variance are learned
  parameters.  Assumed zero mean.

  """

  def __init__(self, batch_size, z_size,
               autocorrelation_taus, noise_variances,
               do_train_prior_ar_atau, do_train_prior_ar_nvar,
               num_steps, name):
    """Create a learnable autoregressive (1) process.

    Args:
      batch_size: The size of the batch, i.e. 0th dim in 2D tensor of samples.
      z_size: The dimension of the distribution, i.e. 1st dim in 2D tensor.
      autocorrelation_taus: The auto correlation time constant of the AR(1)
      process.
        A value of 0 is uncorrelated gaussian noise.
      noise_variances: The variance of the additive noise, *not* the process
        variance.
      do_train_prior_ar_atau: Train or leave as constant, the autocorrelation?
      do_train_prior_ar_nvar: Train or leave as constant, the noise variance?
      num_steps: Number of steps to run the process.
      name: The name to prefix to learned TF variables.
    """

    # Note the use of the plural in all of these quantities.  This is intended
    # to mark that even though a sample z_t from the posterior is thought of a
    # single sample of a multidimensional gaussian, the prior is actually
    # thought of as U AR(1) processes, where U is the dimension of the inferred
    # input.
    size_bx1 = tf.stack([batch_size, 1])
    size__xu = [None, z_size]
    # process variance, the variance at time t over all instantiations of AR(1)
    # with these parameters.
    log_evar_inits_1xu = tf.expand_dims(tf.log(noise_variances), 0)
    self.logevars_1xu = logevars_1xu = \
        tf.Variable(log_evar_inits_1xu, name=name+"/logevars", dtype=tf.float32,
                    trainable=do_train_prior_ar_nvar)
    self.logevars_bxu = logevars_bxu = tf.tile(logevars_1xu, size_bx1)
    logevars_bxu.set_shape(size__xu) # tile loses shape

    # \tau, which is the autocorrelation time constant of the AR(1) process
    log_atau_inits_1xu = tf.expand_dims(tf.log(autocorrelation_taus), 0)
    self.logataus_1xu = logataus_1xu = \
        tf.Variable(log_atau_inits_1xu, name=name+"/logatau", dtype=tf.float32,
                    trainable=do_train_prior_ar_atau)

    # phi in x_t = \mu + phi x_tm1 + \eps
    # phi = exp(-1/tau)
    # phi = exp(-1/exp(logtau))
    # phi = exp(-exp(-logtau))
    phis_1xu = tf.exp(-tf.exp(-logataus_1xu))
    self.phis_bxu = phis_bxu = tf.tile(phis_1xu, size_bx1)
    phis_bxu.set_shape(size__xu)

    # process noise
    # pvar = evar / (1- phi^2)
    # logpvar = log ( exp(logevar) / (1 - phi^2) )
    # logpvar = logevar - log(1-phi^2)
    # logpvar = logevar - (log(1-phi) + log(1+phi))
    self.logpvars_1xu = \
        logevars_1xu - tf.log(1.0-phis_1xu) - tf.log(1.0+phis_1xu)
    self.logpvars_bxu = logpvars_bxu = tf.tile(self.logpvars_1xu, size_bx1)
    logpvars_bxu.set_shape(size__xu)

    # process mean (zero but included in for completeness)
    self.pmeans_bxu = pmeans_bxu = tf.zeros_like(phis_bxu)

    # For sampling from the prior during de-novo generation.
    self.means_t = means_t = [None] * num_steps
    self.logvars_t = logvars_t = [None] * num_steps
    self.samples_t = samples_t = [None] * num_steps
    self.gaussians_t = gaussians_t = [None] * num_steps
    sample_bxu = tf.zeros_like(phis_bxu)
    for t in range(num_steps):
      # process variance used here to make process completely stationary
      if t == 0:
        logvar_pt_bxu = self.logpvars_bxu
      else:
        logvar_pt_bxu = self.logevars_bxu

      z_mean_pt_bxu = pmeans_bxu + phis_bxu * sample_bxu
      gaussians_t[t] = DiagonalGaussian(batch_size, z_size,
                                        mean=z_mean_pt_bxu,
                                        logvar=logvar_pt_bxu)
      sample_bxu = gaussians_t[t].sample
      samples_t[t] = sample_bxu
      logvars_t[t] = logvar_pt_bxu
      means_t[t] = z_mean_pt_bxu

  def logp_t(self, z_t_bxu, z_tm1_bxu=None):
    """Compute the log-likelihood under the distribution for a given time t,
    not the whole sequence.

    Args:
      z_t_bxu: sample to compute likelihood for at time t.
      z_tm1_bxu (optional): sample condition probability of z_t upon.

    Returns:
      The likelihood of p_t under the model at time t. i.e.
        p(z_t|z_tm1_bxu) = N(z_tm1_bxu * phis, eps^2)

    """
    if z_tm1_bxu is None:
      return diag_gaussian_log_likelihood(z_t_bxu, self.pmeans_bxu,
                                          self.logpvars_bxu)
    else:
      means_t_bxu = self.pmeans_bxu + self.phis_bxu * z_tm1_bxu
      logp_tgtm1_bxu = diag_gaussian_log_likelihood(z_t_bxu,
                                                    means_t_bxu,
                                                    self.logevars_bxu)
      return logp_tgtm1_bxu


class KLCost_GaussianGaussian(object):
  """log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian prior. See
  eqn 10 and Appendix B in VAE for latter term,
  http://arxiv.org/abs/1312.6114

  The log p(x|z) term is the reconstruction error under the model.
  The KL term represents the penalty for passing information from the encoder
  to the decoder.
  To sample KL(q||p), we simply sample
        ln q - ln p
  by drawing samples from q and averaging.
  """

  def __init__(self, zs, prior_zs):
    """Create a lower bound in three parts, normalized reconstruction
    cost, normalized KL divergence cost, and their sum.

    E_q[ln p(z_i | z_{i+1}) / q(z_i | x)
       \int q(z) ln p(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_p^2) + \
          sigma_q^2 / sigma_p^2 + (mean_p - mean_q)^2 / sigma_p^2)

       \int q(z) ln q(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_q^2) + 1)

    Args:
      zs: posterior z ~ q(z|x)
      prior_zs: prior zs
    """
    # L = -KL + log p(x|z), to maximize bound on likelihood
    # -L = KL - log p(x|z), to minimize bound on NLL
    # so 'KL cost' is postive KL divergence
    kl_b = 0.0
    for z, prior_z in zip(zs, prior_zs):
      assert isinstance(z, Gaussian)
      assert isinstance(prior_z, Gaussian)
      # ln(2pi) terms cancel
      kl_b += 0.5 * tf.reduce_sum(
          prior_z.logvar - z.logvar
          + tf.exp(z.logvar - prior_z.logvar)
          + tf.square((z.mean - prior_z.mean) / tf.exp(0.5 * prior_z.logvar))
          - 1.0, [1])

    self.kl_cost_b = kl_b
    self.kl_cost = tf.reduce_mean(kl_b)


class KLCost_GaussianGaussianProcessSampled(object):
  """ log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian process
  prior via sampling.

  The log p(x|z) term is the reconstruction error under the model.
  The KL term represents the penalty for passing information from the encoder
  to the decoder.
  To sample KL(q||p), we simply sample
        ln q - ln p
  by drawing samples from q and averaging.
  """

  def __init__(self, post_zs, prior_z_process):
    """Create a lower bound in three parts, normalized reconstruction
    cost, normalized KL divergence cost, and their sum.

    Args:
      post_zs: posterior z ~ q(z|x)
      prior_z_process: prior AR(1) process
    """
    assert len(post_zs) > 1, "GP is for time, need more than 1 time step."
    assert isinstance(prior_z_process, GaussianProcess), "Must use GP."

    # L = -KL + log p(x|z), to maximize bound on likelihood
    # -L = KL - log p(x|z), to minimize bound on NLL
    # so 'KL cost' is postive KL divergence
    z0_bxu = post_zs[0].sample
    logq_bxu = post_zs[0].logp(z0_bxu)
    logp_bxu = prior_z_process.logp_t(z0_bxu)
    z_tm1_bxu = z0_bxu
    for z_t in post_zs[1:]:
      # posterior is independent in time, prior is not
      z_t_bxu = z_t.sample
      logq_bxu += z_t.logp(z_t_bxu)
      logp_bxu += prior_z_process.logp_t(z_t_bxu, z_tm1_bxu)
      z_tm1_bxu = z_t_bxu

    kl_bxu = logq_bxu - logp_bxu
    kl_b = tf.reduce_sum(kl_bxu, [1])
    self.kl_cost_b = kl_b
    self.kl_cost = tf.reduce_mean(kl_b)
