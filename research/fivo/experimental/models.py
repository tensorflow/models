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

"""Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sonnet as snt
import tensorflow as tf
import numpy as np
import math

SQUARED_OBSERVATION = "squared"
ABS_OBSERVATION = "abs"
STANDARD_OBSERVATION = "standard"
OBSERVATION_TYPES = [SQUARED_OBSERVATION, ABS_OBSERVATION, STANDARD_OBSERVATION]

ROUND_TRANSITION = "round"
STANDARD_TRANSITION = "standard"
TRANSITION_TYPES = [ROUND_TRANSITION, STANDARD_TRANSITION]


class Q(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None,
               init_mu0_to_zero=False,
               graph_collection_name="Q_VARS"):
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.graph_collection_name = graph_collection_name
    initializers = []
    for t in xrange(num_timesteps):
      if t == 0 and init_mu0_to_zero:
        initializers.append(
            {"w": tf.zeros_initializer, "b": tf.zeros_initializer})
      else:
        initializers.append(
            {"w": tf.random_uniform_initializer(seed=random_seed),
             "b": tf.zeros_initializer})

    def custom_getter(getter, *args, **kwargs):
      out = getter(*args, **kwargs)
      ref = tf.get_collection_ref(self.graph_collection_name)
      if out not in ref:
        ref.append(out)
      return out

    self.mus = [
        snt.Linear(output_size=state_size,
                   initializers=initializers[t],
                   name="q_mu_%d" % t,
                   custom_getter=custom_getter
                  )
        for t in xrange(num_timesteps)
    ]
    self.sigmas = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="q_sigma_%d" % (t + 1),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, graph_collection_name],
            initializer=tf.random_uniform_initializer(seed=random_seed))
        for t in xrange(num_timesteps)
    ]

  def q_zt(self, observation, prev_state, t):
    batch_size = tf.shape(prev_state)[0]
    q_mu = self.mus[t](tf.concat([observation, prev_state], axis=1))
    q_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    q_sigma = tf.tile(q_sigma[tf.newaxis, :], [batch_size, 1])
    q_zt = tf.contrib.distributions.Normal(loc=q_mu, scale=tf.sqrt(q_sigma))
    return q_zt

  def summarize_weights(self):
    for t, sigma in enumerate(self.sigmas):
      tf.summary.scalar("q_sigma/%d" % t, sigma[0])
    for t, f in enumerate(self.mus):
      tf.summary.scalar("q_mu/b_%d" % t, f.b[0])
      tf.summary.scalar("q_mu/w_obs_%d" % t, f.w[0,0])
      if t != 0:
        tf.summary.scalar("q_mu/w_prev_state_%d" % t, f.w[1,0])


class PreviousStateQ(Q):

  def q_zt(self, unused_observation, prev_state, t):
    batch_size = tf.shape(prev_state)[0]
    q_mu = self.mus[t](prev_state)
    q_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    q_sigma = tf.tile(q_sigma[tf.newaxis, :], [batch_size, 1])
    q_zt = tf.contrib.distributions.Normal(loc=q_mu, scale=tf.sqrt(q_sigma))
    return q_zt

  def summarize_weights(self):
    for t, sigma in enumerate(self.sigmas):
      tf.summary.scalar("q_sigma/%d" % t, sigma[0])
    for t, f in enumerate(self.mus):
      tf.summary.scalar("q_mu/b_%d" % t, f.b[0])
      tf.summary.scalar("q_mu/w_prev_state_%d" % t, f.w[0,0])


class ObservationQ(Q):

  def q_zt(self, observation, prev_state, t):
    batch_size = tf.shape(prev_state)[0]
    q_mu = self.mus[t](observation)
    q_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    q_sigma = tf.tile(q_sigma[tf.newaxis, :], [batch_size, 1])
    q_zt = tf.contrib.distributions.Normal(loc=q_mu, scale=tf.sqrt(q_sigma))
    return q_zt

  def summarize_weights(self):
    for t, sigma in enumerate(self.sigmas):
      tf.summary.scalar("q_sigma/%d" % t, sigma[0])
    for t, f in enumerate(self.mus):
      tf.summary.scalar("q_mu/b_%d" % t, f.b[0])
      tf.summary.scalar("q_mu/w_obs_%d" % t, f.w[0,0])


class SimpleMeanQ(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None,
               init_mu0_to_zero=False,
               graph_collection_name="Q_VARS"):
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.graph_collection_name = graph_collection_name
    initializers = []
    for t in xrange(num_timesteps):
      if t == 0 and init_mu0_to_zero:
        initializers.append(tf.zeros_initializer)
      else:
        initializers.append(tf.random_uniform_initializer(seed=random_seed))

    self.mus = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="q_mu_%d" % (t + 1),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, graph_collection_name],
            initializer=initializers[t])
        for t in xrange(num_timesteps)
    ]
    self.sigmas = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="q_sigma_%d" % (t + 1),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, graph_collection_name],
            initializer=tf.random_uniform_initializer(seed=random_seed))
        for t in xrange(num_timesteps)
    ]

  def q_zt(self, unused_observation, prev_state, t):
    batch_size = tf.shape(prev_state)[0]
    q_mu = tf.tile(self.mus[t][tf.newaxis, :], [batch_size, 1])
    q_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    q_sigma = tf.tile(q_sigma[tf.newaxis, :], [batch_size, 1])
    q_zt = tf.contrib.distributions.Normal(loc=q_mu, scale=tf.sqrt(q_sigma))
    return q_zt

  def summarize_weights(self):
    for t, sigma in enumerate(self.sigmas):
      tf.summary.scalar("q_sigma/%d" % t, sigma[0])
    for t, f in enumerate(self.mus):
      tf.summary.scalar("q_mu/%d" % t, f[0])


class R(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               dtype=tf.float32,
               sigma_init=1.,
               random_seed=None,
               graph_collection_name="R_VARS"):
    self.dtype = dtype
    self.sigma_min = sigma_min
    initializers = {"w": tf.truncated_normal_initializer(seed=random_seed),
                    "b": tf.zeros_initializer}
    self.graph_collection_name=graph_collection_name

    def custom_getter(getter, *args, **kwargs):
      out = getter(*args, **kwargs)
      ref = tf.get_collection_ref(self.graph_collection_name)
      if out not in ref:
        ref.append(out)
      return out

    self.mus= [
        snt.Linear(output_size=state_size,
                   initializers=initializers,
                   name="r_mu_%d" % t,
                   custom_getter=custom_getter)
        for t in xrange(num_timesteps)
    ]

    self.sigmas = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="r_sigma_%d" % (t + 1),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, graph_collection_name],
            #initializer=tf.random_uniform_initializer(seed=random_seed, maxval=100))
            initializer=tf.constant_initializer(sigma_init))
        for t in xrange(num_timesteps)
    ]

  def r_xn(self, z_t, t):
    batch_size = tf.shape(z_t)[0]
    r_mu = self.mus[t](z_t)
    r_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    r_sigma = tf.tile(r_sigma[tf.newaxis, :], [batch_size, 1])
    return tf.contrib.distributions.Normal(
        loc=r_mu, scale=tf.sqrt(r_sigma))

  def summarize_weights(self):
    for t in range(len(self.mus) - 1):
      tf.summary.scalar("r_mu/%d" % t, self.mus[t][0])
      tf.summary.scalar("r_sigma/%d" % t, self.sigmas[t][0])


class P(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               variance=1.0,
               dtype=tf.float32,
               random_seed=None,
               trainable=True,
               init_bs_to_zero=False,
               graph_collection_name="P_VARS"):
    self.state_size = state_size
    self.num_timesteps = num_timesteps
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.variance = variance
    self.graph_collection_name = graph_collection_name
    if init_bs_to_zero:
      initializers = [tf.zeros_initializer for _ in xrange(num_timesteps)]
    else:
      initializers = [tf.random_uniform_initializer(seed=random_seed) for _ in xrange(num_timesteps)]

    self.bs = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="p_b_%d" % (t + 1),
            initializer=initializers[t],
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, graph_collection_name],
            trainable=trainable) for t in xrange(num_timesteps)
    ]
    self.Bs = tf.cumsum(self.bs, reverse=True, axis=0)

  def posterior(self, observation, prev_state, t):
    """Computes the true posterior p(z_t|z_{t-1}, z_n)."""
    # bs[0] is really b_1
    # Bs[i] is sum from k=i+1^n b_k
    mu = observation - self.Bs[t]
    if t > 0:
      mu += (prev_state + self.bs[t - 1]) * float(self.num_timesteps - t)
    mu /= float(self.num_timesteps - t + 1)
    sigma = tf.ones_like(mu) * self.variance * (
        float(self.num_timesteps - t) / float(self.num_timesteps - t + 1))
    return tf.contrib.distributions.Normal(loc=mu, scale=tf.sqrt(sigma))

  def lookahead(self, state, t):
    """Computes the true lookahead distribution p(z_n|z_t)."""
    mu = state + self.Bs[t]
    sigma = tf.ones_like(state) * self.variance * float(self.num_timesteps - t)
    return tf.contrib.distributions.Normal(loc=mu, scale=tf.sqrt(sigma))

  def likelihood(self, observation):
    batch_size = tf.shape(observation)[0]
    mu = tf.tile(tf.reduce_sum(self.bs, axis=0)[tf.newaxis, :], [batch_size, 1])
    sigma = tf.ones_like(mu) * self.variance * (self.num_timesteps + 1)
    dist = tf.contrib.distributions.Normal(loc=mu, scale=tf.sqrt(sigma))
    # Average over the batch and take the sum over the state size
    return tf.reduce_mean(tf.reduce_sum(dist.log_prob(observation), axis=1))

  def p_zt(self, prev_state, t):
    """Computes the model p(z_t| z_{t-1})."""
    batch_size = tf.shape(prev_state)[0]
    if t > 0:
      z_mu_p = prev_state + self.bs[t - 1]
    else:  # p(z_0) is Normal(0,1)
      z_mu_p = tf.zeros([batch_size, self.state_size], dtype=self.dtype)
    p_zt = tf.contrib.distributions.Normal(
        loc=z_mu_p, scale=tf.sqrt(tf.ones_like(z_mu_p) * self.variance))
    return p_zt

  def generative(self, unused_observation, z_nm1):
    """Computes the model's generative distribution p(z_n| z_{n-1})."""
    generative_p_mu = z_nm1 + self.bs[-1]
    return tf.contrib.distributions.Normal(
        loc=generative_p_mu, scale=tf.sqrt(tf.ones_like(generative_p_mu) * self.variance))


class ShortChainNonlinearP(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               variance=1.0,
               observation_variance=1.0,
               transition_type=STANDARD_TRANSITION,
               transition_dist=tf.contrib.distributions.Normal,
               dtype=tf.float32,
               random_seed=None):
    self.state_size = state_size
    self.num_timesteps = num_timesteps
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.variance = variance
    self.observation_variance = observation_variance
    self.transition_type = transition_type
    self.transition_dist = transition_dist

  def p_zt(self, prev_state, t):
    """Computes the model p(z_t| z_{t-1})."""
    batch_size = tf.shape(prev_state)[0]
    if t > 0:
      if self.transition_type == ROUND_TRANSITION:
        loc = tf.round(prev_state)
        tf.logging.info("p(z_%d | z_%d) ~ N(round(z_%d), %0.1f)" % (t, t-1, t-1, self.variance))
      elif self.transition_type == STANDARD_TRANSITION:
        loc = prev_state
        tf.logging.info("p(z_%d | z_%d) ~ N(z_%d, %0.1f)" % (t, t-1, t-1, self.variance))
    else:  # p(z_0) is Normal(0,1)
      loc = tf.zeros([batch_size, self.state_size], dtype=self.dtype)
      tf.logging.info("p(z_0) ~ N(0,%0.1f)" % self.variance)

    p_zt = self.transition_dist(
        loc=loc,
        scale=tf.sqrt(tf.ones_like(loc) * self.variance))
    return p_zt

  def generative(self, unused_obs, z_ni):
    """Computes the model's generative distribution p(x_i| z_{ni})."""
    if self.transition_type == ROUND_TRANSITION:
      loc = tf.round(z_ni)
    elif self.transition_type == STANDARD_TRANSITION:
      loc = z_ni
    generative_sigma_sq = tf.ones_like(loc) * self.observation_variance
    return self.transition_dist(
        loc=loc, scale=tf.sqrt(generative_sigma_sq))


class BimodalPriorP(object):

  def __init__(self,
               state_size,
               num_timesteps,
               mixing_coeff=0.5,
               prior_mode_mean=1,
               sigma_min=1e-5,
               variance=1.0,
               dtype=tf.float32,
               random_seed=None,
               trainable=True,
               init_bs_to_zero=False,
               graph_collection_name="P_VARS"):
    self.state_size = state_size
    self.num_timesteps = num_timesteps
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.variance = variance
    self.mixing_coeff = mixing_coeff
    self.prior_mode_mean = prior_mode_mean

    if init_bs_to_zero:
      initializers = [tf.zeros_initializer for _ in xrange(num_timesteps)]
    else:
      initializers = [tf.random_uniform_initializer(seed=random_seed) for _ in xrange(num_timesteps)]

    self.bs = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="b_%d" % (t + 1),
            initializer=initializers[t],
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, graph_collection_name],
            trainable=trainable) for t in xrange(num_timesteps)
    ]
    self.Bs = tf.cumsum(self.bs, reverse=True, axis=0)

  def posterior(self, observation, prev_state, t):
    # NOTE: This is currently wrong, but would require a refactoring of
    # summarize_q to fix as kl is not defined for a mixture
    """Computes the true posterior p(z_t|z_{t-1}, z_n)."""
    # bs[0] is really b_1
    # Bs[i] is sum from k=i+1^n b_k
    mu = observation - self.Bs[t]
    if t > 0:
      mu += (prev_state + self.bs[t - 1]) * float(self.num_timesteps - t)
    mu /= float(self.num_timesteps - t + 1)
    sigma = tf.ones_like(mu) * self.variance * (
        float(self.num_timesteps - t) / float(self.num_timesteps - t + 1))
    return tf.contrib.distributions.Normal(loc=mu, scale=tf.sqrt(sigma))

  def lookahead(self, state, t):
    """Computes the true lookahead distribution p(z_n|z_t)."""
    mu = state + self.Bs[t]
    sigma = tf.ones_like(state) * self.variance * float(self.num_timesteps - t)
    return tf.contrib.distributions.Normal(loc=mu, scale=tf.sqrt(sigma))

  def likelihood(self, observation):
    batch_size = tf.shape(observation)[0]
    sum_of_bs = tf.tile(tf.reduce_sum(self.bs, axis=0)[tf.newaxis, :], [batch_size, 1])
    sigma = tf.ones_like(sum_of_bs) * self.variance * (self.num_timesteps + 1)
    mu_pos = (tf.ones([batch_size, self.state_size], dtype=self.dtype) * self.prior_mode_mean) + sum_of_bs
    mu_neg = (tf.ones([batch_size, self.state_size], dtype=self.dtype) * -self.prior_mode_mean) + sum_of_bs
    zn_pos = tf.contrib.distributions.Normal(
        loc=mu_pos,
        scale=tf.sqrt(sigma))
    zn_neg = tf.contrib.distributions.Normal(
        loc=mu_neg,
        scale=tf.sqrt(sigma))
    mode_probs = tf.convert_to_tensor([self.mixing_coeff, 1-self.mixing_coeff], dtype=tf.float64)
    mode_probs = tf.tile(mode_probs[tf.newaxis, tf.newaxis, :], [batch_size, 1, 1])
    mode_selection_dist = tf.contrib.distributions.Categorical(probs=mode_probs)
    zn_dist = tf.contrib.distributions.Mixture(
        cat=mode_selection_dist,
        components=[zn_pos, zn_neg],
        validate_args=True)
    # Average over the batch and take the sum over the state size
    return tf.reduce_mean(tf.reduce_sum(zn_dist.log_prob(observation), axis=1))

  def p_zt(self, prev_state, t):
    """Computes the model p(z_t| z_{t-1})."""
    batch_size = tf.shape(prev_state)[0]
    if t > 0:
      z_mu_p = prev_state + self.bs[t - 1]
      p_zt = tf.contrib.distributions.Normal(
          loc=z_mu_p, scale=tf.sqrt(tf.ones_like(z_mu_p) * self.variance))
      return p_zt
    else:  # p(z_0) is mixture of two Normals
      mu_pos = tf.ones([batch_size, self.state_size], dtype=self.dtype) * self.prior_mode_mean
      mu_neg = tf.ones([batch_size, self.state_size], dtype=self.dtype) * -self.prior_mode_mean
      z0_pos = tf.contrib.distributions.Normal(
          loc=mu_pos,
          scale=tf.sqrt(tf.ones_like(mu_pos) * self.variance))
      z0_neg = tf.contrib.distributions.Normal(
          loc=mu_neg,
          scale=tf.sqrt(tf.ones_like(mu_neg) * self.variance))
      mode_probs = tf.convert_to_tensor([self.mixing_coeff, 1-self.mixing_coeff], dtype=tf.float64)
      mode_probs = tf.tile(mode_probs[tf.newaxis, tf.newaxis, :], [batch_size, 1, 1])
      mode_selection_dist = tf.contrib.distributions.Categorical(probs=mode_probs)
      z0_dist = tf.contrib.distributions.Mixture(
          cat=mode_selection_dist,
          components=[z0_pos, z0_neg],
          validate_args=False)
      return z0_dist

  def generative(self, unused_observation, z_nm1):
    """Computes the model's generative distribution p(z_n| z_{n-1})."""
    generative_p_mu = z_nm1 + self.bs[-1]
    return tf.contrib.distributions.Normal(
        loc=generative_p_mu, scale=tf.sqrt(tf.ones_like(generative_p_mu) * self.variance))

class Model(object):

  def __init__(self,
               p,
               q,
               r,
               state_size,
               num_timesteps,
               dtype=tf.float32):
    self.p = p
    self.q = q
    self.r = r
    self.state_size = state_size
    self.num_timesteps = num_timesteps
    self.dtype = dtype

  def zero_state(self, batch_size):
    return tf.zeros([batch_size, self.state_size], dtype=self.dtype)

  def __call__(self, prev_state, observation, t):
    # Compute the q distribution over z, q(z_t|z_n, z_{t-1}).
    q_zt = self.q.q_zt(observation, prev_state, t)
    # Compute the p distribution over z, p(z_t|z_{t-1}).
    p_zt = self.p.p_zt(prev_state, t)
    # sample from q
    zt = q_zt.sample()
    r_xn = self.r.r_xn(zt, t)
    # Calculate the logprobs and sum over the state size.
    log_q_zt = tf.reduce_sum(q_zt.log_prob(zt), axis=1)
    log_p_zt = tf.reduce_sum(p_zt.log_prob(zt), axis=1)
    log_r_xn = tf.reduce_sum(r_xn.log_prob(observation), axis=1)
    # If we're at the last timestep, also calc the logprob of the observation.
    if t == self.num_timesteps - 1:
      generative_dist = self.p.generative(observation, zt)
      log_p_x_given_z = tf.reduce_sum(generative_dist.log_prob(observation), axis=1)
    else:
      log_p_x_given_z = tf.zeros_like(log_q_zt)
    return (zt, log_q_zt, log_p_zt, log_p_x_given_z, log_r_xn)

  @staticmethod
  def create(state_size,
             num_timesteps,
             sigma_min=1e-5,
             r_sigma_init=1,
             variance=1.0,
             mixing_coeff=0.5,
             prior_mode_mean=1.0,
             dtype=tf.float32,
             random_seed=None,
             train_p=True,
             p_type="unimodal",
             q_type="normal",
             observation_variance=1.0,
             transition_type=STANDARD_TRANSITION,
             use_bs=True):
    if p_type == "unimodal":
      p = P(state_size,
            num_timesteps,
            sigma_min=sigma_min,
            variance=variance,
            dtype=dtype,
            random_seed=random_seed,
            trainable=train_p,
            init_bs_to_zero=not use_bs)
    elif p_type == "bimodal":
      p = BimodalPriorP(
          state_size,
          num_timesteps,
          mixing_coeff=mixing_coeff,
          prior_mode_mean=prior_mode_mean,
          sigma_min=sigma_min,
          variance=variance,
          dtype=dtype,
          random_seed=random_seed,
          trainable=train_p,
          init_bs_to_zero=not use_bs)
    elif "nonlinear" in p_type:
      if "cauchy" in p_type:
        trans_dist = tf.contrib.distributions.Cauchy
      else:
        trans_dist = tf.contrib.distributions.Normal
      p = ShortChainNonlinearP(
          state_size,
          num_timesteps,
          sigma_min=sigma_min,
          variance=variance,
          observation_variance=observation_variance,
          transition_type=transition_type,
          transition_dist=trans_dist,
          dtype=dtype,
          random_seed=random_seed
      )

    if q_type == "normal":
      q_class = Q
    elif q_type == "simple_mean":
      q_class = SimpleMeanQ
    elif q_type == "prev_state":
      q_class = PreviousStateQ
    elif q_type == "observation":
      q_class = ObservationQ

    q = q_class(state_size,
                num_timesteps,
                sigma_min=sigma_min,
                dtype=dtype,
                random_seed=random_seed,
                init_mu0_to_zero=not use_bs)
    r = R(state_size,
          num_timesteps,
          sigma_min=sigma_min,
          sigma_init=r_sigma_init,
          dtype=dtype,
          random_seed=random_seed)
    model = Model(p, q, r, state_size, num_timesteps, dtype=dtype)
    return model


class BackwardsModel(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               dtype=tf.float32):
    self.state_size = state_size
    self.num_timesteps = num_timesteps
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.bs = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="b_%d" % (t + 1),
            initializer=tf.zeros_initializer) for t in xrange(num_timesteps)
    ]
    self.Bs = tf.cumsum(self.bs, reverse=True, axis=0)
    self.q_mus = [
        snt.Linear(output_size=state_size) for _ in xrange(num_timesteps)
    ]
    self.q_sigmas = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="q_sigma_%d" % (t + 1),
            initializer=tf.zeros_initializer) for t in xrange(num_timesteps)
    ]
    self.r_mus = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="r_mu_%d" % (t + 1),
            initializer=tf.zeros_initializer) for t in xrange(num_timesteps)
    ]
    self.r_sigmas = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="r_sigma_%d" % (t + 1),
            initializer=tf.zeros_initializer) for t in xrange(num_timesteps)
    ]

  def zero_state(self, batch_size):
    return tf.zeros([batch_size, self.state_size], dtype=self.dtype)

  def posterior(self, unused_observation, prev_state, unused_t):
    # TODO(dieterichl): Correct this.
    return tf.contrib.distributions.Normal(
        loc=tf.zeros_like(prev_state), scale=tf.zeros_like(prev_state))

  def lookahead(self, state, unused_t):
    # TODO(dieterichl): Correct this.
    return tf.contrib.distributions.Normal(
        loc=tf.zeros_like(state), scale=tf.zeros_like(state))

  def q_zt(self, observation, next_state, t):
    """Computes the variational posterior q(z_{t}|z_{t+1}, z_n)."""
    t_backwards = self.num_timesteps - t - 1
    batch_size = tf.shape(next_state)[0]
    q_mu = self.q_mus[t_backwards](tf.concat([observation, next_state], axis=1))
    q_sigma = tf.maximum(
        tf.nn.softplus(self.q_sigmas[t_backwards]), self.sigma_min)
    q_sigma = tf.tile(q_sigma[tf.newaxis, :], [batch_size, 1])
    q_zt = tf.contrib.distributions.Normal(loc=q_mu, scale=tf.sqrt(q_sigma))
    return q_zt

  def p_zt(self, prev_state, t):
    """Computes the model p(z_{t+1}| z_{t})."""
    t_backwards = self.num_timesteps - t - 1
    z_mu_p = prev_state + self.bs[t_backwards]
    p_zt = tf.contrib.distributions.Normal(
        loc=z_mu_p, scale=tf.ones_like(z_mu_p))
    return p_zt

  def generative(self, unused_observation, z_nm1):
    """Computes the model's generative distribution p(z_n| z_{n-1})."""
    generative_p_mu = z_nm1 + self.bs[-1]
    return tf.contrib.distributions.Normal(
        loc=generative_p_mu, scale=tf.ones_like(generative_p_mu))

  def r(self, z_t, t):
    t_backwards = self.num_timesteps - t - 1
    batch_size = tf.shape(z_t)[0]
    r_mu = tf.tile(self.r_mus[t_backwards][tf.newaxis, :], [batch_size, 1])
    r_sigma = tf.maximum(
        tf.nn.softplus(self.r_sigmas[t_backwards]), self.sigma_min)
    r_sigma = tf.tile(r_sigma[tf.newaxis, :], [batch_size, 1])
    return tf.contrib.distributions.Normal(loc=r_mu, scale=tf.sqrt(r_sigma))

  def likelihood(self, observation):
    batch_size = tf.shape(observation)[0]
    mu = tf.tile(tf.reduce_sum(self.bs, axis=0)[tf.newaxis, :], [batch_size, 1])
    sigma = tf.ones_like(mu) * (self.num_timesteps + 1)
    dist = tf.contrib.distributions.Normal(loc=mu, scale=tf.sqrt(sigma))
    # Average over the batch and take the sum over the state size
    return tf.reduce_mean(tf.reduce_sum(dist.log_prob(observation), axis=1))

  def __call__(self, next_state, observation, t):
    # next state = z_{t+1}
    # Compute the q distribution over z, q(z_{t}|z_n, z_{t+1}).
    q_zt = self.q_zt(observation, next_state, t)
    # sample from q
    zt = q_zt.sample()
    # Compute the p distribution over z, p(z_{t+1}|z_{t}).
    p_zt = self.p_zt(zt, t)
    # Compute log p(z_{t+1} | z_t)
    if t == 0:
      log_p_zt = p_zt.log_prob(observation)
    else:
      log_p_zt = p_zt.log_prob(next_state)

    # Compute r prior over zt
    r_zt = self.r(zt, t)
    log_r_zt = r_zt.log_prob(zt)
    # Compute proposal density at zt
    log_q_zt = q_zt.log_prob(zt)
    # If we're at the last timestep, also calc the logprob of the observation.

    if t == self.num_timesteps - 1:
      p_z0_dist = tf.contrib.distributions.Normal(
          loc=tf.zeros_like(zt), scale=tf.ones_like(zt))
      z0_log_prob = p_z0_dist.log_prob(zt)
    else:
      z0_log_prob = tf.zeros_like(log_q_zt)
    return (zt, log_q_zt, log_p_zt, z0_log_prob, log_r_zt)


class LongChainP(object):

  def __init__(self,
               state_size,
               num_obs,
               steps_per_obs,
               sigma_min=1e-5,
               variance=1.0,
               observation_variance=1.0,
               observation_type=STANDARD_OBSERVATION,
               transition_type=STANDARD_TRANSITION,
               dtype=tf.float32,
               random_seed=None):
    self.state_size = state_size
    self.steps_per_obs = steps_per_obs
    self.num_obs = num_obs
    self.num_timesteps = steps_per_obs*num_obs + 1
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.variance = variance
    self.observation_variance = observation_variance
    self.observation_type = observation_type
    self.transition_type = transition_type

  def likelihood(self, observations):
    """Computes the model's true likelihood of the observations.

    Args:
      observations: A [batch_size, m, state_size] Tensor representing each of
        the m observations.
    Returns:
      logprob: The true likelihood of the observations given the model.
    """
    raise ValueError("Likelihood is not defined for long-chain models")
    # batch_size = tf.shape(observations)[0]
    # mu = tf.zeros([batch_size, self.state_size, self.num_obs], dtype=self.dtype)
    # sigma = np.fromfunction(
    #     lambda i, j: 1 + self.steps_per_obs*np.minimum(i+1, j+1),
    #     [self.num_obs, self.num_obs])
    # sigma += np.eye(self.num_obs)
    # sigma = tf.convert_to_tensor(sigma * self.variance, dtype=self.dtype)
    # sigma = tf.tile(sigma[tf.newaxis, tf.newaxis, ...],
    #                 [batch_size, self.state_size, 1, 1])
    # dist = tf.contrib.distributions.MultivariateNormalFullCovariance(
    #     loc=mu,
    #     covariance_matrix=sigma)
    # Average over the batch and take the sum over the state size
    #return tf.reduce_mean(tf.reduce_sum(dist.log_prob(observations), axis=1))

  def p_zt(self, prev_state, t):
    """Computes the model p(z_t| z_{t-1})."""
    batch_size = tf.shape(prev_state)[0]
    if t > 0:
      if self.transition_type == ROUND_TRANSITION:
        loc = tf.round(prev_state)
        tf.logging.info("p(z_%d | z_%d) ~ N(round(z_%d), %0.1f)" % (t, t-1, t-1, self.variance))
      elif self.transition_type == STANDARD_TRANSITION:
        loc = prev_state
        tf.logging.info("p(z_%d | z_%d) ~ N(z_%d, %0.1f)" % (t, t-1, t-1, self.variance))
    else:  # p(z_0) is Normal(0,1)
      loc = tf.zeros([batch_size, self.state_size], dtype=self.dtype)
      tf.logging.info("p(z_0) ~ N(0,%0.1f)" % self.variance)

    p_zt = tf.contrib.distributions.Normal(
        loc=loc,
        scale=tf.sqrt(tf.ones_like(loc) * self.variance))
    return p_zt

  def generative(self, z_ni, t):
    """Computes the model's generative distribution p(x_i| z_{ni})."""
    if self.observation_type == SQUARED_OBSERVATION:
      generative_mu = tf.square(z_ni)
      tf.logging.info("p(x_%d | z_%d) ~ N(z_%d^2, %0.1f)" % (t, t, t, self.variance))
    elif self.observation_type == ABS_OBSERVATION:
      generative_mu = tf.abs(z_ni)
      tf.logging.info("p(x_%d | z_%d) ~ N(|z_%d|, %0.1f)" % (t, t, t, self.variance))
    elif self.observation_type == STANDARD_OBSERVATION:
      generative_mu = z_ni
      tf.logging.info("p(x_%d | z_%d) ~ N(z_%d, %0.1f)" % (t, t, t, self.variance))
    generative_sigma_sq = tf.ones_like(generative_mu) * self.observation_variance
    return tf.contrib.distributions.Normal(
        loc=generative_mu, scale=tf.sqrt(generative_sigma_sq))


class LongChainQ(object):

  def __init__(self,
               state_size,
               num_obs,
               steps_per_obs,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None):
    self.state_size = state_size
    self.sigma_min = sigma_min
    self.dtype = dtype
    self.steps_per_obs = steps_per_obs
    self.num_obs = num_obs
    self.num_timesteps = num_obs*steps_per_obs +1

    initializers =  {
      "w": tf.random_uniform_initializer(seed=random_seed),
      "b": tf.zeros_initializer
    }
    self.mus = [
        snt.Linear(output_size=state_size, initializers=initializers)
        for t in xrange(self.num_timesteps)
    ]
    self.sigmas = [
        tf.get_variable(
            shape=[state_size],
            dtype=self.dtype,
            name="q_sigma_%d" % (t + 1),
            initializer=tf.random_uniform_initializer(seed=random_seed))
        for t in xrange(self.num_timesteps)
    ]

  def first_relevant_obs_index(self, t):
    return int(max((t-1)/self.steps_per_obs, 0))

  def q_zt(self, observations, prev_state, t):
    """Computes a distribution over z_t.

    Args:
      observations: a [batch_size, num_observations, state_size] Tensor.
      prev_state: a [batch_size, state_size] Tensor.
      t: The current timestep, an int Tensor.
    """
    # filter out unneeded past obs
    first_relevant_obs_index = int(math.floor(max(t-1, 0) / self.steps_per_obs))
    num_relevant_observations = self.num_obs - first_relevant_obs_index
    observations = observations[:,first_relevant_obs_index:,:]
    batch_size = tf.shape(prev_state)[0]
    # concatenate the prev state and observations along the second axis (that is
    # not the batch or state size axis, and then flatten it to
    # [batch_size, (num_relevant_observations + 1) * state_size] to feed it into
    # the linear layer.
    q_input = tf.concat([observations, prev_state[:,tf.newaxis, :]], axis=1)
    q_input = tf.reshape(q_input,
                         [batch_size, (num_relevant_observations + 1) * self.state_size])
    q_mu = self.mus[t](q_input)
    q_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    q_sigma = tf.tile(q_sigma[tf.newaxis, :], [batch_size, 1])
    q_zt = tf.contrib.distributions.Normal(loc=q_mu, scale=tf.sqrt(q_sigma))
    tf.logging.info(
        "q(z_{t} | z_{tm1}, x_{obsf}:{obst}) ~ N(Linear([z_{tm1},x_{obsf}:{obst}]), sigma_{t})".format(
            **{"t": t,
               "tm1": t-1,
               "obsf": (first_relevant_obs_index+1)*self.steps_per_obs,
               "obst":self.steps_per_obs*self.num_obs}))
    return q_zt

  def summarize_weights(self):
    pass

class LongChainR(object):

  def __init__(self,
               state_size,
               num_obs,
               steps_per_obs,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None):
    self.state_size = state_size
    self.dtype = dtype
    self.sigma_min = sigma_min
    self.steps_per_obs = steps_per_obs
    self.num_obs = num_obs
    self.num_timesteps = num_obs*steps_per_obs + 1
    self.sigmas = [
        tf.get_variable(
            shape=[self.num_future_obs(t)],
            dtype=self.dtype,
            name="r_sigma_%d" % (t + 1),
            #initializer=tf.random_uniform_initializer(seed=random_seed, maxval=100))
            initializer=tf.constant_initializer(1.0))
        for t in range(self.num_timesteps)
    ]

  def first_future_obs_index(self, t):
    return int(math.floor(t / self.steps_per_obs))

  def num_future_obs(self, t):
    return int(self.num_obs - self.first_future_obs_index(t))

  def r_xn(self, z_t, t):
    """Computes a distribution over the future observations given current latent
    state.

    The indexing in these messages is 1 indexed and inclusive. This is
    consistent with the latex documents.

    Args:
      z_t: [batch_size, state_size] Tensor
      t: Current timestep
    """
    tf.logging.info(
        "r(x_{start}:{end} | z_{t}) ~ N(z_{t}, sigma_{t})".format(
            **{"t": t,
               "start": (self.first_future_obs_index(t)+1)*self.steps_per_obs,
               "end": self.num_timesteps-1}))
    batch_size = tf.shape(z_t)[0]
    # the mean for all future observations is the same.
    # this tiling results in a [batch_size, num_future_obs, state_size] Tensor
    r_mu = tf.tile(z_t[:,tf.newaxis,:], [1, self.num_future_obs(t), 1])
    # compute the variance
    r_sigma = tf.maximum(tf.nn.softplus(self.sigmas[t]), self.sigma_min)
    # the variance is the same across all state dimensions, so we only have to
    # time sigma to be [batch_size, num_future_obs].
    r_sigma = tf.tile(r_sigma[tf.newaxis,:, tf.newaxis], [batch_size, 1, self.state_size])
    return tf.contrib.distributions.Normal(
        loc=r_mu, scale=tf.sqrt(r_sigma))

  def summarize_weights(self):
    pass


class LongChainModel(object):

  def __init__(self,
               p,
               q,
               r,
               state_size,
               num_obs,
               steps_per_obs,
               dtype=tf.float32,
               disable_r=False):
    self.p = p
    self.q = q
    self.r = r
    self.disable_r = disable_r
    self.state_size = state_size
    self.num_obs = num_obs
    self.steps_per_obs = steps_per_obs
    self.num_timesteps = steps_per_obs*num_obs + 1
    self.dtype = dtype

  def zero_state(self, batch_size):
    return tf.zeros([batch_size, self.state_size], dtype=self.dtype)

  def next_obs_ind(self, t):
    return int(math.floor(max(t-1,0)/self.steps_per_obs))

  def __call__(self, prev_state, observations, t):
    """Computes the importance weight for the model system.

    Args:
      prev_state: [batch_size, state_size] Tensor
      observations: [batch_size, num_observations, state_size] Tensor
    """
    # Compute the q distribution over z, q(z_t|z_n, z_{t-1}).
    q_zt = self.q.q_zt(observations, prev_state, t)
    # Compute the p distribution over z, p(z_t|z_{t-1}).
    p_zt = self.p.p_zt(prev_state, t)
    # sample from q and evaluate the logprobs, summing over the state size
    zt = q_zt.sample()
    log_q_zt = tf.reduce_sum(q_zt.log_prob(zt), axis=1)
    log_p_zt = tf.reduce_sum(p_zt.log_prob(zt), axis=1)
    if not self.disable_r and t < self.num_timesteps-1:
      # score the remaining observations using r
      r_xn = self.r.r_xn(zt, t)
      log_r_xn = r_xn.log_prob(observations[:, self.next_obs_ind(t+1):, :])
      # sum over state size and observation, leaving the batch index
      log_r_xn = tf.reduce_sum(log_r_xn, axis=[1,2])
    else:
      log_r_xn = tf.zeros_like(log_p_zt)
    if t != 0 and t % self.steps_per_obs == 0:
      generative_dist = self.p.generative(zt, t)
      log_p_x_given_z = generative_dist.log_prob(observations[:,self.next_obs_ind(t),:])
      log_p_x_given_z = tf.reduce_sum(log_p_x_given_z, axis=1)
    else:
      log_p_x_given_z = tf.zeros_like(log_q_zt)
    return (zt, log_q_zt, log_p_zt, log_p_x_given_z, log_r_xn)

  @staticmethod
  def create(state_size,
             num_obs,
             steps_per_obs,
             sigma_min=1e-5,
             variance=1.0,
             observation_variance=1.0,
             observation_type=STANDARD_OBSERVATION,
             transition_type=STANDARD_TRANSITION,
             dtype=tf.float32,
             random_seed=None,
             disable_r=False):
    p = LongChainP(
        state_size,
        num_obs,
        steps_per_obs,
        sigma_min=sigma_min,
        variance=variance,
        observation_variance=observation_variance,
        observation_type=observation_type,
        transition_type=transition_type,
        dtype=dtype,
        random_seed=random_seed)
    q = LongChainQ(
        state_size,
        num_obs,
        steps_per_obs,
        sigma_min=sigma_min,
        dtype=dtype,
        random_seed=random_seed)
    r = LongChainR(
        state_size,
        num_obs,
        steps_per_obs,
        sigma_min=sigma_min,
        dtype=dtype,
        random_seed=random_seed)
    model = LongChainModel(
        p, q, r, state_size, num_obs, steps_per_obs,
        dtype=dtype,
        disable_r=disable_r)
    return model


class RTilde(object):

  def __init__(self,
               state_size,
               num_timesteps,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None,
               graph_collection_name="R_TILDE_VARS"):
    self.dtype = dtype
    self.sigma_min = sigma_min
    initializers = {"w": tf.truncated_normal_initializer(seed=random_seed),
                    "b": tf.zeros_initializer}
    self.graph_collection_name=graph_collection_name

    def custom_getter(getter, *args, **kwargs):
      out = getter(*args, **kwargs)
      ref = tf.get_collection_ref(self.graph_collection_name)
      if out not in ref:
        ref.append(out)
      return out

    self.fns = [
        snt.Linear(output_size=2*state_size,
                   initializers=initializers,
                   name="r_tilde_%d" % t,
                   custom_getter=custom_getter)
        for t in xrange(num_timesteps)
    ]

  def r_zt(self, z_t, observation, t):
    #out = self.fns[t](tf.stop_gradient(tf.concat([z_t, observation], axis=1)))
    out = self.fns[t](tf.concat([z_t, observation], axis=1))
    mu, raw_sigma_sq = tf.split(out, 2, axis=1)
    sigma_sq = tf.maximum(tf.nn.softplus(raw_sigma_sq), self.sigma_min)
    return mu, sigma_sq

class TDModel(object):

  def __init__(self,
               p,
               q,
               r_tilde,
               state_size,
               num_timesteps,
               dtype=tf.float32,
               disable_r=False):
    self.p = p
    self.q = q
    self.r_tilde = r_tilde
    self.disable_r = disable_r
    self.state_size = state_size
    self.num_timesteps = num_timesteps
    self.dtype = dtype

  def zero_state(self, batch_size):
    return tf.zeros([batch_size, self.state_size], dtype=self.dtype)

  def __call__(self, prev_state, observation, t):
    """Computes the importance weight for the model system.

    Args:
      prev_state: [batch_size, state_size] Tensor
      observations: [batch_size, num_observations, state_size] Tensor
    """
    # Compute the q distribution over z, q(z_t|z_n, z_{t-1}).
    q_zt = self.q.q_zt(observation, prev_state, t)
    # Compute the p distribution over z, p(z_t|z_{t-1}).
    p_zt = self.p.p_zt(prev_state, t)
    # sample from q and evaluate the logprobs, summing over the state size
    zt = q_zt.sample()
    # If it isn't the last timestep, compute the distribution over the next z.
    if t < self.num_timesteps - 1:
      p_ztplus1 = self.p.p_zt(zt, t+1)
    else:
      p_ztplus1 = None
    log_q_zt = tf.reduce_sum(q_zt.log_prob(zt), axis=1)
    log_p_zt = tf.reduce_sum(p_zt.log_prob(zt), axis=1)

    if not self.disable_r and t < self.num_timesteps-1:
      # score the remaining observations using r
      r_tilde_mu, r_tilde_sigma_sq = self.r_tilde.r_zt(zt, observation, t+1)
    else:
      r_tilde_mu = None
      r_tilde_sigma_sq = None
    if t == self.num_timesteps - 1:
      generative_dist = self.p.generative(observation, zt)
      log_p_x_given_z = tf.reduce_sum(generative_dist.log_prob(observation), axis=1)
    else:
      log_p_x_given_z = tf.zeros_like(log_q_zt)
    return (zt, log_q_zt, log_p_zt, log_p_x_given_z,
            r_tilde_mu, r_tilde_sigma_sq, p_ztplus1)

  @staticmethod
  def create(state_size,
             num_timesteps,
             sigma_min=1e-5,
             variance=1.0,
             dtype=tf.float32,
             random_seed=None,
             train_p=True,
             p_type="unimodal",
             q_type="normal",
             mixing_coeff=0.5,
             prior_mode_mean=1.0,
             observation_variance=1.0,
             transition_type=STANDARD_TRANSITION,
             use_bs=True):
    if p_type == "unimodal":
      p = P(state_size,
            num_timesteps,
            sigma_min=sigma_min,
            variance=variance,
            dtype=dtype,
            random_seed=random_seed,
            trainable=train_p,
            init_bs_to_zero=not use_bs)
    elif p_type == "bimodal":
      p = BimodalPriorP(
          state_size,
          num_timesteps,
          mixing_coeff=mixing_coeff,
          prior_mode_mean=prior_mode_mean,
          sigma_min=sigma_min,
          variance=variance,
          dtype=dtype,
          random_seed=random_seed,
          trainable=train_p,
          init_bs_to_zero=not use_bs)
    elif "nonlinear" in p_type:
      if "cauchy" in p_type:
        trans_dist = tf.contrib.distributions.Cauchy
      else:
        trans_dist = tf.contrib.distributions.Normal

      p = ShortChainNonlinearP(
          state_size,
          num_timesteps,
          sigma_min=sigma_min,
          variance=variance,
          observation_variance=observation_variance,
          transition_type=transition_type,
          transition_dist=trans_dist,
          dtype=dtype,
          random_seed=random_seed
      )

    if q_type == "normal":
      q_class = Q
    elif q_type == "simple_mean":
      q_class = SimpleMeanQ
    elif q_type == "prev_state":
      q_class = PreviousStateQ
    elif q_type == "observation":
      q_class = ObservationQ

    q = q_class(state_size,
                num_timesteps,
                sigma_min=sigma_min,
                dtype=dtype,
                random_seed=random_seed,
                init_mu0_to_zero=not use_bs)
    r_tilde = RTilde(
        state_size,
        num_timesteps,
        sigma_min=sigma_min,
        dtype=dtype,
        random_seed=random_seed)
    model = TDModel(p, q, r_tilde, state_size, num_timesteps, dtype=dtype)
    return model
