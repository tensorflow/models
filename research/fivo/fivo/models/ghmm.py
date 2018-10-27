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

"""A Gaussian hidden markov model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fivo.models import base

tfd = tf.contrib.distributions


class GaussianHMM(object):
  """A hidden markov model with 1-D Gaussian latent space and observations.

  This is a hidden markov model where the state and observations are
  one-dimensional Gaussians. The mean of each latent state is a linear
  function of the previous latent state, and the mean of each observation
  is a linear function of the current latent state.

  The description that follows is 0-indexed instead of 1-indexed to make
  it easier to reason about the parameters passed to the model.

  The parameters of the model are:
    T: The number timesteps, latent states, and observations.
    vz_t, t=0 to T-1: The variance of the latent state at timestep t.
    vx_t, t=0 to T-1: The variance of the observation at timestep t.
    wz_t, t=1 to T-1: The weight that defines the latent transition at t.
    wx_t, t=0 to T-1: The weight that defines the observation function at t.

  There are T vz_t, vx_t, and wx_t but only T-1 wz_t because there are only
  T-1 transitions in the model.

  Given these parameters, sampling from the model is defined as

    z_0 ~ N(0, vz_0)
    x_0 | z_0 ~ N(wx_0 * z_0, vx_0)
    z_1 | z_0 ~ N(wz_1 * z_0, vz_1)
    x_1 | z_1 ~ N(wx_1 * z_1, vx_1)
    ...
    z_{T-1} | z_{T-2} ~ N(wz_{T-1} * z_{T-2}, vz_{T-1})
    x_{T-1} | z_{T-1} ~ N(wx_{T-1} * z_{T-1}, vx_{T-1}).
  """

  def __init__(self,
               num_timesteps,
               transition_variances=1.,
               emission_variances=1.,
               transition_weights=1.,
               emission_weights=1.,
               dtype=tf.float32):
    """Creates a gaussian hidden markov model.

    Args:
      num_timesteps: A python int, the number of timesteps in the model.
      transition_variances: The variance of p(z_t | z_t-1). Can be a scalar,
        setting all variances to be the same, or a Tensor of shape
        [num_timesteps].
      emission_variances: The variance of p(x_t | z_t). Can be a scalar,
        setting all variances to be the same, or a Tensor of shape
        [num_timesteps].
      transition_weights: The weight that defines the linear function that
        produces the mean of z_t given z_{t-1}. Can be a scalar, setting
        all weights to be the same, or a Tensor of shape [num_timesteps-1].
      emission_weights: The weight that defines the linear function that
        produces the mean of x_t given z_t. Can be a scalar, setting
        all weights to be the same, or a Tensor of shape [num_timesteps].
      dtype: The datatype of the state.
    """
    self.num_timesteps = num_timesteps
    self.dtype = dtype

    def _expand_param(param, size):
      param = tf.convert_to_tensor(param, dtype=self.dtype)
      if not param.get_shape().as_list():
        param = tf.tile(param[tf.newaxis], [size])

      return param

    def _ta_for_param(param):
      size = tf.shape(param)[0]
      ta = tf.TensorArray(dtype=param.dtype,
                          size=size,
                          dynamic_size=False,
                          clear_after_read=False).unstack(param)
      return ta

    self.transition_variances = _ta_for_param(
        _expand_param(transition_variances, num_timesteps))
    self.transition_weights = _ta_for_param(
        _expand_param(transition_weights, num_timesteps-1))
    em_var = _expand_param(emission_variances, num_timesteps)
    self.emission_variances = _ta_for_param(em_var)
    em_w = _expand_param(emission_weights, num_timesteps)
    self.emission_weights = _ta_for_param(em_w)
    self._compute_covariances(em_w, em_var)

  def _compute_covariances(self, emission_weights, emission_variances):
    """Compute all covariance matrices.

    Computes the covaraince matrix for the latent variables, the observations,
    and the covariance between the latents and observations.

    Args:
      emission_weights: A Tensor of shape [num_timesteps] containing
        the emission distribution weights at each timestep.
      emission_variances: A Tensor of shape [num_timesteps] containing
        the emiision distribution variances at each timestep.
    """
    # Compute the marginal variance of each latent.
    z_variances = [self.transition_variances.read(0)]
    for i in range(1, self.num_timesteps):
      z_variances.append(
          z_variances[i-1] * tf.square(self.transition_weights.read(i-1)) +
          self.transition_variances.read(i))
    # Compute the latent covariance matrix.
    sigma_z = []
    for i in range(self.num_timesteps):
      sigma_z_row = []
      for j in range(self.num_timesteps):
        if i == j:
          sigma_z_row.append(z_variances[i])
          continue
        min_ind = min(i, j)
        max_ind = max(i, j)
        weight = tf.reduce_prod(
            self.transition_weights.gather(tf.range(min_ind, max_ind)))
        sigma_z_row.append(z_variances[min_ind] * weight)
      sigma_z.append(tf.stack(sigma_z_row))
    self.sigma_z = tf.stack(sigma_z)
    # Compute the observation covariance matrix.
    x_weights_outer = tf.einsum("i,j->ij", emission_weights, emission_weights)
    self.sigma_x = x_weights_outer * self.sigma_z + tf.diag(emission_variances)
    # Compute the latent - observation covariance matrix.
    # The first axis will index latents, the second axis will index observtions.
    self.sigma_zx = emission_weights[tf.newaxis, :] * self.sigma_z
    self.obs_dist = tfd.MultivariateNormalFullCovariance(
        loc=tf.zeros([self.num_timesteps], dtype=tf.float32),
        covariance_matrix=self.sigma_x)

  def transition(self, t, z_prev):
    """Compute the transition distribution p(z_t | z_t-1).

    Args:
      t: The current timestep, a scalar integer Tensor. When t=0 z_prev is
        mostly ignored and the distribution p(z_0) is returned. z_prev is
        'mostly' ignored because it is still used to derive batch_size.
      z_prev: A [batch_size] set of states.
    Returns:
      p(z_t | z_t-1) as a univariate normal distribution.
    """
    batch_size = tf.shape(z_prev)[0]
    scale = tf.sqrt(self.transition_variances.read(t))
    scale = tf.tile(scale[tf.newaxis], [batch_size])
    loc = tf.cond(tf.greater(t, 0),
                  lambda: self.transition_weights.read(t-1)*z_prev,
                  lambda: tf.zeros_like(scale))
    return tfd.Normal(loc=loc, scale=scale)

  def emission(self, t, z):
    """Compute the emission distribution p(x_t | z_t).

    Args:
      t: The current timestep, a scalar integer Tensor.
      z: A [batch_size] set of the current states.
    Returns:
      p(x_t | z_t) as a univariate normal distribution.
    """
    batch_size = tf.shape(z)[0]
    scale = tf.sqrt(self.emission_variances.read(t))
    scale = tf.tile(scale[tf.newaxis], [batch_size])
    loc = self.emission_weights.read(t)*z
    return tfd.Normal(loc=loc, scale=scale)

  def filtering(self, t, z_prev, x_cur):
    """Computes the filtering distribution p(z_t | z_{t-1}, x_t).

    Args:
      t: A python int, the index for z_t. When t is 0, z_prev is ignored,
        giving p(z_0 | x_0).
      z_prev: z_{t-1}, the previous z to condition on. A Tensor of shape
        [batch_size].
      x_cur: x_t, the current x to condition on. A Tensor of shape [batch_size].
    Returns:
      p(z_t | z_{t-1}, x_t) as a univariate normal distribution.
    """
    z_prev = tf.convert_to_tensor(z_prev)
    x_cur = tf.convert_to_tensor(x_cur)
    batch_size = tf.shape(z_prev)[0]
    z_var = self.transition_variances.read(t)
    x_var = self.emission_variances.read(t)
    x_weight = self.emission_weights.read(t)
    prev_state_weight = x_var/(tf.square(x_weight)*z_var + x_var)
    prev_state_weight *= tf.cond(tf.greater(t, 0),
                                 lambda: self.transition_weights.read(t-1),
                                 lambda: tf.zeros_like(prev_state_weight))
    cur_obs_weight = (x_weight*z_var)/(tf.square(x_weight)*z_var + x_var)
    loc = prev_state_weight*z_prev + cur_obs_weight*x_cur
    scale = tf.sqrt((z_var*x_var)/(tf.square(x_weight)*z_var + x_var))
    scale = tf.tile(scale[tf.newaxis], [batch_size])
    return tfd.Normal(loc=loc, scale=scale)

  def smoothing(self, t, z_prev, xs):
    """Computes the smoothing distribution p(z_t | z_{t-1}, x_{t:num_timesteps).

    Args:
      t: A python int, the index for z_t. When t is 0, z_prev is ignored,
        giving p(z_0 | x_{0:num_timesteps-1}).
      z_prev: z_{t-1}, the previous z to condition on. A Tensor of shape
        [batch_size].
      xs: x_{t:num_timesteps}, the future xs to condition on. A Tensor of shape
        [num_timesteps - t, batch_size].
    Returns:
      p(z_t | z_{t-1}, x_{t:num_timesteps}) as a univariate normal distribution.
    """
    xs = tf.convert_to_tensor(xs)
    z_prev = tf.convert_to_tensor(z_prev)
    batch_size = tf.shape(xs)[1]
    mess_mean, mess_prec = tf.cond(
        tf.less(t, self.num_timesteps-1),
        lambda: tf.unstack(self._compute_backwards_messages(xs[1:]).read(0)),
        lambda: [tf.zeros([batch_size]), tf.zeros([batch_size])])
    return self._smoothing_from_message(t, z_prev, xs[0], mess_mean, mess_prec)

  def _smoothing_from_message(self, t, z_prev, x_t, mess_mean, mess_prec):
    """Computes the smoothing distribution given message incoming to z_t.

    Computes p(z_t | z_{t-1}, x_{t:num_timesteps}) given the message incoming
    to the node for z_t.

    Args:
      t: A python int, the index for z_t. When t is 0, z_prev is ignored.
      z_prev: z_{t-1}, the previous z to condition on. A Tensor of shape
        [batch_size].
      x_t: The observation x at timestep t.
      mess_mean: The mean of the message incoming to z_t, in information form.
      mess_prec: The precision of the message incoming to z_t.
    Returns:
      p(z_t | z_{t-1}, x_{t:num_timesteps}) as a univariate normal distribution.
    """

    batch_size = tf.shape(x_t)[0]
    z_var = self.transition_variances.read(t)
    x_var = self.emission_variances.read(t)
    w_x = self.emission_weights.read(t)

    def transition_term():
      return (tf.square(self.transition_weights.read(t))/
              self.transition_variances.read(t+1))

    prec = 1./z_var + tf.square(w_x)/x_var + mess_prec
    prec += tf.cond(tf.less(t, self.num_timesteps-1),
                    transition_term, lambda: 0.)
    mean = x_t*(w_x/x_var) + mess_mean
    mean += tf.cond(tf.greater(t, 0),
                    lambda: z_prev*(self.transition_weights.read(t-1)/z_var),
                    lambda: 0.)
    mean = tf.reshape(mean / prec, [batch_size])
    scale = tf.reshape(tf.sqrt(1./prec), [batch_size])
    return tfd.Normal(loc=mean, scale=scale)

  def _compute_backwards_messages(self, xs):
    """Computes the backwards messages used in smoothing."""
    batch_size = tf.shape(xs)[1]
    num_xs = tf.shape(xs)[0]
    until_t = self.num_timesteps - num_xs
    xs = tf.TensorArray(dtype=xs.dtype,
                        size=num_xs,
                        dynamic_size=False,
                        clear_after_read=True).unstack(xs)
    messages_ta = tf.TensorArray(dtype=xs.dtype,
                                 size=num_xs,
                                 dynamic_size=False,
                                 clear_after_read=False)

    def compute_message(t, prev_mean, prev_prec, messages_ta):
      """Computes one step of the backwards messages."""
      z_var = self.transition_variances.read(t)
      w_z = self.transition_weights.read(t-1)
      x_var = self.emission_variances.read(t)
      w_x = self.emission_weights.read(t)
      cur_x = xs.read(t - until_t)

      # If it isn't the first message, add the terms from the transition.
      def transition_term():
        return (tf.square(self.transition_weights.read(t))/
                self.transition_variances.read(t+1))

      unary_prec = 1/z_var + tf.square(w_x)/x_var
      unary_prec += tf.cond(tf.less(t, self.num_timesteps-1),
                            transition_term, lambda: 0.)

      unary_mean = (w_x / x_var) * cur_x
      pairwise_prec = w_z / z_var

      next_prec = -tf.square(pairwise_prec)/(unary_prec + prev_prec)
      next_mean = (pairwise_prec * (unary_mean + prev_mean) /
                   (unary_prec + prev_prec))
      next_prec = tf.reshape(next_prec, [batch_size])
      next_mean = tf.reshape(next_mean, [batch_size])
      messages_ta = messages_ta.write(t - until_t,
                                      tf.stack([next_mean, next_prec]))
      return t-1, next_mean, next_prec, messages_ta

    def pred(t, *unused_args):
      return tf.greater_equal(t, until_t)

    init_prec = tf.zeros([batch_size], dtype=xs.dtype)
    init_mean = tf.zeros([batch_size], dtype=xs.dtype)
    t0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)

    outs = tf.while_loop(pred, compute_message,
                         (t0, init_mean, init_prec, messages_ta))
    messages = outs[-1]
    return messages

  def lookahead(self, t, z_prev):
    """Compute the 'lookahead' distribution, p(x_{t:T} | z_{t-1}).

    Args:
      t: A scalar Tensor int, the current timestep. Must be at least 1.
      z_prev: The latent state at time t-1. A Tensor of shape [batch_size].
    Returns:
      p(x_{t:T} | z_{t-1}) as a multivariate normal distribution.
    """
    z_prev = tf.convert_to_tensor(z_prev)
    sigma_zx = self.sigma_zx[t-1, t:]
    z_var = self.sigma_z[t-1, t-1]
    mean = tf.einsum("i,j->ij", z_prev, sigma_zx) / z_var
    variance = (self.sigma_x[t:, t:] -
                tf.einsum("i,j->ij", sigma_zx, sigma_zx) / z_var)
    return tfd.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=variance)

  def likelihood(self, xs):
    """Compute the true marginal likelihood of the data.

    Args:
      xs: The observations, a [num_timesteps, batch_size] float Tensor.
    Returns:
      likelihoods: A [batch_size] float Tensor representing the likelihood of
        each sequence of observations in the batch.
    """
    return self.obs_dist.log_prob(tf.transpose(xs))


class TrainableGaussianHMM(GaussianHMM, base.ELBOTrainableSequenceModel):
  """An interface between importance-sampling training methods and the GHMM."""

  def __init__(self,
               num_timesteps,
               proposal_type,
               transition_variances=1.,
               emission_variances=1.,
               transition_weights=1.,
               emission_weights=1.,
               random_seed=None,
               dtype=tf.float32):
    """Constructs a trainable Gaussian HMM.

    Args:
      num_timesteps: A python int, the number of timesteps in the model.
      proposal_type: The type of proposal to use in the importance sampling
        setup. Could be "filtering", "smoothing", "prior", "true-filtering",
        or "true-smoothing". If "true-filtering" or "true-smoothing" are
        selected, then the true filtering or smoothing distributions are used to
        propose new states. If "learned-filtering" is selected then a
        distribution with learnable parameters is used. Specifically at each
        timestep the proposal is Gaussian with mean that is a learnable linear
        function of the previous state and current observation. The log variance
        is a per-timestep learnable constant. "learned-smoothing" is similar,
        but the mean is a learnable linear function of the previous state and
        all future observations. Note that this proposal class includes the true
        posterior. If "prior" is selected then states are proposed from the
        model's prior.
      transition_variances: The variance of p(z_t | z_t-1). Can be a scalar,
        setting all variances to be the same, or a Tensor of shape
        [num_timesteps].
      emission_variances: The variance of p(x_t | z_t). Can be a scalar,
        setting all variances to be the same, or a Tensor of shape
        [num_timesteps].
      transition_weights: The weight that defines the linear function that
        produces the mean of z_t given z_{t-1}. Can be a scalar, setting
        all weights to be the same, or a Tensor of shape [num_timesteps-1].
      emission_weights: The weight that defines the linear function that
        produces the mean of x_t given z_t. Can be a scalar, setting
        all weights to be the same, or a Tensor of shape [num_timesteps].
      random_seed: A seed for the proposal sampling, mainly useful for testing.
      dtype: The datatype of the state.
    """
    super(TrainableGaussianHMM, self).__init__(
        num_timesteps, transition_variances, emission_variances,
        transition_weights, emission_weights, dtype=dtype)
    self.random_seed = random_seed
    assert proposal_type in ["filtering", "smoothing", "prior",
                             "true-filtering", "true-smoothing"]
    if proposal_type == "true-filtering":
      self.proposal = self._filtering_proposal
    elif proposal_type == "true-smoothing":
      self.proposal = self._smoothing_proposal
    elif proposal_type == "prior":
      self.proposal = self.transition
    elif proposal_type == "filtering":
      self._learned_proposal_fn = base.NonstationaryLinearDistribution(
          num_timesteps, inputs_per_timestep=[1] + [2] * (num_timesteps-1))
      self.proposal = self._learned_filtering_proposal
    elif proposal_type == "smoothing":
      inputs_per_timestep = [num_timesteps] + [num_timesteps - t
                                               for t in range(num_timesteps-1)]
      self._learned_proposal_fn = base.NonstationaryLinearDistribution(
          num_timesteps, inputs_per_timestep=inputs_per_timestep)
      self.proposal = self._learned_smoothing_proposal

  def set_observations(self, xs, seq_lengths):
    """Sets the observations and stores the backwards messages."""
    # Squeeze out data dimension since everything is 1-d.
    xs = tf.squeeze(xs)
    self.batch_size = tf.shape(xs)[1]
    super(TrainableGaussianHMM, self).set_observations(xs, seq_lengths)
    self.messages = self._compute_backwards_messages(xs[1:])

  def zero_state(self, batch_size, dtype):
    return tf.zeros([batch_size], dtype=dtype)

  def propose_and_weight(self, state, t):
    """Computes the next state and log weights for the GHMM."""
    state_shape = tf.shape(state)
    xt = self.observations[t]
    p_zt = self.transition(t, state)
    q_zt = self.proposal(t, state)
    zt = q_zt.sample(seed=self.random_seed)
    zt = tf.reshape(zt, state_shape)
    p_xt_given_zt = self.emission(t, zt)
    log_p_zt = p_zt.log_prob(zt)
    log_q_zt = q_zt.log_prob(zt)
    log_p_xt_given_zt = p_xt_given_zt.log_prob(xt)
    weight = log_p_zt + log_p_xt_given_zt - log_q_zt
    return weight, zt

  def _filtering_proposal(self, t, state):
    """Uses the stored observations to compute the filtering distribution."""
    cur_x = self.observations[t]
    return self.filtering(t, state, cur_x)

  def _smoothing_proposal(self, t, state):
    """Uses the stored messages to compute the smoothing distribution."""
    mess_mean, mess_prec = tf.cond(
        tf.less(t, self.num_timesteps-1),
        lambda: tf.unstack(self.messages.read(t)),
        lambda: [tf.zeros([self.batch_size]), tf.zeros([self.batch_size])])
    return self._smoothing_from_message(t, state, self.observations[t],
                                        mess_mean, mess_prec)

  def _learned_filtering_proposal(self, t, state):
    cur_x = self.observations[t]
    inputs = tf.cond(tf.greater(t, 0),
                     lambda: tf.stack([state, cur_x], axis=0),
                     lambda: cur_x[tf.newaxis, :])
    return self._learned_proposal_fn(t, inputs)

  def _learned_smoothing_proposal(self, t, state):
    xs = self.observations_ta.gather(tf.range(t, self.num_timesteps))
    inputs = tf.cond(tf.greater(t, 0),
                     lambda: tf.concat([state[tf.newaxis, :], xs], axis=0),
                     lambda: xs)
    return self._learned_proposal_fn(t, inputs)
