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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
import summary_utils as summ

Loss = namedtuple("Loss", "name loss vars")
Loss.__new__.__defaults__ = (tf.GraphKeys.TRAINABLE_VARIABLES,)


def iwae(model, observation, num_timesteps, num_samples=1,
         summarize=False):
  """Compute the IWAE evidence lower bound.

  Args:
    model: A callable that computes one timestep of the model.
    observation: A shape [batch_size*num_samples, state_size] Tensor
      containing z_n, the observation for each sequence in the batch.
    num_timesteps: The number of timesteps in each sequence, an integer.
    num_samples: The number of samples to use to compute the IWAE bound.
  Returns:
    log_p_hat: The IWAE estimator of the lower bound on the log marginal.
    loss: A tensor that you can perform gradient descent on to optimize the
      bound.
    maintain_ema_op: A no-op included for compatibility with FIVO.
    states: The sequence of states sampled.
  """
  # Initialization
  num_instances = tf.shape(observation)[0]
  batch_size = tf.cast(num_instances / num_samples, tf.int32)
  states = [model.zero_state(num_instances)]
  log_weights = []
  log_weight_acc = tf.zeros([num_samples, batch_size], dtype=observation.dtype)

  for t in xrange(num_timesteps):
    # run the model for one timestep
    (zt, log_q_zt, log_p_zt, log_p_x_given_z, _) = model(
        states[-1], observation, t)
    # update accumulators
    states.append(zt)
    log_weight = log_p_zt + log_p_x_given_z - log_q_zt
    log_weight_acc += tf.reshape(log_weight, [num_samples, batch_size])
    if summarize:
      weight_dist = tf.contrib.distributions.Categorical(
          logits=tf.transpose(log_weight_acc, perm=[1, 0]),
          allow_nan_stats=False)
      weight_entropy = weight_dist.entropy()
      weight_entropy = tf.reduce_mean(weight_entropy)
      tf.summary.scalar("weight_entropy/%d" % t, weight_entropy)
    log_weights.append(log_weight_acc)
  # Compute the lower bound on the log evidence.
  log_p_hat = (tf.reduce_logsumexp(log_weight_acc, axis=0) -
               tf.log(tf.cast(num_samples, observation.dtype))) / num_timesteps
  loss = -tf.reduce_mean(log_p_hat)
  losses = [Loss("log_p_hat", loss)]

  # we clip off the initial state before returning.
  # there are no emas for iwae, so we return a noop for that
  return log_p_hat, losses, tf.no_op(), states[1:], log_weights


def multinomial_resampling(log_weights, states, n, b):
  """Resample states with multinomial resampling.

  Args:
    log_weights: A (n x b) Tensor representing a batch of b logits for n-ary
      Categorical distribution.
    states: A list of (b*n x d) Tensors that will be resample in from the groups
     of every n-th row.

  Returns:
    resampled_states: A list of (b*n x d) Tensors resampled via stratified sampling.
    log_probs: A (n x b) Tensor of the log probabilities of the ancestry decisions.
    resampling_parameters: The Tensor of parameters of the resampling distribution.
    ancestors: An (n x b) Tensor of integral indices representing the ancestry decisions.
    resampling_dist: The distribution object for resampling.
  """
  log_weights = tf.convert_to_tensor(log_weights)
  states = [tf.convert_to_tensor(state) for state in states]

  resampling_parameters = tf.transpose(log_weights, perm=[1,0])
  resampling_dist = tf.contrib.distributions.Categorical(logits=resampling_parameters)
  ancestors = tf.stop_gradient(
      resampling_dist.sample(sample_shape=n))
  log_probs = resampling_dist.log_prob(ancestors)

  offset = tf.expand_dims(tf.range(b), 0)
  ancestor_inds = tf.reshape(ancestors * b + offset, [-1])

  resampled_states = []
  for state in states:
    resampled_states.append(tf.gather(state, ancestor_inds))
  return resampled_states, log_probs, resampling_parameters, ancestors, resampling_dist

def stratified_resampling(log_weights, states, n, b):
  """Resample states with straitified resampling.

  Args:
    log_weights: A (n x b) Tensor representing a batch of b logits for n-ary
      Categorical distribution.
    states: A list of (b*n x d) Tensors that will be resample in from the groups
     of every n-th row.

  Returns:
    resampled_states: A list of (b*n x d) Tensors resampled via stratified sampling.
    log_probs: A (n x b) Tensor of the log probabilities of the ancestry decisions.
    resampling_parameters: The Tensor of parameters of the resampling distribution.
    ancestors: An (n x b) Tensor of integral indices representing the ancestry decisions.
    resampling_dist: The distribution object for resampling.
  """
  log_weights = tf.convert_to_tensor(log_weights)
  states = [tf.convert_to_tensor(state) for state in states]

  log_weights = tf.transpose(log_weights, perm=[1,0])

  probs = tf.nn.softmax(
            tf.tile(tf.expand_dims(log_weights, axis=1),
                    [1, n, 1])
  )

  cdfs = tf.concat([tf.zeros((b,n,1), dtype=probs.dtype), tf.cumsum(probs, axis=2)], 2)

  bins = tf.range(n, dtype=probs.dtype) / n
  bins = tf.tile(tf.reshape(bins, [1,-1,1]), [b,1,n+1])

  strat_cdfs = tf.minimum(tf.maximum((cdfs - bins) * n, 0.0), 1.0)
  resampling_parameters = strat_cdfs[:,:,1:] - strat_cdfs[:,:,:-1]

  resampling_dist = tf.contrib.distributions.Categorical(
      probs = resampling_parameters,
      allow_nan_stats=False)

  ancestors = tf.stop_gradient(
      resampling_dist.sample())
  log_probs = resampling_dist.log_prob(ancestors)

  ancestors = tf.transpose(ancestors, perm=[1,0])
  log_probs = tf.transpose(log_probs, perm=[1,0])

  offset = tf.expand_dims(tf.range(b), 0)
  ancestor_inds = tf.reshape(ancestors * b + offset, [-1])

  resampled_states = []
  for state in states:
    resampled_states.append(tf.gather(state, ancestor_inds))

  return resampled_states, log_probs, resampling_parameters, ancestors, resampling_dist

def systematic_resampling(log_weights, states, n, b):
  """Resample states with systematic resampling.

  Args:
    log_weights: A (n x b) Tensor representing a batch of b logits for n-ary
      Categorical distribution.
    states: A list of (b*n x d) Tensors that will be resample in from the groups
     of every n-th row.

  Returns:
    resampled_states: A list of (b*n x d) Tensors resampled via stratified sampling.
    log_probs: A (n x b) Tensor of the log probabilities of the ancestry decisions.
    resampling_parameters: The Tensor of parameters of the resampling distribution.
    ancestors: An (n x b) Tensor of integral indices representing the ancestry decisions.
    resampling_dist: The distribution object for resampling.
  """

  log_weights = tf.convert_to_tensor(log_weights)
  states = [tf.convert_to_tensor(state) for state in states]

  log_weights = tf.transpose(log_weights, perm=[1,0])

  probs = tf.nn.softmax(
            tf.tile(tf.expand_dims(log_weights, axis=1),
                    [1, n, 1])
  )

  cdfs = tf.concat([tf.zeros((b,n,1), dtype=probs.dtype), tf.cumsum(probs, axis=2)], 2)

  bins = tf.range(n, dtype=probs.dtype) / n
  bins = tf.tile(tf.reshape(bins, [1,-1,1]), [b,1,n+1])

  strat_cdfs = tf.minimum(tf.maximum((cdfs - bins) * n, 0.0), 1.0)
  resampling_parameters = strat_cdfs[:,:,1:] - strat_cdfs[:,:,:-1]

  resampling_dist = tf.contrib.distributions.Categorical(
      probs=resampling_parameters,
      allow_nan_stats=True)

  U = tf.random_uniform((b, 1, 1), dtype=probs.dtype)

  ancestors = tf.stop_gradient(tf.reduce_sum(tf.to_float(U > strat_cdfs[:,:,1:]), axis=-1))
  log_probs = resampling_dist.log_prob(ancestors)

  ancestors = tf.transpose(ancestors, perm=[1,0])
  log_probs = tf.transpose(log_probs, perm=[1,0])

  offset = tf.expand_dims(tf.range(b, dtype=probs.dtype), 0)
  ancestor_inds = tf.reshape(ancestors * b + offset, [-1])

  resampled_states = []
  for state in states:
    resampled_states.append(tf.gather(state, ancestor_inds))

  return resampled_states, log_probs, resampling_parameters, ancestors, resampling_dist


def log_blend(inputs, weights):
  """Blends state in the log space.

  Args:
    inputs: A set of scalar states, one for each particle in each particle filter.
      Should be [num_samples, batch_size].
    weights: A set of weights used to blend the state. Each set of weights
      should be of dimension [num_samples] (one weight for each previous particle).
      There should be one set of weights for each new particle in each particle filter.
      Thus the shape should be [num_samples, batch_size, num_samples] where
      the first axis indexes new particle and the last axis indexes old particles.
  Returns:
    blended: The blended states, a tensor of shape [num_samples, batch_size].
  """
  raw_max = tf.reduce_max(inputs, axis=0, keepdims=True)
  my_max = tf.stop_gradient(
      tf.where(tf.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
  )
  # Don't ask.
  blended = tf.log(tf.einsum("ijk,kj->ij", weights, tf.exp(inputs - raw_max))) + my_max
  return blended


def relaxed_resampling(log_weights, states, num_samples, batch_size,
                       log_r_x=None, blend_type="log", temperature=0.5,
                       straight_through=False):
  """Resample states with relaxed resampling.

  Args:
    log_weights: A (n x b) Tensor representing a batch of b logits for n-ary
      Categorical distribution.
    states: A list of (b*n x d) Tensors that will be resample in from the groups
     of every n-th row.

  Returns:
    resampled_states: A list of (b*n x d) Tensors resampled via stratified sampling.
    log_probs: A (n x b) Tensor of the log probabilities of the ancestry decisions.
    resampling_parameters: The Tensor of parameters of the resampling distribution.
    ancestors: An (n x b x n) Tensor of relaxed one hot representations of the ancestry decisions.
    resampling_dist: The distribution object for resampling.
  """
  assert blend_type in ["log", "linear"], "Blend type must be 'log' or 'linear'."
  log_weights = tf.convert_to_tensor(log_weights)
  states = [tf.convert_to_tensor(state) for state in states]
  state_dim = states[0].get_shape().as_list()[-1]
  # weights are num_samples by batch_size, so we transpose to get a
  # set of batch_size distributions over [0,num_samples).
  resampling_parameters = tf.transpose(log_weights, perm=[1, 0])
  resampling_dist = tf.contrib.distributions.RelaxedOneHotCategorical(
      temperature,
      logits=resampling_parameters)

  # sample num_samples samples from the distribution, resulting in a
  # [num_samples, batch_size, num_samples] Tensor that represents a set of
  # [num_samples, batch_size] blending weights. The dimensions represent
  # [sample index, batch index, blending weight index]
  ancestors = resampling_dist.sample(sample_shape=num_samples)
  if straight_through:
    # Forward pass discrete choices, backwards pass soft choices
    hard_ancestor_indices = tf.argmax(ancestors, axis=-1)
    hard_ancestors = tf.one_hot(hard_ancestor_indices, num_samples,
                                dtype=ancestors.dtype)
    ancestors = tf.stop_gradient(hard_ancestors - ancestors) + ancestors
  log_probs = resampling_dist.log_prob(ancestors)
  if log_r_x is not None and blend_type == "log":
    log_r_x = tf.reshape(log_r_x, [num_samples, batch_size])
    log_r_x = log_blend(log_r_x, ancestors)
    log_r_x = tf.reshape(log_r_x, [num_samples*batch_size])
  elif log_r_x is not None and blend_type == "linear":
    # If blend type is linear just add log_r to the states that will be blended
    # linearly.
    states.append(log_r_x)

  # transpose the 'indices' to be [batch_index, blending weight index, sample index]
  ancestor_inds = tf.transpose(ancestors, perm=[1, 2, 0])
  resampled_states = []
  for state in states:
    # state is currently [num_samples * batch_size, state_dim] so we reshape
    # to [num_samples, batch_size, state_dim] and then transpose to
    # [batch_size, state_size, num_samples]
    state = tf.transpose(tf.reshape(state, [num_samples, batch_size, -1]), perm=[1, 2, 0])
    # state is now (batch_size, state_size, num_samples)
    # and ancestor is (batch index, blending weight index, sample index)
    # multiplying these gives a matrix of size [batch_size, state_size, num_samples]
    next_state = tf.matmul(state, ancestor_inds)
    # transpose the state to be [num_samples, batch_size, state_size]
    # and then reshape it to match the state format.
    next_state = tf.reshape(tf.transpose(next_state, perm=[2,0,1]), [num_samples*batch_size, state_dim])
    resampled_states.append(next_state)

  new_dist = tf.contrib.distributions.Categorical(
      logits=resampling_parameters)

  if log_r_x is not None and blend_type == "linear":
    # If blend type is linear pop off log_r that we added to the states.
    log_r_x = tf.squeeze(resampled_states[-1])
    resampled_states = resampled_states[:-1]
  return resampled_states, log_probs, log_r_x, resampling_parameters, ancestors, new_dist


def fivo(model,
         observation,
         num_timesteps,
         resampling_schedule,
         num_samples=1,
         use_resampling_grads=True,
         resampling_type="multinomial",
         resampling_temperature=0.5,
         aux=True,
         summarize=False):
  """Compute the FIVO evidence lower bound.

  Args:
    model: A callable that computes one timestep of the model.
    observation: A shape [batch_size*num_samples, state_size] Tensor
      containing z_n, the observation for each sequence in the batch.
    num_timesteps: The number of timesteps in each sequence, an integer.
    resampling_schedule: A list of booleans of length num_timesteps, contains
      True if a resampling should occur on a specific timestep.
    num_samples: The number of samples to use to compute the IWAE bound.
    use_resampling_grads: Whether or not to include the resampling gradients
      in loss.
    resampling type: The type of resampling, one of "multinomial", "stratified",
      "relaxed-logblend", "relaxed-linearblend", "relaxed-stateblend", or
      "systematic".
    resampling_temperature: A positive temperature only used for relaxed
      resampling.
    aux: If true, compute the FIVO-AUX bound.
  Returns:
    log_p_hat: The IWAE estimator of the lower bound on the log marginal.
    loss: A tensor that you can perform gradient descent on to optimize the
      bound.
    maintain_ema_op: An op to update the baseline ema used for the resampling
      gradients.
    states: The sequence of states sampled.
  """
  # Initialization
  num_instances = tf.cast(tf.shape(observation)[0], tf.int32)
  batch_size = tf.cast(num_instances / num_samples, tf.int32)
  states = [model.zero_state(num_instances)]
  prev_state = states[0]
  log_weight_acc = tf.zeros(shape=[num_samples, batch_size], dtype=observation.dtype)
  prev_log_r_zt = tf.zeros([num_instances], dtype=observation.dtype)
  log_weights = []
  log_weights_all = []
  log_p_hats = []
  resampling_log_probs = []
  for t in xrange(num_timesteps):
    # run the model for one timestep
    (zt, log_q_zt, log_p_zt, log_p_x_given_z, log_r_zt) = model(
        prev_state, observation, t)
    # update accumulators
    states.append(zt)
    log_weight = log_p_zt + log_p_x_given_z - log_q_zt
    if aux:
      if t == num_timesteps - 1:
        log_weight -= prev_log_r_zt
      else:
        log_weight += log_r_zt - prev_log_r_zt
      prev_log_r_zt = log_r_zt
    log_weight_acc += tf.reshape(log_weight, [num_samples, batch_size])
    log_weights_all.append(log_weight_acc)
    if resampling_schedule[t]:

      # These objects will be resampled
      to_resample = [states[-1]]
      if aux and "relaxed" not in resampling_type:
        to_resample.append(prev_log_r_zt)

      # do the resampling
      if resampling_type == "multinomial":
        (resampled,
         resampling_log_prob,
         _, _, _) = multinomial_resampling(log_weight_acc,
                                           to_resample,
                                           num_samples,
                                           batch_size)
      elif resampling_type == "stratified":
        (resampled,
         resampling_log_prob,
         _, _, _) = stratified_resampling(log_weight_acc,
                                          to_resample,
                                          num_samples,
                                          batch_size)
      elif resampling_type == "systematic":
        (resampled,
         resampling_log_prob,
         _, _, _) = systematic_resampling(log_weight_acc,
                                          to_resample,
                                          num_samples,
                                          batch_size)
      elif "relaxed" in resampling_type:
        if aux:
          if resampling_type == "relaxed-logblend":
            (resampled,
             resampling_log_prob,
             prev_log_r_zt,
             _, _, _) = relaxed_resampling(log_weight_acc,
                                           to_resample,
                                           num_samples,
                                           batch_size,
                                           temperature=resampling_temperature,
                                           log_r_x=prev_log_r_zt,
                                           blend_type="log")
          elif resampling_type == "relaxed-linearblend":
            (resampled,
             resampling_log_prob,
             prev_log_r_zt,
             _, _, _) = relaxed_resampling(log_weight_acc,
                                           to_resample,
                                           num_samples,
                                           batch_size,
                                           temperature=resampling_temperature,
                                           log_r_x=prev_log_r_zt,
                                           blend_type="linear")
          elif resampling_type == "relaxed-stateblend":
            (resampled,
             resampling_log_prob,
             _, _, _, _) = relaxed_resampling(log_weight_acc,
                                              to_resample,
                                              num_samples,
                                              batch_size,
                                              temperature=resampling_temperature)
            # Calculate prev_log_r_zt from the post-resampling state
            prev_r_zt = model.r.r_xn(resampled[0], t)
            prev_log_r_zt = tf.reduce_sum(
                prev_r_zt.log_prob(observation), axis=[1])
          elif resampling_type == "relaxed-stateblend-st":
            (resampled,
             resampling_log_prob,
             _, _, _, _) = relaxed_resampling(log_weight_acc,
                                              to_resample,
                                              num_samples,
                                              batch_size,
                                              temperature=resampling_temperature,
                                              straight_through=True)
            # Calculate prev_log_r_zt from the post-resampling state
            prev_r_zt = model.r.r_xn(resampled[0], t)
            prev_log_r_zt = tf.reduce_sum(
                prev_r_zt.log_prob(observation), axis=[1])
        else:
          (resampled,
           resampling_log_prob,
           _, _, _, _) = relaxed_resampling(log_weight_acc,
                                            to_resample,
                                            num_samples,
                                            batch_size,
                                            temperature=resampling_temperature)
      #if summarize:
      #  resampling_entropy = resampling_dist.entropy()
      #  resampling_entropy = tf.reduce_mean(resampling_entropy)
      #  tf.summary.scalar("weight_entropy/%d" % t, resampling_entropy)

      resampling_log_probs.append(tf.reduce_sum(resampling_log_prob, axis=0))
      prev_state = resampled[0]
      if aux and "relaxed" not in resampling_type:
        # Squeeze out the extra dim potentially added by resampling.
        # prev_log_r_zt should always be [num_instances]
        prev_log_r_zt = tf.squeeze(resampled[1])
      # Update the log p hat estimate, taking a log sum exp over the sample
      # dimension. The appended tensor is [batch_size].
      log_p_hats.append(
          tf.reduce_logsumexp(log_weight_acc, axis=0) - tf.log(
              tf.cast(num_samples, dtype=observation.dtype)))
      # reset the weights
      log_weights.append(log_weight_acc)
      log_weight_acc = tf.zeros_like(log_weight_acc)
    else:
      prev_state = states[-1]
  # Compute the final weight update. If we just resampled this will be zero.
  final_update = (tf.reduce_logsumexp(log_weight_acc, axis=0) -
                  tf.log(tf.cast(num_samples, dtype=observation.dtype)))
  # If we ever resampled, then sum up the previous log p hat terms
  if len(log_p_hats) > 0:
    log_p_hat = tf.reduce_sum(log_p_hats, axis=0) + final_update
  else:  # otherwise, log_p_hat only comes from the final update
    log_p_hat = final_update

  if use_resampling_grads and any(resampling_schedule):
    # compute the rewards
    # cumsum([a, b, c]) => [a, a+b, a+b+c]
    # learning signal at timestep t is
    #   [sum from i=t+1 to T of log_p_hat_i for t=1:T]
    # so we will compute (sum from i=1 to T of log_p_hat_i)
    # and at timestep t will subtract off (sum from i=1 to t of log_p_hat_i)
    # rewards is a [num_resampling_events, batch_size] Tensor
    rewards = tf.stop_gradient(
        tf.expand_dims(log_p_hat, 0) - tf.cumsum(log_p_hats, axis=0))
    batch_avg_rewards = tf.reduce_mean(rewards, axis=1)
    # compute ema baseline.
    # centered_rewards is [num_resampling_events, batch_size]
    baseline_ema = tf.train.ExponentialMovingAverage(decay=0.94)
    maintain_baseline_op = baseline_ema.apply([batch_avg_rewards])
    baseline = tf.expand_dims(baseline_ema.average(batch_avg_rewards), 1)
    centered_rewards = rewards - baseline
    if summarize:
      summ.summarize_learning_signal(rewards, "rewards")
      summ.summarize_learning_signal(centered_rewards, "centered_rewards")
    # compute the loss tensor.
    resampling_grads = tf.reduce_sum(
        tf.stop_gradient(centered_rewards) * resampling_log_probs, axis=0)
    losses = [Loss("log_p_hat", -tf.reduce_mean(log_p_hat)/num_timesteps),
              Loss("resampling_grads", -tf.reduce_mean(resampling_grads)/num_timesteps)]
  else:
    losses = [Loss("log_p_hat", -tf.reduce_mean(log_p_hat)/num_timesteps)]
    maintain_baseline_op = tf.no_op()

  log_p_hat /= num_timesteps
  # we clip off the initial state before returning.
  return log_p_hat, losses, maintain_baseline_op, states[1:], log_weights_all


def fivo_aux_td(
    model,
    observation,
    num_timesteps,
    resampling_schedule,
    num_samples=1,
    summarize=False):
  """Compute the FIVO_AUX evidence lower bound."""
  # Initialization
  num_instances = tf.cast(tf.shape(observation)[0], tf.int32)
  batch_size = tf.cast(num_instances / num_samples, tf.int32)
  states = [model.zero_state(num_instances)]
  prev_state = states[0]
  log_weight_acc = tf.zeros(shape=[num_samples, batch_size], dtype=observation.dtype)
  prev_log_r = tf.zeros([num_instances], dtype=observation.dtype)
  # must be pre-resampling
  log_rs = []
  # must be post-resampling
  r_tilde_params = [model.r_tilde.r_zt(states[0], observation, 0)]
  log_r_tildes = []
  log_p_xs = []
  # contains the weight at each timestep before resampling only on resampling timesteps
  log_weights = []
  # contains weight at each timestep before resampling
  log_weights_all = []
  log_p_hats = []
  for t in xrange(num_timesteps):
    # run the model for one timestep
    # zt is state, [num_instances, state_dim]
    # log_q_zt, log_p_x_given_z is [num_instances]
    # r_tilde_mu, r_tilde_sigma is [num_instances, state_dim]
    # p_ztplus1 is a normal distribution on [num_instances, state_dim]
    (zt, log_q_zt, log_p_zt, log_p_x_given_z,
     r_tilde_mu, r_tilde_sigma_sq, p_ztplus1) = model(prev_state, observation, t)

    # Compute the log weight without log r.
    log_weight = log_p_zt + log_p_x_given_z - log_q_zt

    # Compute log r.
    if t == num_timesteps - 1:
      log_r = tf.zeros_like(prev_log_r)
    else:
      p_mu = p_ztplus1.mean()
      p_sigma_sq = p_ztplus1.variance()
      log_r = (tf.log(r_tilde_sigma_sq) -
               tf.log(r_tilde_sigma_sq + p_sigma_sq) -
               tf.square(r_tilde_mu - p_mu)/(r_tilde_sigma_sq + p_sigma_sq))
      log_r = 0.5*tf.reduce_sum(log_r, axis=-1)

    #log_weight += tf.stop_gradient(log_r - prev_log_r)
    log_weight += log_r - prev_log_r
    log_weight_acc += tf.reshape(log_weight, [num_samples, batch_size])

    # Update accumulators
    states.append(zt)
    log_weights_all.append(log_weight_acc)
    log_p_xs.append(log_p_x_given_z)
    log_rs.append(log_r)

    # Compute log_r_tilde as [num_instances] Tensor.
    prev_r_tilde_mu, prev_r_tilde_sigma_sq = r_tilde_params[-1]
    prev_log_r_tilde = -0.5*tf.reduce_sum(
        tf.square(zt - prev_r_tilde_mu)/prev_r_tilde_sigma_sq, axis=-1)
        #tf.square(tf.stop_gradient(zt) - r_tilde_mu)/r_tilde_sigma_sq, axis=-1)
        #tf.square(zt - r_tilde_mu)/r_tilde_sigma_sq, axis=-1)
    log_r_tildes.append(prev_log_r_tilde)

    # optionally resample
    if resampling_schedule[t]:
      # These objects will be resampled
      if t < num_timesteps - 1:
        to_resample = [zt, log_r, r_tilde_mu, r_tilde_sigma_sq]
      else:
        to_resample = [zt, log_r]
      (resampled,
       _, _, _, _) = multinomial_resampling(log_weight_acc,
                                            to_resample,
                                            num_samples,
                                            batch_size)
      prev_state = resampled[0]
      # Squeeze out the extra dim potentially added by resampling.
      # prev_log_r_zt and log_r_tilde should always be [num_instances]
      prev_log_r = tf.squeeze(resampled[1])
      if t < num_timesteps -1:
        r_tilde_params.append((resampled[2], resampled[3]))
      # Update the log p hat estimate, taking a log sum exp over the sample
      # dimension. The appended tensor is [batch_size].
      log_p_hats.append(
          tf.reduce_logsumexp(log_weight_acc, axis=0) - tf.log(
              tf.cast(num_samples, dtype=observation.dtype)))
      # reset the weights
      log_weights.append(log_weight_acc)
      log_weight_acc = tf.zeros_like(log_weight_acc)
    else:
      prev_state = zt
      prev_log_r = log_r
      if t < num_timesteps - 1:
        r_tilde_params.append((r_tilde_mu, r_tilde_sigma_sq))

  # Compute the final weight update. If we just resampled this will be zero.
  final_update = (tf.reduce_logsumexp(log_weight_acc, axis=0) -
                  tf.log(tf.cast(num_samples, dtype=observation.dtype)))
  # If we ever resampled, then sum up the previous log p hat terms
  if len(log_p_hats) > 0:
    log_p_hat = tf.reduce_sum(log_p_hats, axis=0) + final_update
  else:  # otherwise, log_p_hat only comes from the final update
    log_p_hat = final_update

  # Compute the bellman loss.
  # Will remove the first timestep as it is not used.
  # log p(x_t|z_t) is in row t-1.
  log_p_x = tf.reshape(tf.stack(log_p_xs),
                       [num_timesteps, num_samples, batch_size])
  # log r_t is contained in row t-1.
  # last column is zeros (because at timestep T (num_timesteps) r is 1.
  log_r = tf.reshape(tf.stack(log_rs),
                     [num_timesteps, num_samples, batch_size])
  # [num_timesteps, num_instances]. log r_tilde_t is in row t-1.
  log_r_tilde = tf.reshape(tf.stack(log_r_tildes),
                           [num_timesteps, num_samples, batch_size])
  log_lambda = tf.reduce_mean(log_r_tilde - log_p_x - log_r, axis=1,
                              keepdims=True)
  bellman_sos = tf.reduce_mean(tf.square(
      log_r_tilde - tf.stop_gradient(log_lambda + log_p_x + log_r)), axis=[0, 1])
  bellman_loss = tf.reduce_mean(bellman_sos)/num_timesteps
  tf.summary.scalar("bellman_loss", bellman_loss)

  if len(tf.get_collection("LOG_P_HAT_VARS")) == 0:
    log_p_hat_collection = list(set(tf.trainable_variables()) -
                                set(tf.get_collection("R_TILDE_VARS")))
    for v in log_p_hat_collection:
      tf.add_to_collection("LOG_P_HAT_VARS", v)

  log_p_hat /= num_timesteps
  losses = [Loss("log_p_hat", -tf.reduce_mean(log_p_hat), "LOG_P_HAT_VARS"),
            Loss("bellman_loss", bellman_loss, "R_TILDE_VARS")]

  return log_p_hat, losses, tf.no_op(), states[1:], log_weights_all
