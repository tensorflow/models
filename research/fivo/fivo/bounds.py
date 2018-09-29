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

"""Implementation of objectives for training stochastic latent variable models.

Contains implementations of the Importance Weighted Autoencoder objective (IWAE)
and the Filtering Variational objective (FIVO).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from fivo import nested_utils as nested
from fivo import smc


def iwae(model,
         observations,
         seq_lengths,
         num_samples=1,
         parallel_iterations=30,
         swap_memory=True):
  """Computes the IWAE lower bound on the log marginal probability.

  This method accepts a stochastic latent variable model and some observations
  and computes a stochastic lower bound on the log marginal probability of the
  observations. The IWAE estimator is defined by averaging multiple importance
  weights. For more details see "Importance Weighted Autoencoders" by Burda
  et al. https://arxiv.org/abs/1509.00519.

  When num_samples = 1, this bound becomes the evidence lower bound (ELBO).

  Args:
    model: A subclass of ELBOTrainableSequenceModel that implements one
      timestep of the model. See models/vrnn.py for an example.
    observations: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. The model
      will be provided with the observations before computing the bound.
    seq_lengths: A [batch_size] Tensor of ints encoding the length of each
      sequence in the batch (sequences can be padded to a common length).
    num_samples: The number of samples to use.
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.

  Returns:
    log_p_hat: A Tensor of shape [batch_size] containing IWAE's estimate of the
      log marginal probability of the observations.
    log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
      containing the log weights at each timestep. Will not be valid for
      timesteps past the end of a sequence.
  """
  log_p_hat, log_weights, _, final_state = fivo(
      model,
      observations,
      seq_lengths,
      num_samples=num_samples,
      resampling_criterion=smc.never_resample_criterion,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)
  return log_p_hat, log_weights, final_state


def fivo(model,
         observations,
         seq_lengths,
         num_samples=1,
         resampling_criterion=smc.ess_criterion,
         resampling_type='multinomial',
         relaxed_resampling_temperature=0.5,
         parallel_iterations=30,
         swap_memory=True,
         random_seed=None):
  """Computes the FIVO lower bound on the log marginal probability.

  This method accepts a stochastic latent variable model and some observations
  and computes a stochastic lower bound on the log marginal probability of the
  observations. The lower bound is defined by a particle filter's unbiased
  estimate of the marginal probability of the observations. For more details see
  "Filtering Variational Objectives" by Maddison et al.
  https://arxiv.org/abs/1705.09279.

  When the resampling criterion is "never resample", this bound becomes IWAE.

  Args:
    model: A subclass of ELBOTrainableSequenceModel that implements one
      timestep of the model. See models/vrnn.py for an example.
    observations: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. The model
      will be provided with the observations before computing the bound.
    seq_lengths: A [batch_size] Tensor of ints encoding the length of each
      sequence in the batch (sequences can be padded to a common length).
    num_samples: The number of particles to use in each particle filter.
    resampling_criterion: The resampling criterion to use for this particle
      filter. Must accept the number of samples, the current log weights,
      and the current timestep and return a boolean Tensor of shape [batch_size]
      indicating whether each particle filter should resample. See
      ess_criterion and related functions for examples. When
      resampling_criterion is never_resample_criterion, resampling_fn is ignored
      and never called.
    resampling_type: The type of resampling, one of "multinomial" or "relaxed".
    relaxed_resampling_temperature: A positive temperature only used for relaxed
      resampling.
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop. Note that values greater than 1 can introduce
      non-determinism even when random_seed is provided.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.
    random_seed: The random seed to pass to the resampling operations in
      the particle filter. Mainly useful for testing.

  Returns:
    log_p_hat: A Tensor of shape [batch_size] containing FIVO's estimate of the
      log marginal probability of the observations.
    log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
      containing the log weights at each timestep of the particle filter. Note
      that on timesteps when a resampling operation is performed the log weights
      are reset to 0. Will not be valid for timesteps past the end of a
      sequence.
    resampled: A Tensor of shape [max_seq_len, batch_size] indicating when the
      particle filters resampled. Will be 1.0 on timesteps when resampling
      occurred and 0.0 on timesteps when it did not.
  """
  # batch_size is the number of particle filters running in parallel.
  batch_size = tf.shape(seq_lengths)[0]

  # Each sequence in the batch will be the input data for a different
  # particle filter. The batch will be laid out as:
  #   particle 1 of particle filter 1
  #   particle 1 of particle filter 2
  #   ...
  #   particle 1 of particle filter batch_size
  #   particle 2 of particle filter 1
  #   ...
  #   particle num_samples of particle filter batch_size
  observations = nested.tile_tensors(observations, [1, num_samples])
  tiled_seq_lengths = tf.tile(seq_lengths, [num_samples])
  model.set_observations(observations, tiled_seq_lengths)

  if resampling_type == 'multinomial':
    resampling_fn = smc.multinomial_resampling
  elif resampling_type == 'relaxed':
    resampling_fn = functools.partial(
        smc.relaxed_resampling, temperature=relaxed_resampling_temperature)
  resampling_fn = functools.partial(resampling_fn, random_seed=random_seed)

  def transition_fn(prev_state, t):
    if prev_state is None:
      return model.zero_state(batch_size * num_samples, tf.float32)
    return model.propose_and_weight(prev_state, t)

  log_p_hat, log_weights, resampled, final_state, _ = smc.smc(
      transition_fn,
      seq_lengths,
      num_particles=num_samples,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  return log_p_hat, log_weights, resampled, final_state

def fivo_aux_td(
    model,
    observations,
    seq_lengths,
    num_samples=1,
    resampling_criterion=smc.ess_criterion,
    resampling_type='multinomial',
    relaxed_resampling_temperature=0.5,
    parallel_iterations=30,
    swap_memory=True,
    random_seed=None):
  """Experimental."""
  # batch_size is the number of particle filters running in parallel.
  batch_size = tf.shape(seq_lengths)[0]
  max_seq_len = tf.reduce_max(seq_lengths)

  # Each sequence in the batch will be the input data for a different
  # particle filter. The batch will be laid out as:
  #   particle 1 of particle filter 1
  #   particle 1 of particle filter 2
  #   ...
  #   particle 1 of particle filter batch_size
  #   particle 2 of particle filter 1
  #   ...
  #   particle num_samples of particle filter batch_size
  observations = nested.tile_tensors(observations, [1, num_samples])
  tiled_seq_lengths = tf.tile(seq_lengths, [num_samples])
  model.set_observations(observations, tiled_seq_lengths)

  if resampling_type == 'multinomial':
    resampling_fn = smc.multinomial_resampling
  elif resampling_type == 'relaxed':
    resampling_fn = functools.partial(
        smc.relaxed_resampling, temperature=relaxed_resampling_temperature)
  resampling_fn = functools.partial(resampling_fn, random_seed=random_seed)

  def transition_fn(prev_state, t):
    if prev_state is None:
      model_init_state = model.zero_state(batch_size * num_samples, tf.float32)
      return (tf.zeros([num_samples*batch_size], dtype=tf.float32),
              (tf.zeros([num_samples*batch_size, model.latent_size], dtype=tf.float32),
               tf.zeros([num_samples*batch_size, model.latent_size], dtype=tf.float32)),
              model_init_state)

    prev_log_r, prev_log_r_tilde, prev_model_state = prev_state
    (new_model_state, zt, log_q_zt, log_p_zt,
     log_p_x_given_z, log_r_tilde, p_ztplus1) = model(prev_model_state, t)
    r_tilde_mu, r_tilde_sigma_sq = log_r_tilde
    # Compute the weight without r.
    log_weight = log_p_zt + log_p_x_given_z - log_q_zt
    # Compute log_r and log_r_tilde.
    p_mu = tf.stop_gradient(p_ztplus1.mean())
    p_sigma_sq = tf.stop_gradient(p_ztplus1.variance())
    log_r = (tf.log(r_tilde_sigma_sq) -
             tf.log(r_tilde_sigma_sq + p_sigma_sq) -
             tf.square(r_tilde_mu - p_mu)/(r_tilde_sigma_sq + p_sigma_sq))
    # log_r is [num_samples*batch_size, latent_size]. We sum it along the last
    # dimension to compute log r.
    log_r = 0.5*tf.reduce_sum(log_r, axis=-1)
    # Compute prev log r tilde
    prev_r_tilde_mu, prev_r_tilde_sigma_sq = prev_log_r_tilde
    prev_log_r_tilde = -0.5*tf.reduce_sum(
        tf.square(tf.stop_gradient(zt) - prev_r_tilde_mu)/prev_r_tilde_sigma_sq, axis=-1)
    # If the sequence is on the last timestep, log_r and log_r_tilde are just zeros.
    last_timestep = t >= (tiled_seq_lengths - 1)
    log_r = tf.where(last_timestep,
                     tf.zeros_like(log_r),
                     log_r)
    prev_log_r_tilde = tf.where(last_timestep,
                                tf.zeros_like(prev_log_r_tilde),
                                prev_log_r_tilde)
    log_weight += tf.stop_gradient(log_r - prev_log_r)
    new_state = (log_r, log_r_tilde, new_model_state)
    loop_fn_args = (log_r, prev_log_r_tilde, log_p_x_given_z, log_r - prev_log_r)
    return log_weight, new_state, loop_fn_args

  def loop_fn(loop_state, loop_args, unused_model_state, log_weights, resampled, mask, t):
    if loop_state is None:
      return (tf.zeros([batch_size], dtype=tf.float32),
              tf.zeros([batch_size], dtype=tf.float32),
              tf.zeros([num_samples, batch_size], dtype=tf.float32))
    log_p_hat_acc, bellman_loss_acc, log_r_diff_acc = loop_state
    log_r, prev_log_r_tilde, log_p_x_given_z, log_r_diff = loop_args
    # Compute the log_p_hat update
    log_p_hat_update = tf.reduce_logsumexp(
        log_weights, axis=0) - tf.log(tf.to_float(num_samples))
    # If it is the last timestep, we always add the update.
    log_p_hat_acc += tf.cond(t >= max_seq_len-1,
                             lambda: log_p_hat_update,
                             lambda: log_p_hat_update * resampled)
    # Compute the Bellman update.
    log_r = tf.reshape(log_r, [num_samples, batch_size])
    prev_log_r_tilde = tf.reshape(prev_log_r_tilde, [num_samples, batch_size])
    log_p_x_given_z = tf.reshape(log_p_x_given_z, [num_samples, batch_size])
    mask = tf.reshape(mask, [num_samples, batch_size])
    # On the first timestep there is no bellman error because there is no
    # prev_log_r_tilde.
    mask = tf.cond(tf.equal(t, 0),
                   lambda: tf.zeros_like(mask),
                   lambda: mask)
    # On the first timestep also fix up prev_log_r_tilde, which will be -inf.
    prev_log_r_tilde = tf.where(
        tf.is_inf(prev_log_r_tilde),
        tf.zeros_like(prev_log_r_tilde),
        prev_log_r_tilde)
    # log_lambda is [num_samples, batch_size]
    log_lambda = tf.reduce_mean(prev_log_r_tilde - log_p_x_given_z - log_r,
                                axis=0, keepdims=True)
    bellman_error = mask * tf.square(
        prev_log_r_tilde -
        tf.stop_gradient(log_lambda + log_p_x_given_z + log_r)
    )
    bellman_loss_acc += tf.reduce_mean(bellman_error, axis=0)
    # Compute the log_r_diff update
    log_r_diff_acc += mask * tf.reshape(log_r_diff, [num_samples, batch_size])
    return (log_p_hat_acc, bellman_loss_acc, log_r_diff_acc)

  log_weights, resampled, accs = smc.smc(
      transition_fn,
      seq_lengths,
      num_particles=num_samples,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn,
      loop_fn=loop_fn,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  log_p_hat, bellman_loss, log_r_diff = accs
  loss_per_seq = [- log_p_hat, bellman_loss]
  tf.summary.scalar("bellman_loss",
                    tf.reduce_mean(bellman_loss / tf.to_float(seq_lengths)))
  tf.summary.scalar("log_r_diff",
                    tf.reduce_mean(tf.reduce_mean(log_r_diff, axis=0) / tf.to_float(seq_lengths)))
  return loss_per_seq, log_p_hat, log_weights, resampled
