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

"""Implementation of objectives for training stochastic latent variable models.

Contains implementations of the Importance Weighted Autoencoder objective (IWAE)
and the Filtering Variational objective (FIVO).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import nested_utils as nested


def iwae(cell,
         inputs,
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
    cell: A callable that implements one timestep of the model. See
      models/vrnn.py for an example.
    inputs: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. At each
      timestep 'cell' will be called with a slice of the Tensors in inputs.
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
    kl: A Tensor of shape [batch_size] containing the kl divergence
      from q(z|x) to p(z), averaged over samples.
    log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
      containing the log weights at each timestep. Will not be valid for
      timesteps past the end of a sequence.
    log_ess: A Tensor of shape [max_seq_len, batch_size] containing the log
      effective sample size at each timestep. Will not be valid for timesteps
      past the end of a sequence.
  """
  batch_size = tf.shape(seq_lengths)[0]
  max_seq_len = tf.reduce_max(seq_lengths)
  seq_mask = tf.transpose(
      tf.sequence_mask(seq_lengths, maxlen=max_seq_len, dtype=tf.float32),
      perm=[1, 0])
  if num_samples > 1:
    inputs, seq_mask = nested.tile_tensors([inputs, seq_mask], [1, num_samples])
  inputs_ta, mask_ta = nested.tas_for_tensors([inputs, seq_mask], max_seq_len)

  t0 = tf.constant(0, tf.int32)
  init_states = cell.zero_state(batch_size * num_samples, tf.float32)
  ta_names = ['log_weights', 'log_ess']
  tas = [tf.TensorArray(tf.float32, max_seq_len, name='%s_ta' % n)
         for n in ta_names]
  log_weights_acc = tf.zeros([num_samples, batch_size], dtype=tf.float32)
  kl_acc = tf.zeros([num_samples * batch_size], dtype=tf.float32)
  accs = (log_weights_acc, kl_acc)

  def while_predicate(t, *unused_args):
    return t < max_seq_len

  def while_step(t, rnn_state, tas, accs):
    """Implements one timestep of IWAE computation."""
    log_weights_acc, kl_acc = accs
    cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
    # Run the cell for one step.
    log_q_z, log_p_z, log_p_x_given_z, kl, new_state = cell(
        cur_inputs,
        rnn_state,
        cur_mask,
    )
    # Compute the incremental weight and use it to update the current
    # accumulated weight.
    kl_acc += kl * cur_mask
    log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
    log_alpha = tf.reshape(log_alpha, [num_samples, batch_size])
    log_weights_acc += log_alpha
    # Calculate the effective sample size.
    ess_num = 2 * tf.reduce_logsumexp(log_weights_acc, axis=0)
    ess_denom = tf.reduce_logsumexp(2 * log_weights_acc, axis=0)
    log_ess = ess_num - ess_denom
    # Update the  Tensorarrays and accumulators.
    ta_updates = [log_weights_acc, log_ess]
    new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]
    new_accs = (log_weights_acc, kl_acc)
    return t + 1, new_state, new_tas, new_accs

  _, _, tas, accs = tf.while_loop(
      while_predicate,
      while_step,
      loop_vars=(t0, init_states, tas, accs),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  log_weights, log_ess = [x.stack() for x in tas]
  final_log_weights, kl = accs
  log_p_hat = (tf.reduce_logsumexp(final_log_weights, axis=0) -
               tf.log(tf.to_float(num_samples)))
  kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)
  log_weights = tf.transpose(log_weights, perm=[0, 2, 1])
  return log_p_hat, kl, log_weights, log_ess


def ess_criterion(num_samples, log_ess, unused_t):
  """A criterion that resamples based on effective sample size."""
  return log_ess <= tf.log(num_samples / 2.0)


def never_resample_criterion(unused_num_samples, log_ess, unused_t):
  """A criterion that never resamples."""
  return tf.cast(tf.zeros_like(log_ess), tf.bool)


def always_resample_criterion(unused_num_samples, log_ess, unused_t):
  """A criterion resamples at every timestep."""
  return tf.cast(tf.ones_like(log_ess), tf.bool)


def fivo(cell,
         inputs,
         seq_lengths,
         num_samples=1,
         resampling_criterion=ess_criterion,
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
    cell: A callable that implements one timestep of the model. See
      models/vrnn.py for an example.
    inputs: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. At each
      timestep 'cell' will be called with a slice of the Tensors in inputs.
    seq_lengths: A [batch_size] Tensor of ints encoding the length of each
      sequence in the batch (sequences can be padded to a common length).
    num_samples: The number of particles to use in each particle filter.
    resampling_criterion: The resampling criterion to use for this particle
      filter. Must accept the number of samples, the effective sample size,
      and the current timestep and return a boolean Tensor of shape [batch_size]
      indicating whether each particle filter should resample. See
      ess_criterion and related functions defined in this file for examples.
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
    kl: A Tensor of shape [batch_size] containing the sum over time of the kl
      divergence from q_t(z_t|x) to p_t(z_t), averaged over particles. Note that
      this includes kl terms from trajectories that are culled during resampling
      steps.
    log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
      containing the log weights at each timestep of the particle filter. Note
      that on timesteps when a resampling operation is performed the log weights
      are reset to 0. Will not be valid for timesteps past the end of a
      sequence.
    log_ess: A Tensor of shape [max_seq_len, batch_size] containing the log
      effective sample size of each particle filter at each timestep. Will not
      be valid for timesteps past the end of a sequence.
    resampled: A Tensor of shape [max_seq_len, batch_size] indicating when the
      particle filters resampled. Will be 1.0 on timesteps when resampling
      occurred and 0.0 on timesteps when it did not.
  """
  # batch_size represents the number of particle filters running in parallel.
  batch_size = tf.shape(seq_lengths)[0]
  max_seq_len = tf.reduce_max(seq_lengths)
  seq_mask = tf.transpose(
      tf.sequence_mask(seq_lengths, maxlen=max_seq_len, dtype=tf.float32),
      perm=[1, 0])

  # Each sequence in the batch will be the input data for a different
  # particle filter. The batch will be laid out as:
  #   particle 1 of particle filter 1
  #   particle 1 of particle filter 2
  #   ...
  #   particle 1 of particle filter batch_size
  #   particle 2 of particle filter 1
  #   ...
  #   particle num_samples of particle filter batch_size
  if num_samples > 1:
    inputs, seq_mask = nested.tile_tensors([inputs, seq_mask], [1, num_samples])
  inputs_ta, mask_ta = nested.tas_for_tensors([inputs, seq_mask], max_seq_len)

  t0 = tf.constant(0, tf.int32)
  init_states = cell.zero_state(batch_size * num_samples, tf.float32)
  ta_names = ['log_weights', 'log_ess', 'resampled']
  tas = [tf.TensorArray(tf.float32, max_seq_len, name='%s_ta' % n)
         for n in ta_names]
  log_weights_acc = tf.zeros([num_samples, batch_size], dtype=tf.float32)
  log_p_hat_acc = tf.zeros([batch_size], dtype=tf.float32)
  kl_acc = tf.zeros([num_samples * batch_size], dtype=tf.float32)
  accs = (log_weights_acc, log_p_hat_acc, kl_acc)

  def while_predicate(t, *unused_args):
    return t < max_seq_len

  def while_step(t, rnn_state, tas, accs):
    """Implements one timestep of FIVO computation."""
    log_weights_acc, log_p_hat_acc, kl_acc = accs
    cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
    # Run the cell for one step.
    log_q_z, log_p_z, log_p_x_given_z, kl, new_state = cell(
        cur_inputs,
        rnn_state,
        cur_mask,
    )
    # Compute the incremental weight and use it to update the current
    # accumulated weight.
    kl_acc += kl * cur_mask
    log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
    log_alpha = tf.reshape(log_alpha, [num_samples, batch_size])
    log_weights_acc += log_alpha
    # Calculate the effective sample size.
    ess_num = 2 * tf.reduce_logsumexp(log_weights_acc, axis=0)
    ess_denom = tf.reduce_logsumexp(2 * log_weights_acc, axis=0)
    log_ess = ess_num - ess_denom
    # Calculate the ancestor indices via resampling. Because we maintain the
    # log unnormalized weights, we pass the weights in as logits, allowing
    # the distribution object to apply a softmax and normalize them.
    resampling_dist = tf.contrib.distributions.Categorical(
        logits=tf.transpose(log_weights_acc, perm=[1, 0]))
    ancestor_inds = tf.stop_gradient(
        resampling_dist.sample(sample_shape=num_samples, seed=random_seed))
    # Because the batch is flattened and laid out as discussed
    # above, we must modify ancestor_inds to index the proper samples.
    # The particles in the ith filter are distributed every batch_size rows
    # in the batch, and offset i rows from the top. So, to correct the indices
    # we multiply by the batch_size and add the proper offset. Crucially,
    # when ancestor_inds is flattened the layout of the batch is maintained.
    offset = tf.expand_dims(tf.range(batch_size), 0)
    ancestor_inds = tf.reshape(ancestor_inds * batch_size + offset, [-1])
    noresample_inds = tf.range(num_samples * batch_size)
    # Decide whether or not we should resample; don't resample if we are past
    # the end of a sequence.
    should_resample = resampling_criterion(num_samples, log_ess, t)
    should_resample = tf.logical_and(should_resample,
                                     cur_mask[:batch_size] > 0.)
    float_should_resample = tf.to_float(should_resample)
    ancestor_inds = tf.where(
        tf.tile(should_resample, [num_samples]),
        ancestor_inds,
        noresample_inds)
    new_state = nested.gather_tensors(new_state, ancestor_inds)
    # Update the TensorArrays before we reset the weights so that we capture
    # the incremental weights and not zeros.
    ta_updates = [log_weights_acc, log_ess, float_should_resample]
    new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]
    # For the particle filters that resampled, update log_p_hat and
    # reset weights to zero.
    log_p_hat_update = tf.reduce_logsumexp(
        log_weights_acc, axis=0) - tf.log(tf.to_float(num_samples))
    log_p_hat_acc += log_p_hat_update * float_should_resample
    log_weights_acc *= (1. - tf.tile(float_should_resample[tf.newaxis, :],
                                     [num_samples, 1]))
    new_accs = (log_weights_acc, log_p_hat_acc, kl_acc)
    return t + 1, new_state, new_tas, new_accs

  _, _, tas, accs = tf.while_loop(
      while_predicate,
      while_step,
      loop_vars=(t0, init_states, tas, accs),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  log_weights, log_ess, resampled = [x.stack() for x in tas]
  final_log_weights, log_p_hat, kl = accs
  # Add in the final weight update to log_p_hat.
  log_p_hat += (tf.reduce_logsumexp(final_log_weights, axis=0) -
                tf.log(tf.to_float(num_samples)))
  kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)
  log_weights = tf.transpose(log_weights, perm=[0, 2, 1])
  return log_p_hat, kl, log_weights, log_ess, resampled
