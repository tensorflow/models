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

"""Implementation of sequential Monte Carlo algorithms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import fivo.nested_utils as nested


def ess_criterion(log_weights, unused_t):
  """A criterion that resamples based on effective sample size."""
  num_particles = tf.shape(log_weights)[0]
  # Calculate the effective sample size.
  ess_num = 2 * tf.reduce_logsumexp(log_weights, axis=0)
  ess_denom = tf.reduce_logsumexp(2 * log_weights, axis=0)
  log_ess = ess_num - ess_denom
  return log_ess <= tf.log(tf.to_float(num_particles) / 2.0)


def never_resample_criterion(log_weights, unused_t):
  """A criterion that never resamples."""
  batch_size = tf.shape(log_weights)[1]
  return tf.cast(tf.zeros([batch_size]), tf.bool)


def always_resample_criterion(log_weights, unused_t):
  """A criterion resamples at every timestep."""
  batch_size = tf.shape(log_weights)[1]
  return tf.cast(tf.ones([batch_size]), tf.bool)


def multinomial_resampling(log_weights, states, num_particles, batch_size,
                           random_seed=None):
  """Resample states with multinomial resampling.

  Args:
    log_weights: A [num_particles, batch_size] Tensor representing a batch
      of batch_size logits for num_particles-ary Categorical distribution.
    states: A nested list of [batch_size*num_particles, data_size] Tensors that
      will be resampled from the groups of every num_particles-th row.
    num_particles: The number of particles/samples.
    batch_size: The batch size.
    random_seed: The random seed to pass to the resampling operations in
      the particle filter. Mainly useful for testing.

  Returns:
    resampled_states: A nested list of [batch_size*num_particles, data_size]
      Tensors resampled via multinomial sampling.
  """
  # Calculate the ancestor indices via resampling. Because we maintain the
  # log unnormalized weights, we pass the weights in as logits, allowing
  # the distribution object to apply a softmax and normalize them.
  resampling_parameters = tf.transpose(log_weights, perm=[1, 0])
  resampling_dist = tf.contrib.distributions.Categorical(
      logits=resampling_parameters)
  ancestors = tf.stop_gradient(
      resampling_dist.sample(sample_shape=num_particles, seed=random_seed))

  # Because the batch is flattened, we must modify ancestor_inds to index the
  # proper samples. The particles in the ith filter are distributed every
  # batch_size rows in the batch, and offset i rows from the top. So, to
  # correct the indices we multiply by the batch_size and add the proper offset.
  # Crucially, when ancestor_inds is flattened the layout of the batch is
  # maintained.
  offset = tf.expand_dims(tf.range(batch_size), 0)
  ancestor_inds = tf.reshape(ancestors * batch_size + offset, [-1])

  resampled_states = nested.gather_tensors(states, ancestor_inds)
  return resampled_states


def _blend_tensor(blending_weights, tensor, num_particles, batch_size):
  """Blend tensor according to the weights.

  The first dimension of tensor is actually a 2d index compacted to a 1d
  index and similarly for blended_tensor. So if we index these Tensors
  by [(i, j), k], then

    blended_tensor[(i, j), k] =
      sum_l tensor[(l, j), :] * blending_weights[i, j, l].

  Args:
    blending_weights: [num_particles, batch_size, num_particles] weights where
      the indices represent [sample index, batch index, blending weight index].
    tensor: [num_particles * batch_size, state_dim] Tensor to be blended.
    num_particles: The number of particles/samples.
    batch_size: The batch size.

  Returns:
    blended_tensor: [num_particles*batch_size, state_dim] blended Tensor.
  """
  # tensor is currently [num_particles * batch_size, state_dim], so we reshape
  # it to [num_particles, batch_size, state_dim]. Then, transpose it to
  # [batch_size, state_size, num_particles].
  tensor = tf.transpose(
      tf.reshape(tensor, [num_particles, batch_size, -1]), perm=[1, 2, 0])
  blending_weights = tf.transpose(blending_weights, perm=[1, 2, 0])
  # blendeding_weights is [batch index, blending weight index, sample index].
  # Multiplying these gives a matrix of size [batch_size, state_size,
  # num_particles].
  tensor = tf.matmul(tensor, blending_weights)
  # transpose the tensor to be [num_particles, batch_size, state_size]
  # and then reshape it to match the original format.
  tensor = tf.reshape(tf.transpose(tensor, perm=[2, 0, 1]),
                      [num_particles*batch_size, -1])
  return tensor


def relaxed_resampling(log_weights, states, num_particles, batch_size,
                       temperature=0.5, random_seed=None):
  """Resample states with relaxed resampling.

  Draw soft "ancestors" using the Gumbel-Softmax distribution.

  Args:
    log_weights: A [num_particles, batch_size] Tensor representing a batch
      of batch_size logits for num_particles-ary Categorical distribution.
    states: A nested list of [batch_size * num_particles, d] Tensors that will
      be resampled from the groups of every num_particles-th row.
    num_particles: The number of particles/samples.
    batch_size: The batch size.
    temperature: The temperature used for the relaxed one hot distribution.
    random_seed: The random seed to pass to the resampling operations in
      the particle filter. Mainly useful for testing.

  Returns:
    resampled_states: A nested list of [batch_size * num_particles, d]
      Tensors resampled via multinomial sampling.
  """
  # log_weights are [num_particles, batch_size], so we transpose to get a
  # set of batch_size distributions over [0, num_particles).
  resampling_parameters = tf.transpose(log_weights, perm=[1, 0])
  resampling_dist = tf.contrib.distributions.RelaxedOneHotCategorical(
      temperature,
      logits=resampling_parameters)

  # Sample num_particles samples from the distribution, resulting in a
  # [num_particles, batch_size, num_particles] Tensor that represents a set of
  # [num_particles, batch_size] blending weights. The dimensions represent
  # [particle index, batch index, blending weight index].
  ancestors = resampling_dist.sample(sample_shape=num_particles,
                                     seed=random_seed)
  def map_fn(tensor):
    return _blend_tensor(ancestors, tensor, num_particles, batch_size)

  resampled_states = nested.map_nested(map_fn, states)
  return resampled_states


def smc(
    transition_fn,
    num_steps,
    num_particles=1,
    resampling_criterion=ess_criterion,
    resampling_fn=multinomial_resampling,
    loop_fn=None,
    parallel_iterations=30,
    swap_memory=True):
  """Run a sequential Monte Carlo (SMC) algorithm.

  This method runs an SMC algorithm that evolves systems of particles
  using the supplied transition function for the specified number of steps. The
  particles are optionally resampled using resampling_fn when indicated by
  resampling_criterion.

  Args:
    transition_fn: A callable that propogates a batch of particles one step.
      Must accept as arguments a batch of particle states and the current
      timestep. Must return the particle states one timestep in the future, the
      incremental weights of each particle as a [num_samples*batch_size] float
      Tensor, and optionally a set of arguments to pass to the loop_fn. If
      the loop args are not provided, they will be set to None. Before the
      first timestep transition_fn will be called with the arguments None, -1
      and should return the initial particle states.
    num_steps: A [batch_size] Tensor of ints representing the number of steps
      to run each filter for.
    num_particles: A scalar int, the number of particles to use in each filter.
    resampling_criterion: The resampling criterion to use for this particle
      filter. Must accept the current log weights and timestep and
      return a boolean Tensor of shape [batch_size] indicating whether each
      particle filter should resample. See ess_criterion and related functions
      for examples. When resampling_criterion is never_resample_criterion,
      resampling_fn is ignored and never called.
    resampling_fn: A callable that performs the resampling operation. Must
      accept as arguments the log weights, particle states, num_particles,
      and batch_size and return the resampled particle states. See
      multinomial_resampling and relaxed_resampling for examples.
    loop_fn: A callable that performs operations on the weights and
      particle states, useful for accumulating and processing state that
      shouldn't be resampled. At each timestep after (possibly) resampling
      loop_fn will be called with the previous loop_state, a set of arguments
      produced by transition_fn called loop_args, the resampled particle states,
      the current log weights as [num_particles, batch_size] float Tensor, a
      [batch_size] float Tensor representing whether or not each filter
      resampled, the current mask indicating which filters are active, and the
      current timestep. It must return the next loop state. Before the first
      timestep loop_fn will be called with the arguments None, None, None, None,
      -1 and must return the initial loop state. The loop state can be a
      possibly nested structure of Tensors and TensorArrays.
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop. Note that values greater than 1 can introduce
      non-determinism even when resampling is deterministic.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.

  Returns:
    log_z_hat: A Tensor of shape [batch_size] containing an estimate of the log
      normalizing constant that converts between the unormalized target
      distribution (as defined by the weights) and the true target distribution.
    log_weights: A Tensor of shape [max_num_steps, batch_size, num_particles]
      containing the log weights at each timestep of the particle filter.
      Will not be valid for timesteps past the supplied num_steps.
    resampled: A float Tensor of shape [max_num_steps, batch_size] indicating
      when the particle filters resampled. Will be 1.0 on timesteps when
      resampling occurred and 0.0 on timesteps when it did not.
    final_loop_state: The final state returned by loop_fn. If loop_fn is None
      then 0 will be returned.
  """
  # batch_size represents the number of particle filters running in parallel.
  batch_size = tf.shape(num_steps)[0]
  # Create a TensorArray where element t is the [num_particles*batch_size]
  # sequence mask for timestep t.
  max_num_steps = tf.reduce_max(num_steps)
  seq_mask = tf.transpose(
      tf.sequence_mask(num_steps, maxlen=max_num_steps, dtype=tf.float32),
      perm=[1, 0])
  seq_mask = tf.tile(seq_mask, [1, num_particles])
  mask_ta = tf.TensorArray(seq_mask.dtype,
                           max_num_steps,
                           name='mask_ta')
  mask_ta = mask_ta.unstack(seq_mask)
  # Initialize the state.
  t0 = tf.constant(0, tf.int32)
  init_particle_state = transition_fn(None, -1)

  def transition(*args):
    transition_outs = transition_fn(*args)
    if len(transition_outs) == 2:
      return transition_outs + (None,)
    else:
      return transition_outs

  if loop_fn is None:
    loop_fn = lambda *args: 0

  init_loop_state = loop_fn(None, None, None, None, None, None, -1)
  init_states = (init_particle_state, init_loop_state)
  ta_names = ['log_weights', 'resampled']
  tas = [tf.TensorArray(tf.float32, max_num_steps, name='%s_ta' % n)
         for n in ta_names]
  log_weights_acc = tf.zeros([num_particles, batch_size], dtype=tf.float32)
  log_z_hat_acc = tf.zeros([batch_size], dtype=tf.float32)

  def while_predicate(t, *unused_args):
    return t < max_num_steps

  def while_step(t, state, tas, log_weights_acc, log_z_hat_acc):
    """Implements one timestep of the particle filter."""
    particle_state, loop_state = state
    cur_mask = nested.read_tas(mask_ta, t)
    # Propagate the particles one step.
    log_alpha, new_particle_state, loop_args = transition(particle_state, t)
    # Update the current weights with the incremental weights.
    log_alpha *= cur_mask
    log_alpha = tf.reshape(log_alpha, [num_particles, batch_size])
    log_weights_acc += log_alpha

    should_resample = resampling_criterion(log_weights_acc, t)

    if resampling_criterion == never_resample_criterion:
      resampled = tf.to_float(should_resample)
    else:
      # Compute the states as if we did resample.
      resampled_states = resampling_fn(
          log_weights_acc,
          new_particle_state,
          num_particles,
          batch_size)
      # Decide whether or not we should resample; don't resample if we are past
      # the end of a sequence.
      should_resample = tf.logical_and(should_resample,
                                       cur_mask[:batch_size] > 0.)
      float_should_resample = tf.to_float(should_resample)
      new_particle_state = nested.where_tensors(
          tf.tile(should_resample, [num_particles]),
          resampled_states,
          new_particle_state)
      resampled = float_should_resample

    new_loop_state = loop_fn(loop_state, loop_args, new_particle_state,
                             log_weights_acc, resampled, cur_mask, t)
    # Update log Z hat.
    log_z_hat_update = tf.reduce_logsumexp(
        log_weights_acc, axis=0) - tf.log(tf.to_float(num_particles))
    # If it is the last timestep, always add the update.
    log_z_hat_acc += tf.cond(t < max_num_steps - 1,
                             lambda: log_z_hat_update * resampled,
                             lambda: log_z_hat_update)
    # Update the TensorArrays before we reset the weights so that we capture
    # the incremental weights and not zeros.
    ta_updates = [log_weights_acc, resampled]
    new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]
    # For the particle filters that resampled, reset weights to zero.
    log_weights_acc *= (1. - tf.tile(resampled[tf.newaxis, :],
                                     [num_particles, 1]))
    new_state = (new_particle_state, new_loop_state)
    return t + 1, new_state, new_tas, log_weights_acc, log_z_hat_acc

  _, final_state, tas, _, log_z_hat = tf.while_loop(
      while_predicate,
      while_step,
      loop_vars=(t0, init_states, tas, log_weights_acc, log_z_hat_acc),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  log_weights, resampled = [x.stack() for x in tas]
  log_weights = tf.transpose(log_weights, perm=[0, 2, 1])
  final_particle_state, final_loop_state = final_state
  return (log_z_hat, log_weights, resampled,
          final_particle_state, final_loop_state)
