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

"""Datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import models


def make_long_chain_dataset(
    state_size=1,
    num_obs=5,
    steps_per_obs=3,
    variance=1.,
    observation_variance=1.,
    batch_size=4,
    num_samples=1,
    observation_type=models.STANDARD_OBSERVATION,
    transition_type=models.STANDARD_TRANSITION,
    fixed_observation=None,
    dtype="float32"):
  """Creates a long chain data generating process.

  Creates a tf.data.Dataset that provides batches of data from a long
  chain.

  Args:
    state_size: The dimension of the state space of the process.
    num_obs: The number of observations in the chain.
    steps_per_obs: The number of steps between each observation.
    variance: The variance of the normal distributions used at each timestep.
    batch_size: The number of trajectories to include in each batch.
    num_samples: The number of replicas of each trajectory to include in each
      batch.
    dtype: The datatype of the states and observations.
  Returns:
    dataset: A tf.data.Dataset that can be iterated over.
  """
  num_timesteps = num_obs * steps_per_obs
  def data_generator():
    """An infinite generator of latents and observations from the model."""
    while True:
      states = []
      observations = []
      # z0 ~ Normal(0, sqrt(variance)).
      states.append(
          np.random.normal(size=[state_size],
                           scale=np.sqrt(variance)).astype(dtype))
      # start at 1 because we've already generated z0
      # go to num_timesteps+1 because we want to include the num_timesteps-th step
      for t in xrange(1, num_timesteps+1):
        if transition_type == models.ROUND_TRANSITION:
          loc = np.round(states[-1])
        elif transition_type == models.STANDARD_TRANSITION:
          loc = states[-1]
        new_state = np.random.normal(size=[state_size],
                                     loc=loc,
                                     scale=np.sqrt(variance))
        states.append(new_state.astype(dtype))
        if t % steps_per_obs == 0:
          if fixed_observation is None:
            if observation_type == models.SQUARED_OBSERVATION:
              loc = np.square(states[-1])
            elif observation_type == models.ABS_OBSERVATION:
              loc = np.abs(states[-1])
            elif observation_type == models.STANDARD_OBSERVATION:
              loc = states[-1]
            new_obs = np.random.normal(size=[state_size],
                                       loc=loc,
                                       scale=np.sqrt(observation_variance)).astype(dtype)
          else:
            new_obs = np.ones([state_size])* fixed_observation

          observations.append(new_obs)
      yield states, observations

  dataset = tf.data.Dataset.from_generator(
      data_generator,
      output_types=(tf.as_dtype(dtype), tf.as_dtype(dtype)),
      output_shapes=([num_timesteps+1, state_size], [num_obs, state_size]))
  dataset = dataset.repeat().batch(batch_size)

  def tile_batch(state, observation):
    state = tf.tile(state, [num_samples, 1, 1])
    observation = tf.tile(observation, [num_samples, 1, 1])
    return state, observation

  dataset = dataset.map(tile_batch, num_parallel_calls=12).prefetch(1024)
  return dataset


def make_dataset(bs=None,
                 state_size=1,
                 num_timesteps=10,
                 variance=1.,
                 prior_type="unimodal",
                 bimodal_prior_weight=0.5,
                 bimodal_prior_mean=1,
                 transition_type=models.STANDARD_TRANSITION,
                 fixed_observation=None,
                 batch_size=4,
                 num_samples=1,
                 dtype='float32'):
  """Creates a data generating process.

  Creates a tf.data.Dataset that provides batches of data.

  Args:
    bs: The parameters of the data generating process. If None, new bs are
      randomly generated.
    state_size: The dimension of the state space of the process.
    num_timesteps: The length of the state sequences in the process.
    variance: The variance of the normal distributions used at each timestep.
    batch_size: The number of trajectories to include in each batch.
    num_samples: The number of replicas of each trajectory to include in each
      batch.
  Returns:
    bs: The true bs used to generate the data
    dataset: A tf.data.Dataset that can be iterated over.
  """

  if bs is None:
    bs = [np.random.uniform(size=[state_size]).astype(dtype) for _ in xrange(num_timesteps)]
    tf.logging.info("data generating processs bs: %s",
                    np.array(bs).reshape(num_timesteps))


  def data_generator():
    """An infinite generator of latents and observations from the model."""
    while True:
      states = []
      if prior_type == "unimodal" or prior_type == "nonlinear":
        # Prior is Normal(0, sqrt(variance)).
        states.append(np.random.normal(size=[state_size], scale=np.sqrt(variance)).astype(dtype))
      elif prior_type == "bimodal":
        if np.random.uniform() > bimodal_prior_weight:
          loc = bimodal_prior_mean
        else:
          loc = - bimodal_prior_mean
        states.append(np.random.normal(size=[state_size],
                                       loc=loc,
                                       scale=np.sqrt(variance)
                                      ).astype(dtype))

      for t in xrange(num_timesteps):
        if transition_type == models.ROUND_TRANSITION:
          loc = np.round(states[-1])
        elif transition_type == models.STANDARD_TRANSITION:
          loc = states[-1]
        loc += bs[t]
        new_state = np.random.normal(size=[state_size],
                                     loc=loc,
                                     scale=np.sqrt(variance)).astype(dtype)
        states.append(new_state)

      if fixed_observation is None:
        observation = states[-1]
      else:
        observation = np.ones_like(states[-1]) * fixed_observation
      yield np.array(states[:-1]), observation

  dataset = tf.data.Dataset.from_generator(
      data_generator,
      output_types=(tf.as_dtype(dtype), tf.as_dtype(dtype)),
      output_shapes=([num_timesteps, state_size], [state_size]))
  dataset = dataset.repeat().batch(batch_size)

  def tile_batch(state, observation):
    state = tf.tile(state, [num_samples, 1, 1])
    observation = tf.tile(observation, [num_samples, 1])
    return state, observation

  dataset = dataset.map(tile_batch, num_parallel_calls=12).prefetch(1024)
  return np.array(bs), dataset
