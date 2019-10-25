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

"""Several functions to sample contextual data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sample_contextual_data(num_contexts, dim_context, num_actions, sigma):
  """Samples independent Gaussian data.

  There is nothing to learn here as the rewards do not depend on the context.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sigma: Standard deviation of the independent Gaussian samples.

  Returns:
    data: A [num_contexts, dim_context + num_actions] numpy array with the data.
  """
  size_data = [num_contexts, dim_context + num_actions]
  return np.random.normal(scale=sigma, size=size_data)


def sample_linear_data(num_contexts, dim_context, num_actions, sigma=0.0):
  """Samples data from linearly parameterized arms.

  The reward for context X and arm j is given by X^T beta_j, for some latent
  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly
  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sigma: Standard deviation of the additive noise. Set to zero for no noise.

  Returns:
    data: A [n, d+k] numpy array with the data.
    betas: Latent parameters that determine expected reward for each arm.
    opt: (optimal_rewards, optimal_actions) for all contexts.
  """

  betas = np.random.uniform(-1, 1, (dim_context, num_actions))
  betas /= np.linalg.norm(betas, axis=0)
  contexts = np.random.normal(size=[num_contexts, dim_context])
  rewards = np.dot(contexts, betas)
  opt_actions = np.argmax(rewards, axis=1)
  rewards += np.random.normal(scale=sigma, size=rewards.shape)
  opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
  return np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions)


def sample_sparse_linear_data(num_contexts, dim_context, num_actions,
                              sparse_dim, sigma=0.0):
  """Samples data from sparse linearly parameterized arms.

  The reward for context X and arm j is given by X^T beta_j, for some latent
  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly
  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.
  Only s components out of d are non-zero for each arm's beta.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sparse_dim: Dimension of the latent subspace (sparsity pattern dimension).
    sigma: Standard deviation of the additive noise. Set to zero for no noise.

  Returns:
    data: A [num_contexts, dim_context+num_actions] numpy array with the data.
    betas: Latent parameters that determine expected reward for each arm.
    opt: (optimal_rewards, optimal_actions) for all contexts.
  """

  flatten = lambda l: [item for sublist in l for item in sublist]
  sparse_pattern = flatten(
      [[(j, i) for j in np.random.choice(range(dim_context),
                                         sparse_dim,
                                         replace=False)]
       for i in range(num_actions)])
  betas = np.random.uniform(-1, 1, (dim_context, num_actions))
  mask = np.zeros((dim_context, num_actions))
  for elt in sparse_pattern:
    mask[elt] = 1
  betas = np.multiply(betas, mask)
  betas /= np.linalg.norm(betas, axis=0)
  contexts = np.random.normal(size=[num_contexts, dim_context])
  rewards = np.dot(contexts, betas)
  opt_actions = np.argmax(rewards, axis=1)
  rewards += np.random.normal(scale=sigma, size=rewards.shape)
  opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
  return np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions)


def sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v,
                             mu_large, std_large):
  """Samples from Wheel bandit game (see https://arxiv.org/abs/1802.09127).

  Args:
    num_contexts: Number of points to sample, i.e. (context, action rewards).
    delta: Exploration parameter: high reward in one region if norm above delta.
    mean_v: Mean reward for each action if context norm is below delta.
    std_v: Gaussian reward std for each action if context norm is below delta.
    mu_large: Mean reward for optimal action if context norm is above delta.
    std_large: Reward std for optimal action if context norm is above delta.

  Returns:
    dataset: Sampled matrix with n rows: (context, action rewards).
    opt_vals: Vector of expected optimal (reward, action) for each context.
  """

  context_dim = 2
  num_actions = 5

  data = []
  rewards = []
  opt_actions = []
  opt_rewards = []

  # sample uniform contexts in unit ball
  while len(data) < num_contexts:
    raw_data = np.random.uniform(-1, 1, (int(num_contexts / 3), context_dim))

    for i in range(raw_data.shape[0]):
      if np.linalg.norm(raw_data[i, :]) <= 1:
        data.append(raw_data[i, :])

  contexts = np.stack(data)[:num_contexts, :]

  # sample rewards
  for i in range(num_contexts):
    r = [np.random.normal(mean_v[j], std_v[j]) for j in range(num_actions)]
    if np.linalg.norm(contexts[i, :]) >= delta:
      # large reward in the right region for the context
      r_big = np.random.normal(mu_large, std_large)
      if contexts[i, 0] > 0:
        if contexts[i, 1] > 0:
          r[0] = r_big
          opt_actions.append(0)
        else:
          r[1] = r_big
          opt_actions.append(1)
      else:
        if contexts[i, 1] > 0:
          r[2] = r_big
          opt_actions.append(2)
        else:
          r[3] = r_big
          opt_actions.append(3)
    else:
      opt_actions.append(np.argmax(mean_v))

    opt_rewards.append(r[opt_actions[-1]])
    rewards.append(r)

  rewards = np.stack(rewards)
  opt_rewards = np.array(opt_rewards)
  opt_actions = np.array(opt_actions)

  return np.hstack((contexts, rewards)), (opt_rewards, opt_actions)
