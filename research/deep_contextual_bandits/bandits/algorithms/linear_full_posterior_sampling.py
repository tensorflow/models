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

"""Contextual algorithm that keeps a full linear posterior for each arm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset


class LinearFullPosteriorSampling(BanditAlgorithm):
  """Thompson Sampling with independent linear models and unknown noise var."""

  def __init__(self, name, hparams):
    """Initialize posterior distributions and hyperparameters.

    Assume a linear model for each action i: reward = context^T beta_i + noise
    Each beta_i has a Gaussian prior (lambda parameter), each sigma2_i (noise
    level) has an inverse Gamma prior (a0, b0 parameters). Mean, covariance,
    and precision matrices are initialized, and the ContextualDataset created.

    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters of the algorithm.
    """

    self.name = name
    self.hparams = hparams

    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior

    self.mu = [
        np.zeros(self.hparams.context_dim + 1)
        for _ in range(self.hparams.num_actions)
    ]

    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.hparams.context_dim + 1)
                for _ in range(self.hparams.num_actions)]

    self.precision = [
        self.lambda_prior * np.eye(self.hparams.context_dim + 1)
        for _ in range(self.hparams.num_actions)
    ]

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.hparams.num_actions)]
    self.b = [self._b0 for _ in range(self.hparams.num_actions)]

    self.t = 0
    self.data_h = ContextualDataset(hparams.context_dim,
                                    hparams.num_actions,
                                    intercept=True)

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly.

    Args:
      context: Context for which the action need to be chosen.

    Returns:
      action: Selected action for the context.
    """

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    # Sample sigma2, and beta conditional on sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i])
        for i in range(self.hparams.num_actions)
    ]

    try:
      beta_s = [
          np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
          for i in range(self.hparams.num_actions)
      ]
    except np.linalg.LinAlgError as e:
      # Sampling could fail if covariance is not positive definite
      print('Exception when sampling from {}.'.format(self.name))
      print('Details: {} | {}.'.format(e.message, e.args))
      d = self.hparams.context_dim + 1
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for i in range(self.hparams.num_actions)
      ]

    # Compute sampled expected values, intercept is last component of beta
    vals = [
        np.dot(beta_s[i][:-1], context.T) + beta_s[i][-1]
        for i in range(self.hparams.num_actions)
    ]

    return np.argmax(vals)

  def update(self, context, action, reward):
    """Updates action posterior using the linear Bayesian regression formula.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    self.data_h.add(context, action, reward)

    # Update posterior of action with formulas: \beta | x,y ~ N(mu_q, cov_q)
    x, y = self.data_h.get_data(action)

    # The algorithm could be improved with sequential update formulas (cheaper)
    s = np.dot(x.T, x)

    # Some terms are removed as we assume prior mu_0 = 0.
    precision_a = s + self.lambda_prior * np.eye(self.hparams.context_dim + 1)
    cov_a = np.linalg.inv(precision_a)
    mu_a = np.dot(cov_a, np.dot(x.T, y))

    # Inverse Gamma posterior update
    a_post = self.a0 + x.shape[0] / 2.0
    b_upd = 0.5 * (np.dot(y.T, y) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
    b_post = self.b0 + b_upd

    # Store new posterior distributions
    self.mu[action] = mu_a
    self.cov[action] = cov_a
    self.precision[action] = precision_a
    self.a[action] = a_post
    self.b[action] = b_post

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
