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

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class NeuralLinearPosteriorSampling(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, hparams, optimizer='RMS'):

    self.name = name
    self.hparams = hparams
    self.latent_dim = self.hparams.layer_sizes[-1]

    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior

    self.mu = [
        np.zeros(self.latent_dim)
        for _ in range(self.hparams.num_actions)
    ]

    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.latent_dim)
                for _ in range(self.hparams.num_actions)]

    self.precision = [
        self.lambda_prior * np.eye(self.latent_dim)
        for _ in range(self.hparams.num_actions)
    ]

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.hparams.num_actions)]
    self.b = [self._b0 for _ in range(self.hparams.num_actions)]

    # Regression and NN Update Frequency
    self.update_freq_lr = hparams.training_freq
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.optimizer_n = optimizer

    self.num_epochs = hparams.training_epochs
    self.data_h = ContextualDataset(hparams.context_dim,
                                    hparams.num_actions,
                                    intercept=False)
    self.latent_h = ContextualDataset(self.latent_dim,
                                      hparams.num_actions,
                                      intercept=False)
    self.bnn = NeuralBanditModel(optimizer, hparams, '{}-bnn'.format(name))

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

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
      print('Exception when sampling for {}.'.format(self.name))
      print('Details: {} | {}.'.format(e.message, e.args))
      d = self.latent_dim
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for i in range(self.hparams.num_actions)
      ]

    # Compute last-layer representation for the current context
    with self.bnn.graph.as_default():
      c = context.reshape((1, self.hparams.context_dim))
      z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c})

    # Apply Thompson Sampling to last-layer representation
    vals = [
        np.dot(beta_s[i], z_context.T) for i in range(self.hparams.num_actions)
    ]
    return np.argmax(vals)

  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1
    self.data_h.add(context, action, reward)
    c = context.reshape((1, self.hparams.context_dim))
    z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c})
    self.latent_h.add(z_context, action, reward)

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:

      if self.hparams.reset_lr:
        self.bnn.assign_lr()
      self.bnn.train(self.data_h, self.num_epochs)

      # Update the latent representation of every datapoint collected so far
      new_z = self.bnn.sess.run(self.bnn.nn,
                                feed_dict={self.bnn.x: self.data_h.contexts})
      self.latent_h.replace_data(contexts=new_z)

    # Update the Bayesian Linear Regression
    if self.t % self.update_freq_lr == 0:

      # Find all the actions to update
      actions_to_update = self.latent_h.actions[:-self.update_freq_lr]

      for action_v in np.unique(actions_to_update):

        # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
        z, y = self.latent_h.get_data(action_v)

        # The algorithm could be improved with sequential formulas (cheaper)
        s = np.dot(z.T, z)

        # Some terms are removed as we assume prior mu_0 = 0.
        precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(z.T, y))

        # Inverse Gamma posterior update
        a_post = self.a0 + z.shape[0] / 2.0
        b_upd = 0.5 * np.dot(y.T, y)
        b_upd -= 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
        b_post = self.b0 + b_upd

        # Store new posterior distributions
        self.mu[action_v] = mu_a
        self.cov[action_v] = cov_a
        self.precision[action_v] = precision_a
        self.a[action_v] = a_post
        self.b[action_v] = b_post

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
