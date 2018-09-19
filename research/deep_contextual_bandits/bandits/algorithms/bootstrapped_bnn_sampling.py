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

"""Contextual algorithm based on boostrapping neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel


class BootstrappedBNNSampling(BanditAlgorithm):
  """Thompson Sampling algorithm based on training several neural networks."""

  def __init__(self, name, hparams, optimizer='RMS'):
    """Creates a BootstrappedSGDSampling object based on a specific optimizer.

      hparams.q: Number of models that are independently trained.
      hparams.p: Prob of independently including each datapoint in each model.

    Args:
      name: Name given to the instance.
      hparams: Hyperparameters for each individual model.
      optimizer: Neural network optimization algorithm.
    """

    self.name = name
    self.hparams = hparams
    self.optimizer_n = optimizer

    self.training_freq = hparams.training_freq
    self.training_epochs = hparams.training_epochs
    self.t = 0

    self.q = hparams.q
    self.p = hparams.p

    self.datasets = [
        ContextualDataset(hparams.context_dim,
                          hparams.num_actions,
                          hparams.buffer_s)
        for _ in range(self.q)
    ]

    self.bnn_boot = [
        NeuralBanditModel(optimizer, hparams, '{}-{}-bnn'.format(name, i))
        for i in range(self.q)
    ]

  def action(self, context):
    """Selects action for context based on Thompson Sampling using one BNN."""

    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      # round robin until each action has been taken "initial_pulls" times
      return self.t % self.hparams.num_actions

    # choose model uniformly at random
    model_index = np.random.randint(self.q)

    with self.bnn_boot[model_index].graph.as_default():
      c = context.reshape((1, self.hparams.context_dim))
      output = self.bnn_boot[model_index].sess.run(
          self.bnn_boot[model_index].y_pred,
          feed_dict={self.bnn_boot[model_index].x: c})
      return np.argmax(output)

  def update(self, context, action, reward):
    """Updates the data buffer, and re-trains the BNN every self.freq_update."""

    self.t += 1
    for i in range(self.q):
      # include the data point with probability p independently in each dataset
      if np.random.random() < self.p or self.t < 2:
        self.datasets[i].add(context, action, reward)

    if self.t % self.training_freq == 0:
      # update all the models:
      for i in range(self.q):
        if self.hparams.reset_lr:
          self.bnn_boot[i].assign_lr()
        self.bnn_boot[i].train(self.datasets[i], self.training_epochs)
