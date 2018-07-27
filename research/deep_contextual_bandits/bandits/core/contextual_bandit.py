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

"""Define a contextual bandit from which we can sample and compute rewards.

We can feed the data, sample a context, its reward for a specific action, and
also the optimal action for a given context.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def run_contextual_bandit(context_dim, num_actions, dataset, algos):
  """Run a contextual bandit problem on a set of algorithms.

  Args:
    context_dim: Dimension of the context.
    num_actions: Number of available actions.
    dataset: Matrix where every row is a context + num_actions rewards.
    algos: List of algorithms to use in the contextual bandit instance.

  Returns:
    h_actions: Matrix with actions: size (num_context, num_algorithms).
    h_rewards: Matrix with rewards: size (num_context, num_algorithms).
  """

  num_contexts = dataset.shape[0]

  # Create contextual bandit
  cmab = ContextualBandit(context_dim, num_actions)
  cmab.feed_data(dataset)

  h_actions = np.empty((0, len(algos)), float)
  h_rewards = np.empty((0, len(algos)), float)

  # Run the contextual bandit process
  for i in range(num_contexts):
    context = cmab.context(i)
    actions = [a.action(context) for a in algos]
    rewards = [cmab.reward(i, action) for action in actions]

    for j, a in enumerate(algos):
      a.update(context, actions[j], rewards[j])

    h_actions = np.vstack((h_actions, np.array(actions)))
    h_rewards = np.vstack((h_rewards, np.array(rewards)))

  return h_actions, h_rewards


class ContextualBandit(object):
  """Implements a Contextual Bandit with d-dimensional contexts and k arms."""

  def __init__(self, context_dim, num_actions):
    """Creates a contextual bandit object.

    Args:
      context_dim: Dimension of the contexts.
      num_actions: Number of arms for the multi-armed bandit.
    """

    self._context_dim = context_dim
    self._num_actions = num_actions

  def feed_data(self, data):
    """Feeds the data (contexts + rewards) to the bandit object.

    Args:
      data: Numpy array with shape [n, d+k], where n is the number of contexts,
        d is the dimension of each context, and k the number of arms (rewards).

    Raises:
      ValueError: when data dimensions do not correspond to the object values.
    """

    if data.shape[1] != self.context_dim + self.num_actions:
      raise ValueError('Data dimensions do not match.')

    self._number_contexts = data.shape[0]
    self.data = data
    self.order = range(self.number_contexts)

  def reset(self):
    """Randomly shuffle the order of the contexts to deliver."""
    self.order = np.random.permutation(self.number_contexts)

  def context(self, number):
    """Returns the number-th context."""
    return self.data[self.order[number]][:self.context_dim]

  def reward(self, number, action):
    """Returns the reward for the number-th context and action."""
    return self.data[self.order[number]][self.context_dim + action]

  def optimal(self, number):
    """Returns the optimal action (in hindsight) for the number-th context."""
    return np.argmax(self.data[self.order[number]][self.context_dim:])

  @property
  def context_dim(self):
    return self._context_dim

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def number_contexts(self):
    return self._number_contexts
