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

""" Original MAB algorithm for Thompson Sampling, used for baseline compare """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset


class ThompsonSampling(BanditAlgorithm):
  """just Original Thompson Sampling"""

  def __init__(self, name, hparams):
    # """

    # Args:
    #   name: Name of the algorithm.
    #   hparams: Hyper-parameters of the algorithm.
    # """

    self.name = name
    self.hparams = hparams

    self.outs = np.array([0 for _ in range(self.hparams.num_actions)])
    self.wins = np.array([0 for _ in range(self.hparams.num_actions)])

  def action(self, context):
    """Samples beta's from stats

    Args:
      context: Context for which the action need to be chosen.

    Returns:
      action: Selected action for the context.
    """

    vals = [
        np.random.beta(1+self.wins[i], 1+self.outs[i]-self.wins[i])
        for i in range(self.hparams.num_actions)
    ]

    return np.argmax(vals)

  def update(self, context, action, reward):
    """Updates action Beta param using stats.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    if (reward > 0):
        self.wins[action] += 1
    self.outs[action] += 1
