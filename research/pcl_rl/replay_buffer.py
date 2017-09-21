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

"""Replay buffer.

Implements replay buffer in Python.
"""

import random
import numpy as np


class ReplayBuffer(object):

  def __init__(self, max_size):
    self.max_size = max_size
    self.cur_size = 0
    self.buffer = {}
    self.init_length = 0

  def __len__(self):
    return self.cur_size

  def seed_buffer(self, episodes):
    self.init_length = len(episodes)
    self.add(episodes, np.ones(self.init_length))

  def add(self, episodes, *args):
    """Add episodes to buffer."""
    idx = 0
    while self.cur_size < self.max_size and idx < len(episodes):
      self.buffer[self.cur_size] = episodes[idx]
      self.cur_size += 1
      idx += 1

    if idx < len(episodes):
      remove_idxs = self.remove_n(len(episodes) - idx)
      for remove_idx in remove_idxs:
        self.buffer[remove_idx] = episodes[idx]
        idx += 1

    assert len(self.buffer) == self.cur_size

  def remove_n(self, n):
    """Get n items for removal."""
    # random removal
    idxs = random.sample(xrange(self.init_length, self.cur_size), n)
    return idxs

  def get_batch(self, n):
    """Get batch of episodes to train on."""
    # random batch
    idxs = random.sample(xrange(self.cur_size), n)
    return [self.buffer[idx] for idx in idxs], None

  def update_last_batch(self, delta):
    pass


class PrioritizedReplayBuffer(ReplayBuffer):

  def __init__(self, max_size, alpha=0.2,
               eviction_strategy='rand'):
    self.max_size = max_size
    self.alpha = alpha
    self.eviction_strategy = eviction_strategy
    assert self.eviction_strategy in ['rand', 'fifo', 'rank']
    self.remove_idx = 0

    self.cur_size = 0
    self.buffer = {}
    self.priorities = np.zeros(self.max_size)
    self.init_length = 0

  def __len__(self):
    return self.cur_size

  def add(self, episodes, priorities, new_idxs=None):
    """Add episodes to buffer."""
    if new_idxs is None:
      idx = 0
      new_idxs = []
      while self.cur_size < self.max_size and idx < len(episodes):
        self.buffer[self.cur_size] = episodes[idx]
        new_idxs.append(self.cur_size)
        self.cur_size += 1
        idx += 1

      if idx < len(episodes):
        remove_idxs = self.remove_n(len(episodes) - idx)
        for remove_idx in remove_idxs:
          self.buffer[remove_idx] = episodes[idx]
          new_idxs.append(remove_idx)
          idx += 1
    else:
      assert len(new_idxs) == len(episodes)
      for new_idx, ep in zip(new_idxs, episodes):
        self.buffer[new_idx] = ep

    self.priorities[new_idxs] = priorities
    self.priorities[0:self.init_length] = np.max(
        self.priorities[self.init_length:])

    assert len(self.buffer) == self.cur_size
    return new_idxs

  def remove_n(self, n):
    """Get n items for removal."""
    assert self.init_length + n <= self.cur_size

    if self.eviction_strategy == 'rand':
      # random removal
      idxs = random.sample(xrange(self.init_length, self.cur_size), n)
    elif self.eviction_strategy == 'fifo':
      # overwrite elements in cyclical fashion
      idxs = [
          self.init_length +
          (self.remove_idx + i) % (self.max_size - self.init_length)
          for i in xrange(n)]
      self.remove_idx = idxs[-1] + 1 - self.init_length
    elif self.eviction_strategy == 'rank':
      # remove lowest-priority indices
      idxs = np.argpartition(self.priorities, n)[:n]

    return idxs

  def sampling_distribution(self):
    p = self.priorities[:self.cur_size]
    p = np.exp(self.alpha * (p - np.max(p)))
    norm = np.sum(p)
    if norm > 0:
      uniform = 0.0
      p = p / norm * (1 - uniform) + 1.0 / self.cur_size * uniform
    else:
      p = np.ones(self.cur_size) / self.cur_size
    return p

  def get_batch(self, n):
    """Get batch of episodes to train on."""
    p = self.sampling_distribution()
    idxs = np.random.choice(self.cur_size, size=n, replace=False, p=p)
    self.last_batch = idxs
    return [self.buffer[idx] for idx in idxs], p[idxs]

  def update_last_batch(self, delta):
    """Update last batch idxs with new priority."""
    self.priorities[self.last_batch] = np.abs(delta)
    self.priorities[0:self.init_length] = np.max(
        self.priorities[self.init_length:])
