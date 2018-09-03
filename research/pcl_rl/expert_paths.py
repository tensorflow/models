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

"""Expert paths/trajectories.

For producing or loading expert trajectories in environment.
"""

import tensorflow as tf
import random
import os
import numpy as np
from six.moves import xrange
import pickle

gfile = tf.gfile


def sample_expert_paths(num, env_str, env_spec,
                        load_trajectories_file=None):
  """Sample a number of expert paths randomly."""
  if load_trajectories_file is not None:
    if not gfile.Exists(load_trajectories_file):
      assert False, 'trajectories file %s does not exist' % load_trajectories_file

    with gfile.GFile(load_trajectories_file, 'r') as f:
      episodes = pickle.load(f)
      episodes = random.sample(episodes, num)
      return [ep[1:] for ep in episodes]

  return [sample_expert_path(env_str, env_spec)
          for _ in xrange(num)]


def sample_expert_path(env_str, env_spec):
  """Algorithmic tasks have known distribution of expert paths we sample from."""
  t = random.randint(2, 10)  # sequence length
  observations = []
  actions = [env_spec.initial_act(None)]
  rewards = []

  if env_str in ['DuplicatedInput-v0', 'Copy-v0']:
    chars = 5
    random_ints = [int(random.random() * 1000) for _ in xrange(t)]
    for tt in xrange(t):
      char_idx = tt // 2 if env_str == 'DuplicatedInput-v0' else tt
      char = random_ints[char_idx] % chars
      observations.append([char])
      actions.append([1, (tt + 1) % 2, char])
      rewards.append((tt + 1) % 2)
  elif env_str in ['RepeatCopy-v0']:
    chars = 5

    random_ints = [int(random.random() * 1000) for _ in xrange(t)]
    for tt in xrange(3 * t + 2):
      char_idx = (tt if tt < t else
                  2 * t - tt if tt <= 2 * t else
                  tt - 2 * t - 2)
      if tt in [t, 2 * t + 1]:
        char = chars
      else:
        char = random_ints[char_idx] % chars
      observations.append([char])
      actions.append([1 if tt < t else 0 if tt <= 2 * t else 1,
                      tt not in [t, 2 * t + 1], char])
      rewards.append(actions[-1][-2])
  elif env_str in ['Reverse-v0']:
    chars = 2
    random_ints = [int(random.random() * 1000) for _ in xrange(t)]
    for tt in xrange(2 * t + 1):
      char_idx = tt if tt < t else 2 * t - tt
      if tt != t:
        char = random_ints[char_idx] % chars
      else:
        char = chars
      observations.append([char])
      actions.append([tt < t, tt > t, char])
      rewards.append(tt > t)
  elif env_str in ['ReversedAddition-v0']:
    chars = 3
    random_ints = [int(random.random() * 1000) for _ in xrange(1 + 2 * t)]
    carry = 0
    char_history = []
    move_map = {0: 3, 1: 1, 2: 2, 3: 1}
    for tt in xrange(2 * t + 1):
      char_idx = tt
      if tt >= 2 * t:
        char = chars
      else:
        char = random_ints[char_idx] % chars
      char_history.append(char)
      if tt % 2 == 1:
        tot = char_history[-2] + char_history[-1] + carry
        carry = tot // chars
        tot = tot % chars
      elif tt == 2 * t:
        tot = carry
      else:
        tot = 0
      observations.append([char])
      actions.append([move_map[tt % len(move_map)],
                      tt % 2 or tt == 2 * t, tot])
      rewards.append(tt % 2 or tt == 2 * t)
  elif env_str in ['ReversedAddition3-v0']:
    chars = 3
    random_ints = [int(random.random() * 1000) for _ in xrange(1 + 3 * t)]
    carry = 0
    char_history = []
    move_map = {0: 3, 1: 3, 2: 1, 3: 2, 4:2, 5: 1}
    for tt in xrange(3 * t + 1):
      char_idx = tt
      if tt >= 3 * t:
        char = chars
      else:
        char = random_ints[char_idx] % chars
      char_history.append(char)
      if tt % 3 == 2:
        tot = char_history[-3] + char_history[-2] + char_history[-1] + carry
        carry = tot // chars
        tot = tot % chars
      elif tt == 3 * t:
        tot = carry
      else:
        tot = 0
      observations.append([char])
      actions.append([move_map[tt % len(move_map)],
                      tt % 3 == 2 or tt == 3 * t, tot])
      rewards.append(tt % 3 == 2 or tt == 3 * t)

  else:
    assert False, 'No expert trajectories for env %s' % env_str

  actions = [
      env_spec.convert_env_actions_to_actions(act)
      for act in actions]
  observations.append([chars])

  observations = [np.array(obs) for obs in zip(*observations)]
  actions = [np.array(act) for act in zip(*actions)]
  rewards = np.array(rewards)
  return [observations, actions, rewards, True]
