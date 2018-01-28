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

"""Utilities for environment interface with agent / tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange


class spaces(object):
  discrete = 0
  box = 1


def get_space(space):
  if hasattr(space, 'n'):
    return space.n, spaces.discrete, None
  elif hasattr(space, 'shape'):
    return np.prod(space.shape), spaces.box, (space.low, space.high)


def get_spaces(spaces):
  if hasattr(spaces, 'spaces'):
    return zip(*[get_space(space) for space in spaces.spaces])
  else:
    return [(ret,) for ret in get_space(spaces)]


class EnvSpec(object):

  def __init__(self, env, try_combining_actions=True,
               discretize_actions=None):
    self.discretize_actions = discretize_actions

    # figure out observation space
    self.obs_space = env.observation_space
    self.obs_dims, self.obs_types, self.obs_info = get_spaces(self.obs_space)

    # figure out action space
    self.act_space = env.action_space
    self.act_dims, self.act_types, self.act_info = get_spaces(self.act_space)

    if self.discretize_actions:
      self._act_dims = self.act_dims[:]
      self._act_types = self.act_types[:]
      self.act_dims = []
      self.act_types = []
      for i, (dim, typ) in enumerate(zip(self._act_dims, self._act_types)):
        if typ == spaces.discrete:
          self.act_dims.append(dim)
          self.act_types.append(spaces.discrete)
        elif typ == spaces.box:
          for _ in xrange(dim):
            self.act_dims.append(self.discretize_actions)
            self.act_types.append(spaces.discrete)
    else:
      self._act_dims = None
      self._act_types = None

    if (try_combining_actions and
        all(typ == spaces.discrete for typ in self.act_types)):
      self.combine_actions = True
      self.orig_act_dims = self.act_dims[:]
      self.orig_act_types = self.act_types[:]
      total_act_dim = 1
      for dim in self.act_dims:
        total_act_dim *= dim
      self.act_dims = [total_act_dim]
      self.act_types = [spaces.discrete]
    else:
      self.combine_actions = False

    self.obs_dims_and_types = zip(self.obs_dims, self.obs_types)
    self.act_dims_and_types = zip(self.act_dims, self.act_types)

    self.total_obs_dim = sum(self.obs_dims)
    self.total_sampling_act_dim = sum(self.sampling_dim(dim, typ)
                                      for dim, typ in self.act_dims_and_types)
    self.total_sampled_act_dim = sum(self.act_dims)

  def sampling_dim(self, dim, typ):
    if typ == spaces.discrete:
      return dim
    elif typ == spaces.box:
      return 2 * dim  # Gaussian mean and std
    else:
      assert False

  def convert_actions_to_env(self, actions):
    if self.combine_actions:
      new_actions = []
      actions = actions[0]
      for dim in self.orig_act_dims:
        new_actions.append(np.mod(actions, dim))
        actions = (actions / dim).astype('int32')
      actions = new_actions

    if self.discretize_actions:
      new_actions = []
      idx = 0
      for i, (dim, typ) in enumerate(zip(self._act_dims, self._act_types)):
        if typ == spaces.discrete:
          new_actions.append(actions[idx])
          idx += 1
        elif typ == spaces.box:
          low, high = self.act_info[i]
          cur_action = []
          for j in xrange(dim):
            cur_action.append(
                low[j] + (high[j] - low[j]) * actions[idx] /
                float(self.discretize_actions))
            idx += 1
          new_actions.append(np.hstack(cur_action))
      actions = new_actions

    return actions

  def convert_env_actions_to_actions(self, actions):
    if not self.combine_actions:
      return actions

    new_actions = 0
    base = 1
    for act, dim in zip(actions, self.orig_act_dims):
      new_actions = new_actions + base * act
      base *= dim

    return [new_actions]

  def convert_obs_to_list(self, obs):
    if len(self.obs_dims) == 1:
      return [obs]
    else:
      return list(obs)

  def convert_action_to_gym(self, action):
    if len(action) == 1:
      return action[0]
    else:
      return list(action)
    if ((not self.combine_actions or len(self.orig_act_dims) == 1) and
        (len(self.act_dims) == 1 or
         (self.discretize_actions and len(self._act_dims) == 1))):
      return action[0]
    else:
      return list(action)

  def initial_obs(self, batch_size):
    batched = batch_size is not None
    batch_size = batch_size or 1

    obs = []
    for dim, typ in self.obs_dims_and_types:
      if typ == spaces.discrete:
        obs.append(np.zeros(batch_size))
      elif typ == spaces.box:
        obs.append(np.zeros([batch_size, dim]))

    if batched:
      return obs
    else:
      return zip(*obs)[0]

  def initial_act(self, batch_size=None):
    batched = batch_size is not None
    batch_size = batch_size or 1

    act = []
    for dim, typ in self.act_dims_and_types:
      if typ == spaces.discrete:
        act.append(-np.ones(batch_size))
      elif typ == spaces.box:
        act.append(-np.ones([batch_size, dim]))

    if batched:
      return act
    else:
      return zip(*act)[0]

  def is_discrete(self, typ):
    return typ == spaces.discrete

  def is_box(self, typ):
    return typ == spaces.box
