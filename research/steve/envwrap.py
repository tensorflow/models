from builtins import object
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

try:
  import roboschool
except:
  pass
import gym
import numpy as np

from config import config

MAX_FRAMES = config["env"]["max_frames"]

gym.logger.level=40

def get_env(env_name, *args, **kwargs):
  MAPPING = {
    "CartPole-v0": CartPoleWrapper,
  }
  if env_name in MAPPING: return MAPPING[env_name](env_name, *args, **kwargs)
  else: return NoTimeLimitMujocoWrapper(env_name, *args, **kwargs)

class GymWrapper(object):
  """
  Generic wrapper for OpenAI gym environments.
  """
  def __init__(self, env_name):
    self.internal_env = gym.make(env_name)
    self.observation_space = self.internal_env.observation_space
    self.action_space = self.internal_env.action_space
    self.custom_init()

  def custom_init(self):
    pass

  def reset(self):
    self.clock = 0
    return self.preprocess_obs(self.internal_env.reset())

  # returns normalized actions
  def sample(self):
    return self.action_space.sample()

  # this is used for converting continuous approximations back to the original domain
  def normalize_actions(self, actions):
    return actions

  # puts actions into a form where they can be predicted. by default, called after sample()
  def unnormalize_actions(self, actions):
    return actions

  def preprocess_obs(self, obs):
    # return np.append(obs, [self.clock/float(MAX_FRAMES)])
    return obs

  def step(self, normalized_action):
    out = self.internal_env.step(normalized_action)
    self.clock += 1
    obs, reward, done = self.preprocess_obs(out[0]), out[1], float(out[2])
    reset = done == 1. or self.clock == MAX_FRAMES
    return obs, reward, done, reset

  def render_rollout(self, states):
    ## states is numpy array of size [timesteps, state]
    self.internal_env.reset()
    for state in states:
      self.internal_env.env.state = state
      self.internal_env.render()

class CartPoleWrapper(GymWrapper):
  """
  Wrap CartPole.
  """
  def sample(self):
    return np.array([np.random.uniform(0., 1.)])

  def normalize_actions(self, action):
    return 1 if action[0] >= 0 else 0

  def unnormalize_actions(self, action):
    return 2. * action - 1.

class NoTimeLimitMujocoWrapper(GymWrapper):
  """
  Wrap Mujoco-style environments, removing the termination condition after time.
  This is needed to keep it Markovian.
  """
  def __init__(self, env_name):
    self.internal_env = gym.make(env_name).env
    self.observation_space = self.internal_env.observation_space
    self.action_space = self.internal_env.action_space
    self.custom_init()
