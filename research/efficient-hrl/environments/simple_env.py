import gin.tf
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
from tf_agents.environments import gym_wrapper
import os

class SimpleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  def __init__(self):
    self.t = 0
    mujoco_env.MujocoEnv.__init__(
      self,
      model_path=os.path.join(os.path.dirname(__file__), 'assets', 'simple.xml'),
      frame_skip=5)
    utils.EzPickle.__init__(self)


  def _get_obs(self):
    return np.array([self.t])

  def _step(self, a):
    return self.step(a)

  def step(self, a):
    self.t += 1
    reward = 0.01
    # done = (self.t >= 20)
    done = False

    return self._get_obs(), reward, done, dict()


  def reset_model(self):
    self.t = 0
    return self._get_obs()


@gin.configurable
def create_simple_env():
  return gym_wrapper.GymWrapper(SimpleEnv())