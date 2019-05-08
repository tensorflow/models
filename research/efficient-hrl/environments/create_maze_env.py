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

from environments.ant_maze_env import AntMazeEnv
from environments.point_maze_env import PointMazeEnv

import gin.tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
import tensorflow as tf


@gin.configurable
def create_maze_env(env_name=None, top_down_view=False, wrapped=True):
  n_bins = 0
  manual_collision = False
  if env_name.startswith('Ego'):
    n_bins = 8
    env_name = env_name[3:]
  if env_name.startswith('Ant'):
    cls = AntMazeEnv
    env_name = env_name[len('Ant'):]
    maze_size_scaling = 8
  elif env_name.startswith('Point'):
    cls = PointMazeEnv
    manual_collision = True
    env_name = env_name[5:]
    maze_size_scaling = 4
  else:
    assert False, 'unknown env %s' % env_name

  maze_id = None
  observe_blocks = False
  put_spin_near_agent = False
  if env_name == 'Maze':
    maze_id = 'Maze'
  elif env_name == 'Push':
    maze_id = 'Push'
  elif env_name == 'Fall':
    maze_id = 'Fall'
  elif env_name == 'Block':
    maze_id = 'Block'
    put_spin_near_agent = True
    observe_blocks = True
  elif env_name == 'BlockMaze':
    maze_id = 'BlockMaze'
    put_spin_near_agent = True
    observe_blocks = True
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  gym_mujoco_kwargs = {
      'maze_id': maze_id,
      'n_bins': n_bins,
      'observe_blocks': observe_blocks,
      'put_spin_near_agent': put_spin_near_agent,
      'top_down_view': top_down_view,
      'manual_collision': manual_collision,
      'maze_size_scaling': maze_size_scaling
  }
  gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  if wrapped:
    wrapped_env = gym_wrapper.GymWrapper(gym_env)
    return wrapped_env
  else:
    return gym_env


# class TFPyEnvironment(tf_py_environment.TFPyEnvironment):
#
#   def __init__(self, *args, **kwargs):
#     super(TFPyEnvironment, self).__init__(*args, **kwargs)
#     self._step_state = self.current_time_step()[-1]
#
#   def start_collect(self):
#     self._step_state = self.current_time_step()[-1]
#
#   def current_obs(self):
#     time_step, self._step_state = self.current_time_step(self._step_state)
#     return time_step.observation[0]  # For some reason, there is an extra dim.
#
#   def step(self, actions):
#     next_step, self._step_state = super(TFPyEnvironment, self).step(
#         actions, self._step_state)
#     return next_step, next_step.reward[0], next_step.discount[0]
#
#   def reset(self):
#     _, self._step_state, reset_op = super(TFPyEnvironment, self).reset(
#         self._step_state)
#     return reset_op


class TFPyEnvironment(tf_py_environment.TFPyEnvironment):

  def __init__(self, *args, **kwargs):
    super(TFPyEnvironment, self).__init__(*args, **kwargs)

  def start_collect(self):
    self._step_state = self.current_time_step()[-1]  # TODO: what does it do?

  def current_obs(self):
    time_step = self.current_time_step()
    return time_step.observation[0]  # For some reason, there is an extra dim.

  def step(self, actions):
    next_step = super(TFPyEnvironment, self).step(actions)
    return next_step, next_step.reward[0], next_step.discount[0]

  def reset(self):
    """Returns the current `TimeStep` after resetting the environment.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          timestep.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

    def _reset():
      with tf_py_environment._check_not_called_concurrently(self._lock):
        self._time_step = self._env.reset()

    with tf.name_scope('reset'):
      reset_op = tf.py_func(
        _reset,
        [],  # No inputs.
        [],
        stateful=True,
        name='reset_py_func')
    return reset_op
