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

"""An interface representing the topology of an environment.

Allows for high level planning and high level instruction generation for
navigation tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum
import gym
import gin


@gin.config.constants_from_enum
class ModalityTypes(enum.Enum):
  """Types of the modalities that can be used."""
  IMAGE = 0
  SEMANTIC_SEGMENTATION = 1
  OBJECT_DETECTION = 2
  DEPTH = 3
  GOAL = 4
  PREV_ACTION = 5
  PREV_SUCCESS = 6
  STATE = 7
  DISTANCE = 8
  CAN_STEP = 9

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value
    return NotImplemented


class TaskEnvInterface(object):
  """Interface for an environment topology.

  An environment can implement this interface if there is a topological graph
  underlying this environment. All paths below are defined as paths in this
  graph. Using path_to_actions function one can translate a topological path
  to a geometric path in the environment.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def random_step_sequence(self, min_len=None, max_len=None):
    """Generates a random sequence of actions and executes them.

    Args:
      min_len: integer, minimum length of a step sequence.
      max_len: integer, if it is set to non-None, the method returns only
        the first n steps of a random sequence. If the environment is
        computationally heavy this argument should be set to speed up the
        training and avoid unnecessary computations by the environment.

    Returns:
      A path, defined as a list of vertex indices, a list of actions, a list of
      states, and a list of step() return tuples.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  @abc.abstractmethod
  def targets(self):
    """A list of targets in the environment.

    Returns:
      A list of target locations.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  @abc.abstractproperty
  def state(self):
    """Returns the position for the current location of agent."""
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  @abc.abstractproperty
  def graph(self):
    """Returns a graph representing the environment topology.

    Returns:
      nx.Graph object.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  @abc.abstractmethod
  def vertex_to_pose(self, vertex_index):
    """Maps a vertex index to a pose in the environment.

    Pose of the camera can be represented by (x,y,theta) or (x,y,z,theta).
    Args:
      vertex_index: index of a vertex in the topology graph.

    Returns:
      A np.array of floats of size 3 or 4 representing the pose of the vertex.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  @abc.abstractmethod
  def pose_to_vertex(self, pose):
    """Maps a coordinate in the maze to the closest vertex in topology graph.

    Args:
      pose: np.array of floats containing a the pose of the view.

    Returns:
      index of a vertex.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  @abc.abstractmethod
  def observation(self, state):
    """Returns observation at location xy and orientation theta.

    Args:
      state: a np.array of floats containing coordinates of a location and
        orientation.

    Returns:
      Dictionary of observations in the case of multiple observations.
      The keys are the modality names and the values are the np.array of float
      of observations for corresponding modality.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')

  def action(self, init_state, final_state):
    """Computes the transition action from state1 to state2.

    If the environment is discrete and the views are not adjacent in the
    environment. i.e. it is not possible to move from the first view to the
    second view with one action it should return None. In the continuous case,
    it will be the continuous difference of first view and second view.

    Args:
      init_state: numpy array, the initial view of the agent.
      final_state: numpy array, the final view of the agent.
    """
    raise NotImplementedError(
        'Needs implementation as part of EnvTopology interface.')


@gin.configurable
class TaskEnv(gym.Env, TaskEnvInterface):
  """An environment which uses a Task to compute reward.

  The environment implements a a gym interface, as well as EnvTopology. The
  former makes sure it can be used within an RL training, while the latter
  makes sure it can be used by a Task.

  This environment requires _step_no_reward to be implemented, which steps
  through it but does not return reward. Instead, the reward calculation is
  delegated to the Task object, which in return can access needed properties
  of the environment. These properties are exposed via the EnvTopology
  interface.
  """

  def __init__(self, task=None):
    self._task = task

  def set_task(self, task):
    self._task = task

  @abc.abstractmethod
  def _step_no_reward(self, action):
    """Same as _step without returning reward.

    Args:
      action: see _step.

    Returns:
      state, done, info as defined in _step.
    """
    raise NotImplementedError('Implement step.')

  @abc.abstractmethod
  def _reset_env(self):
    """Resets the environment. Returns initial observation."""
    raise NotImplementedError('Implement _reset. Must call super!')

  def step(self, action):
    obs, done, info = self._step_no_reward(action)

    reward = 0.0
    if self._task is not None:
      obs, reward, done, info = self._task.reward(obs, done, info)

    return obs, reward, done, info

  def reset(self):
    """Resets the environment. Gym API."""
    obs = self._reset_env()
    if self._task is not None:
      self._task.reset(obs)
    return obs
