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

"""Adapted from rllab maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import gym

import maze_env_utils

# Directory that contains mujoco xml files.
MODEL_DIR = 'environments/assets'


class MazeEnv(gym.Env):
  MODEL_CLASS = None

  MAZE_HEIGHT = None
  MAZE_SIZE_SCALING = None

  def __init__(
      self,
      maze_id=None,
      maze_height=0.5,
      maze_size_scaling=8,
      *args,
      **kwargs):
    self._maze_id = maze_id

    model_cls = self.__class__.MODEL_CLASS
    if model_cls is None:
      raise "MODEL_CLASS unspecified!"
    xml_path = os.path.join(MODEL_DIR, model_cls.FILE)
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    self.MAZE_HEIGHT = height = maze_height
    self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
    self.MAZE_STRUCTURE = structure = maze_env_utils.construct_maze(maze_id=self._maze_id)
    self.elevated = any(-1 in row for row in structure)  # Elevate the maze to allow for falling.
    self.blocks = any(
        any(maze_env_utils.can_move(r) for r in row)
        for row in structure)  # Are there any movable blocks?

    torso_x, torso_y = self._find_robot()
    self._init_torso_x = torso_x
    self._init_torso_y = torso_y

    height_offset = 0.
    if self.elevated:
      # Increase initial z-pos of ant.
      height_offset = height * size_scaling
      torso = tree.find(".//body[@name='torso']")
      torso.set('pos', '0 0 %.2f' % (0.75 + height_offset))
    if self.blocks:
      # If there are movable blocks, change simulation settings to perform
      # better contact detection.
      default = tree.find(".//default")
      default.find('.//geom').set('solimp', '.995 .995 .01')

    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if self.elevated and structure[i][j] not in [-1]:
          # Create elevated platform.
          ET.SubElement(
              worldbody, "geom",
              name="elevated_%d_%d" % (i, j),
              pos="%f %f %f" % (j * size_scaling - torso_x,
                                i * size_scaling - torso_y,
                                height / 2 * size_scaling),
              size="%f %f %f" % (0.5 * size_scaling,
                                 0.5 * size_scaling,
                                 height / 2 * size_scaling),
              type="box",
              material="",
              contype="1",
              conaffinity="1",
              rgba="0.9 0.9 0.9 1",
          )
        if structure[i][j] == 1:  # Unmovable block.
          # Offset all coordinates so that robot starts at the origin.
          ET.SubElement(
              worldbody, "geom",
              name="block_%d_%d" % (i, j),
              pos="%f %f %f" % (j * size_scaling - torso_x,
                                i * size_scaling - torso_y,
                                height_offset +
                                height / 2 * size_scaling),
              size="%f %f %f" % (0.5 * size_scaling,
                                 0.5 * size_scaling,
                                 height / 2 * size_scaling),
              type="box",
              material="",
              contype="1",
              conaffinity="1",
              rgba="0.4 0.4 0.4 1",
          )
        elif maze_env_utils.can_move(structure[i][j]):  # Movable block.
          # The "falling" blocks are shrunk slightly and increased in mass to
          # ensure that it can fall easily through a gap in the platform blocks.
          falling = maze_env_utils.can_move_z(structure[i][j])
          shrink = 0.99 if falling else 1.0
          moveable_body = ET.SubElement(
              worldbody, "body",
              name="moveable_%d_%d" % (i, j),
              pos="%f %f %f" % (j * size_scaling - torso_x,
                                i * size_scaling - torso_y,
                                height_offset +
                                height / 2 * size_scaling),
          )
          ET.SubElement(
              moveable_body, "geom",
              name="block_%d_%d" % (i, j),
              pos="0 0 0",
              size="%f %f %f" % (0.5 * size_scaling * shrink,
                                 0.5 * size_scaling * shrink,
                                 height / 2 * size_scaling),
              type="box",
              material="",
              mass="0.001" if falling else "0.0002",
              contype="1",
              conaffinity="1",
              rgba="0.9 0.1 0.1 1"
          )
          if maze_env_utils.can_move_x(structure[i][j]):
            ET.SubElement(
                moveable_body, "joint",
                armature="0",
                axis="1 0 0",
                damping="0.0",
                limited="true" if falling else "false",
                range="%f %f" % (-size_scaling, size_scaling),
                margin="0.01",
                name="moveable_x_%d_%d" % (i, j),
                pos="0 0 0",
                type="slide"
            )
          if maze_env_utils.can_move_y(structure[i][j]):
            ET.SubElement(
                moveable_body, "joint",
                armature="0",
                axis="0 1 0",
                damping="0.0",
                limited="true" if falling else "false",
                range="%f %f" % (-size_scaling, size_scaling),
                margin="0.01",
                name="moveable_y_%d_%d" % (i, j),
                pos="0 0 0",
                type="slide"
            )
          if maze_env_utils.can_move_z(structure[i][j]):
            ET.SubElement(
                moveable_body, "joint",
                armature="0",
                axis="0 0 1",
                damping="0.0",
                limited="true",
                range="%f 0" % (-height_offset),
                margin="0.01",
                name="moveable_z_%d_%d" % (i, j),
                pos="0 0 0",
                type="slide"
            )

    torso = tree.find(".//body[@name='torso']")
    geoms = torso.findall(".//geom")
    for geom in geoms:
      if 'name' not in geom.attrib:
        raise Exception("Every geom of the torso must have a name "
                        "defined")

    _, file_path = tempfile.mkstemp(text=True)
    tree.write(file_path)

    self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)

  def _get_obs(self):
    return np.concatenate([self.wrapped_env._get_obs(),
                           [self.t * 0.001]])

  def reset(self):
    self.t = 0
    self.wrapped_env.reset()
    return self._get_obs()

  @property
  def viewer(self):
    return self.wrapped_env.viewer

  def render(self, *args, **kwargs):
    return self.wrapped_env.render(*args, **kwargs)

  @property
  def observation_space(self):
    shape = self._get_obs().shape
    high = np.inf * np.ones(shape)
    low = -high
    return gym.spaces.Box(low, high)

  @property
  def action_space(self):
    return self.wrapped_env.action_space

  def _find_robot(self):
    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == 'r':
          return j * size_scaling, i * size_scaling
    assert False, 'No robot in maze specification.'

  def step(self, action):
    self.t += 1
    inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
    next_obs = self._get_obs()
    done = False
    return next_obs, inner_reward, done, info
