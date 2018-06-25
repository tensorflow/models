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

"""Adapted from rllab maze_env_utils.py."""
import numpy as np
import math


class Move(object):
  X = 11
  Y = 12
  Z = 13
  XY = 14
  XZ = 15
  YZ = 16
  XYZ = 17


def can_move_x(movable):
  return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ]


def can_move_y(movable):
  return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ]


def can_move_z(movable):
  return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]


def can_move(movable):
  return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)


def construct_maze(maze_id='Maze'):
  if maze_id == 'Maze':
    structure = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
  elif maze_id == 'Push':
    structure = [
        [1, 1,  1,  1,   1],
        [1, 0, 'r', 1,   1],
        [1, 0,  Move.XY, 0,  1],
        [1, 1,  0,  1,   1],
        [1, 1,  1,  1,   1],
    ]
  elif maze_id == 'Fall':
    structure = [
        [1, 1,   1,  1],
        [1, 'r', 0,  1],
        [1, 0,   Move.YZ,  1],
        [1, -1, -1,  1],
        [1, 0,   0,  1],
        [1, 1,   1,  1],
    ]
  else:
      raise NotImplementedError('The provided MazeId %s is not recognized' % maze_id)

  return structure
