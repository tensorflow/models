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

"""A module with utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def trajectory_to_deltas(trajectory, state):
  """Computes a sequence of deltas of a state to traverse a trajectory in 2D.

  The initial state of the agent contains its pose -- location in 2D and
  orientation. When the computed deltas are incrementally added to it, it
  traverses the specified trajectory while keeping its orientation parallel to
  the trajectory.

  Args:
    trajectory: a np.array of floats of shape n x 2. The n-th row contains the
      n-th point.
    state: a 3 element np.array of floats containing agent's location and
      orientation in radians.

  Returns:
    A np.array of floats of size n x 3.
  """
  state = np.reshape(state, [-1])
  init_xy = state[0:2]
  init_theta = state[2]

  delta_xy = trajectory - np.concatenate(
      [np.reshape(init_xy, [1, 2]), trajectory[:-1, :]], axis=0)

  thetas = np.reshape(np.arctan2(delta_xy[:, 1], delta_xy[:, 0]), [-1, 1])
  thetas = np.concatenate([np.reshape(init_theta, [1, 1]), thetas], axis=0)
  delta_thetas = thetas[1:] - thetas[:-1]

  deltas = np.concatenate([delta_xy, delta_thetas], axis=1)
  return deltas
