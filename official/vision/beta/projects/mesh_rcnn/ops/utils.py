# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Common utility functions used across Mesh R-CNN ops."""

from typing import List

import tensorflow as tf


def create_voxels(
    grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
) -> tf.Tensor:
  """Creates a `Tensor` representing a batch of voxel grids.

  Args:
    grid_dims: An `int` representing the voxel resolution (or number of grid
      units for depth, width, and height) in the voxel grid.
    batch_size: An `int` representing the number of batch elements.
    occupancy_locs: A nest list of `int`s that indicate which voxels should be
      be occupied within the grid. The format is: [[b, z, y, x], ...] where
      `(x,y,z)` represents the coordinate of an occupied voxel in the grid for
      batch element `b`.

  Returns:
    voxels: An int `Tensor` of shape [B, D, H, W] that contains the voxel
      occupancy prediction. D, H, and W are equal.
  """
  ones = tf.ones(shape=[len(occupancy_locs)])
  voxels = tf.scatter_nd(
      indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
      updates=ones,
      shape=[batch_size, grid_dims, grid_dims, grid_dims],
  )

  return voxels
