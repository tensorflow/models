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

from typing import List, Tuple

import tensorflow as tf

# Number of dimensions in coordinate system used.
COORD_DIM = 3


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


def get_face_vertices(
  verts: tf.Tensor,
  verts_mask: tf.Tensor,
  faces: tf.Tensor,
  faces_mask: tf.Tensor,
  batch_size: int,
  num_faces: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Extracts three `Tensors` corresponding to the vertices for each face.

  Args:
    verts: A float `Tensor` of shape [B, Nv, 3], where the last dimension
      contains all (x,y,z) vertex coordinates in the initial mesh.
    verts_mask: An int `Tensor` of shape [B, Nv] representing a mask for
      valid vertices in the watertight mesh.
    faces: An int `Tensor` of shape [B, Nf, 3], where the last dimension
      contains the verts indices that make up the face. This may include
      duplicate faces.
    faces_mask: An int `Tensor` of shape [B, Nf], representing a mask for
      valid faces in the watertight mesh.
    batch_size: `int`, specifies the number of batch elements.
    num_faces: `int`, specifies the number of faces in each mesh.

  Returns:
    v0, v1, v2: A tuple of three float `Tensor`s each of shape [B, Nf, 3]
      holding the ith (0, 1, or 2) vertex for each face of the mesh.
  """
  # Zero out unused vertices and faces
  masked_verts = verts * tf.cast(tf.expand_dims(verts_mask, -1), verts.dtype)
  masked_faces = faces * tf.cast(tf.expand_dims(faces_mask, -1), faces.dtype)

  # IntTensor[B, Nf, 1] where the single element in the rows for each batch is
  # the batch idx.
  batch_ind = tf.repeat(
    tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=-1), axis=-1),
    num_faces,
    axis=1,
  )

  face_verts = [None] * COORD_DIM
  for i in range(COORD_DIM):
    # IntTensor[B, Nf, 1] where the single element in each row is the ith
    # (0, 1, or 2) vertex index from `faces`.
    faces_vert_ind = tf.transpose(
      tf.expand_dims(masked_faces[:, :, i], axis=0), perm=[1, 2, 0]
    )
    # IntTensor[B, Nf, 2]: Concatenated tensor used to index into `verts`.
    vert_ind = tf.concat([batch_ind, faces_vert_ind], -1)
    # FloatTensor[B, Nf, 3]: The ith vertex for each face of the mesh.
    face_verts[i] = tf.gather_nd(masked_verts, vert_ind)

  v0, v1, v2 = face_verts
  return v0, v1, v2
