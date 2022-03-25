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

"""Cubify Operation"""
from typing import Tuple

import tensorflow as tf

# Vertices in the unit cube (x,y,z)
UNIT_CUBOID_VERTS = tf.constant(
    [
        [0, 0, 0], # left-top-front
        [0, 0, 1], # left-top-back
        [0, 1, 0], # left-bottom-front
        [0, 1, 1], # left-bottom-back
        [1, 0, 0], # right-top-front
        [1, 0, 1], # right-top-back
        [1, 1, 0], # right-bottom-front
        [1, 1, 1], # right-bottom-back
    ])

NUM_VERTS_PER_CUBOID = tf.shape(UNIT_CUBOID_VERTS)[0]

# Indices for vertices that make up each face in the unit cuboid
UNIT_CUBOID_FACES = tf.constant(
    [
        [0, 1, 2],
        [1, 3, 2],  # left face: 0, 1
        [2, 3, 6],
        [3, 7, 6],  # bottom face: 2, 3
        [0, 2, 6],
        [0, 6, 4],  # front face: 4, 5
        [0, 5, 1],
        [0, 4, 5],  # up face: 6, 7
        [6, 7, 5],
        [6, 5, 4],  # right face: 8, 9
        [1, 7, 3],
        [1, 5, 7],  # back face: 10, 11
    ])
NUM_FACES_PER_CUBOID = tf.shape(UNIT_CUBOID_FACES)[0]

def generate_3d_coords(x_max: int, y_max: int, z_max: int,
                       flatten_output: bool = False) -> tf.Tensor:
  """Generates a tensor containing cartesian coordinates.

  This function has two modes. By default, the returned tensor has shape
  [x_max+1, y_max+1, z_max+1, 3], where last dimension contains the x, y, and z
  values at the specified coordinate (i.e. coords[x, y, z] == [x, y, z]).
  If flatten_output is True, the output is reshaped to a 2D Tensor and the
  inner dimension is 3.

  Args:
    x_max: An `int`, specifies the maximum x value in the coordinate grid.
    y_max: An `int`, specifies the maximum y value in the coordinate grid.
    z_max: An `int`, specifies the maximum z value in the coordinate grid.
    flatten_output: A `bool`, whether to flatten the output to a 2D Tensor.

  Returns:
    coords: A `Tensor` that contains the coordinate values with shape
      [x_max+1, y_max+1, z_max+1, 3] when flatten_output is False and shape
      [(x_max+1)*(y_max+1)*(z_max+1), 3] when flatten_output is True. The values
      in the last dimension are the x, y, and z values at a coordinate location.
  """
  x = tf.range(0, x_max+1, dtype=tf.int32)
  y = tf.range(0, y_max+1, dtype=tf.int32)
  z = tf.range(0, z_max+1, dtype=tf.int32)

  x_coords, y_coords, z_coords = tf.meshgrid(x, y, z, indexing='ij')
  coords = tf.stack([x_coords, y_coords, z_coords], axis=-1)

  if flatten_output:
    coords = tf.reshape(coords, shape=[-1, 3])

  return coords

def cantor_encode_3d_coordinates(x: tf.Tensor)-> tf.Tensor:
  """Performs Cantor pairing to encode 3D coordinates into scalar values.

  This function applies the Cantor Pairing Function
  (https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function) in
  order to generate a unique scalar value for a coordinate triplet.

  Args:
    x: A `Tensor` of shape [N, 3] where the last dimension corresponds to the
      (x, y, z) coordinates to encode into a single value.

  Returns:
    hashed: A `Tensor` of shape [N, 1], containing the unique encoding for the
      coordinate.
  """

  # Extract the coordinates
  x = tf.cast(x, tf.float32)
  k1 = x[:, 0]
  k2 = x[:, 1]
  k3 = x[:, 2]

  # Apply two rounds of Cantor's Pairing Function
  hash_1 = (k1 + k2) * (k1 + k2 + 1) / 2.0 + k2
  final_hash = (hash_1 + k3) * (hash_1 + k3 + 1) / 2.0 + k3

  return tf.cast(final_hash, tf.int32)

def initialize_mesh(grid_dims: int, align: str) -> Tuple[tf.Tensor, tf.Tensor]:
  """Initializes the vertices and faces for a complete cuboid mesh.

  This function generates 2 rank 2 Tensors, one for vertices and another for
  faces. The vertices are the initial (x,y,z) coordinates of each vertex
  in the mesh normalized between -1 and 1. The faces are the indices of the 3
  vertices that define the face. Note that there may be duplicate faces.

  Args:
    grid_dims: An `int`, specifies the height, width, and depth of the mesh.
    align: A `str`, one of 'topleft', 'corner', or 'center' that defines the
      desired alignment of the mesh vertices.

  Returns:
    verts: A `Tensor` of shape [num_verts, 3], where
      num_verts = (grid_dim+1)**3.
    faces: A `Tensor` of shape [num_faces, 3], where
      num_faces = 12*(grid_dim)**3.
  """
  num_cuboids = (grid_dims) ** 3

  # Generate the vertex locations
  coords = generate_3d_coords(grid_dims, grid_dims, grid_dims, True)
  coords = tf.cast(coords, tf.float32)

  # Apply alignment and normalize verts
  if align == 'center':
    coords -= 0.5

  margin = 0.0 if align == 'corner' else 1.0

  if grid_dims != 1:
    verts = coords * 2.0 / (tf.cast(grid_dims, tf.float32) - margin) - 1.0
  else:
    verts = coords * 2.0 - 1.0

  # Generate offsets for the unit cube verts for the grid
  offsets = generate_3d_coords(grid_dims-1, grid_dims-1, grid_dims-1)
  offsets = tf.expand_dims(offsets, axis=-2)

  # Create unit cubes for each cuboid
  cuboid_verts = tf.tile(UNIT_CUBOID_VERTS, multiples=[num_cuboids, 1])
  cuboid_verts = tf.reshape(
      cuboid_verts,
      shape=[grid_dims, grid_dims, grid_dims, NUM_VERTS_PER_CUBOID, 3])

  # Add the offsets so that each cube in the grid has the correct vertices
  cuboid_verts += offsets

  # Map the vertices of each cuboid in the grid to the predefined coords
  cuboid_verts = tf.reshape(cuboid_verts, shape=[-1, 3])

  # Convert to scalar values to save memory when doing tf.equal
  coords_hashed = cantor_encode_3d_coordinates(coords)
  cuboid_verts_hashed = cantor_encode_3d_coordinates(cuboid_verts)

  cuboid_verts_hashed = tf.reshape(cuboid_verts_hashed, shape=[-1, 1, 1])
  mask = tf.equal(coords_hashed, cuboid_verts_hashed)
  cuboid_verts = tf.where(mask)[:, -1]

  # cuboid_verts is a tensor with shape [num_cuboids, 8], where each entry
  # contains the indices of the 8 vertices in verts that belong to it
  # The ordering of the vertices are:
  # 0. left-up-front
  # 1. left-up-back
  # 2. left-down-front
  # 3. left-down-back
  # 4. right-up-front
  # 5. right-up-back
  # 6. right-down-front
  # 7. right-down-back
  cuboid_verts = tf.reshape(
      cuboid_verts, shape=[num_cuboids, NUM_VERTS_PER_CUBOID])

  # Map each cuboids face's vertex indices to match the actual index of the
  # vertex in cuboid_verts
  cuboid_range = tf.expand_dims(
      tf.repeat(tf.range(num_cuboids), [NUM_FACES_PER_CUBOID]), -1)
  cuboid_face_indices = tf.tile(UNIT_CUBOID_FACES, multiples=[num_cuboids, 1])
  cuboid_face_indices = tf.stack(
      [tf.concat(
          [cuboid_range, tf.expand_dims(cuboid_face_indices[:, 0], -1)], 1),
       tf.concat(
           [cuboid_range, tf.expand_dims(cuboid_face_indices[:, 1], -1)], 1),
       tf.concat(
           [cuboid_range, tf.expand_dims(cuboid_face_indices[:, 2], -1)], 1)],
      axis=1)
  faces = tf.gather_nd(cuboid_verts, cuboid_face_indices)

  return verts, faces

def generate_face_bounds(voxels: tf.Tensor, axis: int):
  """Generates `Tensors` that give masks for boundaries of the voxels.

  This function returns 2 `Tensors`, an upper boundary and a lower boundary, for
  the voxel occupancy grid that indicate whether or not the voxel is considered
  an external surface in the mesh. This is done using 3D convolutions to find
  the adjacent voxels that share a face, and masking them out. The direction
  of the lower boundary is left, top, or front for axes 1, 2, 3 respectively.
  Likewise, the direction of the upper boundary is right, bottom, back.

  Args:
    voxels: A `Tensor` of shape [B, D, H, W] that contains the thresholded
      voxel occupancies.
    axis: An `int` that indicates the axis of interest. Either 1 for the
      z-axis, 2 for the y-axis, or 3 for the x-axis.

  Returns:
    upper: A `Tensor` of shape [B, D, H, W] containing 1s and 0s that indicate
      whether a given voxel has a face present (either right, bottom, or back
      depending on the axis).
    lower: A `Tensor` of shape [B, D, H, W] containing 1s and 0s that indicate
      whether a given voxel has a face present (either left, top, or front
      depending on the axis).
  """
  shape = tf.shape(voxels)
  batch_size, depth, height, width = shape[0], shape[1], shape[2], shape[3]

  # Prepare input for 3D convolution
  conv_input = tf.expand_dims(voxels, axis=-1)

  # Generate the weights depending on the axis of interest
  if axis == 3:
    kernel_shape = [1, 1, 2, 1, 1]
  elif axis == 2:
    kernel_shape = [1, 2, 1, 1, 1]
  elif axis == 1:
    kernel_shape = [2, 1, 1, 1, 1]

  kernel_weights = tf.constant(value=0.5, shape=kernel_shape)

  # Create the mask that finds shared faces
  adj_mask = tf.nn.conv3d(conv_input, kernel_weights, [1, 1, 1, 1, 1], 'VALID')

  # Since the weights values are 0.5 and the voxels occupancy values are either
  # 0 or 1, any occupied voxel that has an adjacent occupied voxel will be
  # identified if the mask value is 1.0 (0.5*1 + 0.5*1)
  adj_mask = tf.cast(adj_mask > 0.5, adj_mask.dtype)

  if axis == 3:
    width -= 1
    concat_shape = [batch_size, depth, height, 1]
  elif axis == 2:
    height -= 1
    concat_shape = [batch_size, depth, 1, width]
  elif axis == 1:
    depth -= 1
    concat_shape = [batch_size, 1, height, width]

  adj_mask = tf.reshape(adj_mask, shape=[batch_size, depth, height, width])

  # Generate the lower and upper tensors [B, D, H, W] that indicate if the
  # voxel does not share an adjacent face in the axis direction. The inverse
  # of the mask is inserted into a tensor of ones to account for voxels that are
  # on the edge of the occupancy grid, as those faces are not captured by the
  # 3D convolution. Later, any unoccupied voxels along the edge of the grid will
  # be zeroed out
  lower = tf.concat(
      [tf.ones(concat_shape),
       1-adj_mask], axis)
  upper = tf.concat(
      [1-adj_mask,
       tf.ones(concat_shape)], axis)

  # Zero out any unoccupied voxels
  lower *= voxels
  upper *= voxels

  return lower, upper

def cubify(voxels: tf.Tensor,
           thresh: float,
           align: str = 'topleft'):
  """Converts a voxel occupancy grid into a mesh.

  Args:
    voxels: A `Tensor` of shape [B, D, H, W] that contains the voxel occupancy
      prediction. D, H, and W must be equal.
    thresh: A `float` that specifies the threshold value of a valid occupied
      voxel.
    align: A `str`, one of 'topleft', 'corner', or 'center' that defines the
      alignment of the mesh vertices. Currently only 'topleft' is supported.

  Returns:
    mesh: A dictinary with the following keys:
      'verts': A `Tensor` of shape [B, num_verts, 3], where the last dimension
        contains all (x,y,z) vertex coordinates in the initial mesh mesh.
      'verts_mask': A `Tensor` of shape [B, num_verts], a mask for valid
        vertices in the watertight mesh.
      'faces': A `Tensor` of shape [B, num_faces, 3], where the last dimension
        contain the verts indices that make up the face. This may include
        duplicate faces.
      'faces_mask': A `Tensor` of shape [B, num_faces], a mask for valid faces
        in the watertight mesh.
  """
  shape = tf.shape(voxels)
  batch_size, depth, _, _ = shape[0], shape[1], shape[2], shape[3]

  # Threshold the voxel occupancy prediction
  voxels = tf.cast(voxels > thresh, voxels.dtype)

  # Determine the non-adjacent faces in the final mesh
  z_front_updates, z_back_updates = generate_face_bounds(voxels, axis=1)
  y_top_updates, y_bot_updates = generate_face_bounds(voxels, axis=2)
  x_left_updates, x_right_updates = generate_face_bounds(voxels, axis=3)

  updates = [
      x_left_updates,
      x_left_updates,

      y_bot_updates,
      y_bot_updates,

      z_front_updates,
      z_front_updates,

      y_top_updates,
      y_top_updates,

      x_right_updates,
      x_right_updates,

      z_back_updates,
      z_back_updates]

  updates = tf.cast(updates, tf.int32)
  faces_idx = tf.stack(updates, axis=0)

  # faces_idx cuboid ordering: left-to-right, top-to-bot, front-to-back
  # because this will be used to generate the mask to mark valid faces. The
  # UNIT_CUBOID_FACES are defined in the same orientation
  faces_idx = tf.transpose(faces_idx, perm=[1, 4, 3, 2, 0])
  faces_idx = tf.reshape(faces_idx, [batch_size, -1, 12])

  # Boolean mask for valid faces
  faces_mask = tf.reshape(faces_idx, shape=[batch_size, -1])

  grid_dim = depth
  # All verts and faces in the mesh
  verts, faces = initialize_mesh(grid_dim, align)

  # Batch the initial verts and faces
  verts = tf.expand_dims(verts, axis=0)
  faces = tf.expand_dims(faces, axis=0)
  verts = tf.tile(verts, multiples=[batch_size, 1, 1])
  faces = tf.tile(faces, multiples=[batch_size, 1, 1])

  # Offset the faces so that each vertex index is greater than 0
  masked_faces = faces + 1

  # Zero out unused faces
  masked_faces = masked_faces * tf.cast(
      tf.expand_dims(faces_mask, -1), faces.dtype)

  # Create a mask based on whether or not a vertex appeared in any of the faces
  num_verts = (depth+1)**3
  verts_mask = tf.math.bincount(
      tf.cast(tf.reshape(masked_faces, [batch_size, -1]), tf.int32),
      minlength=num_verts+1, maxlength=num_verts+1, binary_output=True, axis=-1)

  # The first index was used for any 0s (masked out faces), which can be ignored
  verts_mask = verts_mask[:, 1:]

  mesh = {
      'verts': verts,
      'faces': faces,
      'verts_mask': verts_mask,
      'faces_mask': faces_mask
  }

  return mesh
