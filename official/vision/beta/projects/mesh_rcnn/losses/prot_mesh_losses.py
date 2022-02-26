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

"""Mesh Losses for Mesh R-CNN.

This is still very much in the 'prototyping' phase. The only reason it was
pushed is to allow the evaluation team to reference my current approach
as they start writing the eval implementation.

TODOs
* Move losses into classes and move free functions where applicable.
* Write tests
  1. Hardcoded pointclouds/normals and hand calculate loss values.
  2. Differential test with PyTorch3D losses.
"""

from typing import List, Tuple, Union

import tensorflow as tf

###### Temporary for testing edge loss ######
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import (
    compute_edges, get_verts_from_indices)
#############################################


def compute_square_distances(
    pointcloud_a: tf.Tensor,
    pointcloud_b: tf.Tensor,
) -> tf.Tensor:
  """TODO"""
  # FloatTensor[B, num_points_a, num_points_b, 3] where the vector given by
  # entry [b, i, j, :] is the vector 
  # `(pointcloud_a[b, i, :] - pointcloud_b[b, j, :])`.
  difference = (tf.expand_dims(pointcloud_a, axis=-2) -
                tf.expand_dims(pointcloud_b, axis=-3))

  # FloatTensor[B, num_points_a, num_points_b] where the value given by entry
  # [b, i, j] is the squared l2 norm of the difference between the vectors
  # `pointcloud_a[b, i, :]` and `pointcloud_b[b, j, :]`
  square_distances = tf.norm(difference, ord=2, axis=-1) ** 2

  return square_distances

def get_normals_nearest_neighbors(
    pointcloud_a: tf.Tensor,
    pointcloud_b: tf.Tensor,
    normals_a: tf.Tensor,
    normals_b: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """TODO"""
  shape = tf.shape(pointcloud_a)
  batch_size, num_points_a, _ = shape[0], shape[1], shape[2]
  num_points_b = tf.shape(pointcloud_b)[1]

  square_distances = compute_square_distances(pointcloud_a, pointcloud_b)

  # IntTensor[B, num_points_a] where the element i of the vector holds the value j
  # such that point `pointcloud_a[b, i, :]`'s nearest neightbor is
  # `pointcloud_b[b, j, :]`.
  a_nearest_neighbors_in_b = tf.argmin(
      square_distances, axis=-1, output_type=tf.int32
  )
  # IntTensor[B, num_points_b] where the element i of the vector holds the value j
  # such that point `pointcloud_b[b, i, :]`'s nearest neightbor is
  # `pointcloud_a[b, j, :]`.
  b_nearest_neighbors_in_a = tf.argmin(
      square_distances, axis=-2, output_type=tf.int32
  )

  # IntTensor[B, 1, 1] where the element in each batch is the batch index.
  batch_ind = tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=-1), axis=-1)
  # IntTensor[B, num_points_a, 1] where the single element in the rows for
  # each batch is the batch idx.
  batch_ind_a = tf.repeat(batch_ind, num_points_a, axis=1)
  # IntTensor[B, num_points_b, 1] 
  batch_ind_b = tf.repeat(batch_ind, num_points_b, axis=1)

  a_nearest_neighbors_in_b_ind = tf.concat(
      [batch_ind_a, tf.expand_dims(a_nearest_neighbors_in_b, -1)], -1
  )
  b_nearest_neighbors_in_a_ind = tf.concat(
      [batch_ind_b, tf.expand_dims(b_nearest_neighbors_in_a, -1)], -1
  )

  # FloatTensor[B, num_points_b, 3]: The normals corresponding to the re-ordered
  # points from `normals_a` that are the closest to the points in `normals_b`.
  normals_a_nearest_to_b = tf.gather_nd(normals_a, b_nearest_neighbors_in_a_ind)
  # FloatTensor[B, num_points_a, 3]: The normals corresponding to the re-ordered
  # points from `normals_b` that are the closest to the points in `normals_a`.
  normals_b_nearest_to_a = tf.gather_nd(normals_b, a_nearest_neighbors_in_b_ind)

  return normals_a_nearest_to_b, normals_b_nearest_to_a

def cosine_similarity(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  """TODO"""
  a_norm = tf.linalg.l2_normalize(a, axis=-1)
  b_norm = tf.linalg.l2_normalize(b, axis=-1)
  return tf.reduce_sum(a_norm * b_norm, axis=-1)

def main():
  ######################## Chamfer + Normal Setup #########################
  in_pointcloud_a = [[
      [0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 1, 0]
  ]]
  in_pointcloud_b = [[
      [0.8, 0, 0], [0, 0.8, 0], [-0.8, 0.8, 0], [0, 0.2, 0], [0, 0, 0]
  ]]
  in_normals_a = [[
      [1, 0, 0], [0, 0.1, 1], [1, 0, 0], [-1, 0, 0]
  ]]
  in_normals_b = [[
      [0, 0, 1], [1, 0.1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]
  ]]

  in_pointcloud_a = tf.convert_to_tensor(in_pointcloud_a, dtype=tf.float32)
  in_pointcloud_b = tf.convert_to_tensor(in_pointcloud_b, dtype=tf.float32)
  in_normals_a = tf.convert_to_tensor(in_normals_a, dtype=tf.float32)
  in_normals_b = tf.convert_to_tensor(in_normals_b, dtype=tf.float32)

  batch_size = 1
  num_points_a = 4
  num_points_b = 5

  batch_reduction = None
  #########################################################################

  def add_pointcloud_distances(dist_a, 
                               dist_b, 
                               num_points_a, 
                               num_points_b, 
                               batch_size, 
                               batch_reduction: Union[str, None] = "mean"):
    # Point reduction (sum distances for each point in pointcloud) and
    # normalize based on number of points in pointcloud.
    dist_a = tf.reduce_sum(dist_a, axis=-1) / num_points_a
    dist_b = tf.reduce_sum(dist_b, axis=-1) / num_points_b

    # Batch reduction.
    if batch_reduction is not None:
      dist_a = tf.reduce_sum(dist_a)
      dist_b = tf.reduce_sum(dist_b)
      if batch_reduction == "mean":
        dist_a /= batch_size
        dist_b /= batch_size

    return dist_a + dist_b

  ################################ Chamfer ################################
  square_distances = compute_square_distances(in_pointcloud_a, in_pointcloud_b)

  # FloatTensor[B, num_points_a] representing the minimum of the squared
  # distance from each point in pointcloud a to each of the points in
  # pointcloud b.
  min_square_dist_a_to_b = tf.reduce_min(input_tensor=square_distances,
                                         axis=-1)
  # FloatTensor[B, num_points_b]
  min_square_dist_b_to_a = tf.reduce_min(input_tensor=square_distances,
                                         axis=-2)

  chamfer_dist = add_pointcloud_distances(
      min_square_dist_a_to_b, min_square_dist_b_to_a, num_points_a, 
      num_points_b, batch_size, batch_reduction="mean"
  )
  tf.print("\nchamfer_loss:")
  tf.print(chamfer_dist)
  #########################################################################

  ################################ Normal #################################
  normals_a_nearest_to_b, normals_b_nearest_to_a = \
    get_normals_nearest_neighbors(in_pointcloud_a, in_pointcloud_b,
                                  in_normals_a, in_normals_b)

  # FloatTensor[B, num_points_a] representing the absolute normal distance
  # between the normals in pointcloud a and the normals of the closest points
  # in pointcloud b.
  abs_normal_dist_a = 1 - tf.math.abs(
      cosine_similarity(normals_b_nearest_to_a, in_normals_a)
  )
  # FloatTensor[B, num_points_b]
  abs_normal_dist_b = 1 - tf.math.abs(
      cosine_similarity(normals_a_nearest_to_b, in_normals_b)
  )
  
  normal_dist = add_pointcloud_distances(
      abs_normal_dist_a, abs_normal_dist_b, num_points_a,
      num_points_b, batch_size, batch_reduction="mean"
  )
  tf.print("\nnormal_loss:")
  tf.print(normal_dist)
  #########################################################################

  ################################# Edge ##################################
  def create_voxels(
      grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
  ) -> tf.Tensor:
    ones = tf.ones(shape=[len(occupancy_locs)])
    voxels = tf.scatter_nd(
        indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
        updates=ones,
        shape=[batch_size, grid_dims, grid_dims, grid_dims],
    )
    return voxels

  def get_verts_and_faces(
      grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)

    verts = tf.cast(verts, tf.float32)
    faces = tf.cast(faces, tf.int32)
    verts_mask = tf.cast(verts_mask, tf.int8)
    faces_mask = tf.cast(faces_mask, tf.int8)

    return verts, faces, verts_mask, faces_mask

  grid_dims = 2
  batch_size = 5
  occupancy_locs = [
      [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
      [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
      [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
      [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
  ]
  verts, faces, verts_mask, faces_mask = get_verts_and_faces(
      grid_dims, batch_size, occupancy_locs
  )
  edges, edges_mask = compute_edges(faces, faces_mask)

  def edge_loss(verts, verts_mask, edges, edges_mask) -> tf.Tensor:
    shape = tf.shape(edges_mask)
    batch_size, num_edges = tf.cast(shape[0], tf.float32), shape[1]

    num_valid_edges = tf.reduce_sum(edges_mask, axis=-1, keepdims=True)
    num_valid_edges = tf.where(tf.equal(num_valid_edges, 0),
                               tf.ones_like(num_valid_edges),
                               num_valid_edges)
    weights = 1 / tf.repeat(num_valid_edges, num_edges, axis=-1)

    v0, v1 = get_verts_from_indices(
        verts, verts_mask, edges, edges_mask, num_inds_per_set=2
    )
    sqr_l2_per_edge = tf.norm(v1 - v0, axis=-1) ** 2.0
    norm_sqr_l2_per_edge = sqr_l2_per_edge * weights
    return tf.reduce_sum(norm_sqr_l2_per_edge) / batch_size

  edge_loss = edge_loss(verts, verts_mask, edges, edges_mask)
  tf.print("\nedge loss: ")
  tf.print(edge_loss)

  #########################################################################


if __name__ == '__main__':
  main()
