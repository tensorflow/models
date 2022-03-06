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

"""Mesh Losses for Mesh R-CNN."""

from typing import Optional, Tuple, Union

import tensorflow as tf

from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import (
    MeshSampler, get_verts_from_indices)


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
  batch_size = tf.shape(pointcloud_a)[0]
  num_points_a = tf.shape(pointcloud_a)[1]
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
  batch_ind = tf.expand_dims(
      tf.expand_dims(tf.range(batch_size), axis=-1), axis=-1
  )
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

def add_pointcloud_distances(dist_a: tf.Tensor,
                             dist_b: tf.Tensor,
                             num_points_a: tf.Tensor,
                             num_points_b: tf.Tensor,
                             batch_size: tf.Tensor,
                             weights: Optional[tf.Tensor],
                             batch_reduction:
                             Union[str, None] = "mean") -> tf.Tensor:
  """TODO"""
  # Cast int32 tensors to float32.
  num_points_a = tf.cast(num_points_a, tf.float32)
  num_points_b = tf.cast(num_points_b, tf.float32)
  batch_size = tf.cast(batch_size, tf.float32)

  # Point reduction: sum all dists into one element per batch.
  dist_a = tf.reduce_sum(dist_a, axis=-1) / num_points_a
  dist_b = tf.reduce_sum(dist_b, axis=-1) / num_points_b

  # Apply batch weights.
  if weights is not None:
    dist_a *= weights
    dist_b *= weights

  # Normalize with number of points in pointcloud.
  dist_a /= num_points_a
  dist_b /= num_points_b

  # Batch reduction.
  if batch_reduction is not None:
    dist_a = tf.reduce_sum(dist_a)
    dist_b = tf.reduce_sum(dist_b)
    if batch_reduction == "mean":
      div = tf.reduce_sum(weights) if weights is not None else batch_size
      dist_a /= div
      dist_b /= div

  return dist_a + dist_b


def chamfer_loss(pointcloud_a: tf.Tensor,
                 pointcloud_b: tf.Tensor,
                 weights: Optional[tf.Tensor] = None,
                 batch_reduction: Union[str, None] = "mean") -> tf.Tensor:
  """TODO"""
  square_distances = compute_square_distances(pointcloud_a, pointcloud_b)

  # FloatTensor[B, num_points_a] representing the minimum of the squared
  # distance from each point in pointcloud a to each of the points in
  # pointcloud b.
  min_square_dist_a_to_b = tf.reduce_min(input_tensor=square_distances,
                                         axis=-1)
  # FloatTensor[B, num_points_b]
  min_square_dist_b_to_a = tf.reduce_min(input_tensor=square_distances,
                                         axis=-2)

  batch_size = tf.shape(pointcloud_a)[0]
  num_points_a = tf.shape(pointcloud_a)[1]
  num_points_b = tf.shape(pointcloud_b)[1]

  chamfer_dist = add_pointcloud_distances(
      min_square_dist_a_to_b, min_square_dist_b_to_a,
      num_points_a, num_points_b, batch_size, weights, batch_reduction
  )
  return chamfer_dist

def normal_loss(pointcloud_a: tf.Tensor,
                pointcloud_b: tf.Tensor,
                normals_a: tf.Tensor,
                normals_b: tf.Tensor,
                weights: Optional[tf.Tensor] = None,
                batch_reduction: Union[str, None] = "mean") -> tf.Tensor:
  """TODO"""
  normals_a_nearest_to_b, normals_b_nearest_to_a = \
    get_normals_nearest_neighbors(pointcloud_a, pointcloud_b,
                                  normals_a, normals_b)

  # FloatTensor[B, num_points_a] representing the absolute normal distance
  # between the normals in pointcloud a and the normals of the closest points
  # in pointcloud b.
  abs_normal_dist_a = 1 - tf.math.abs(
      cosine_similarity(normals_b_nearest_to_a, normals_a)
  )
  # FloatTensor[B, num_points_b]
  abs_normal_dist_b = 1 - tf.math.abs(
      cosine_similarity(normals_a_nearest_to_b, normals_b)
  )

  batch_size = tf.shape(pointcloud_a)[0]
  num_points_a = tf.shape(pointcloud_a)[1]
  num_points_b = tf.shape(pointcloud_b)[1]

  normal_dist = add_pointcloud_distances(
      abs_normal_dist_a, abs_normal_dist_b, num_points_a, num_points_b,
      batch_size, weights, batch_reduction
  )
  return normal_dist

def edge_loss(verts, verts_mask, edges, edges_mask) -> tf.Tensor:
  """TODO"""
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

  loss = tf.reduce_sum(norm_sqr_l2_per_edge) / batch_size
  return loss

class MeshLoss(tf.keras.losses.Loss):
  """Mesh R-CNN losses for the penalizing the predicted voxels and meshes."""

  def __init__(self,
               voxel_weight=0.0,
               chamfer_weight=1.0,
               normal_weight=0.0,
               edge_weight=0.1,
               true_num_samples=5000,
               pred_num_samples=5000) -> None:
    """TODO"""
    self._voxel_weight = voxel_weight
    self._chamfer_weight = chamfer_weight
    self._normal_weight = normal_weight
    self._edge_weight = edge_weight
    self._true_num_samples = true_num_samples
    self._pred_num_samples = pred_num_samples
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

  def call(self,
           voxels_true: tf.Tensor,
           voxels_pred: tf.Tensor,
           verts: tf.Tensor,
           verts_mask_true: tf.Tensor,
           verts_mask_pred: tf.Tensor,
           faces: tf.Tensor,
           faces_mask_true: tf.Tensor,
           faces_mask_pred: tf.Tensor,
           edges: tf.Tensor,
           edges_mask_pred: tf.Tensor) -> tf.Tensor:
    """TODO"""
    voxel_loss = 0
    if self._voxel_weight > 0:
      voxel_loss = self._voxel_weight * self._binary_crossentropy(
          voxels_true, voxels_pred
      )

    true_sampler = MeshSampler(self._true_num_samples)
    pred_sampler = MeshSampler(self._pred_num_samples)
    pointcloud_true, normals_true, _ = true_sampler.sample_meshes(
        verts, verts_mask_true, faces, faces_mask_true,
    )
    pointcloud_pred, normals_pred, _ = pred_sampler.sample_meshes(
        verts, verts_mask_pred, faces, faces_mask_pred,
    )

    chamfer_loss_ = self._chamfer_weight * chamfer_loss(
        pointcloud_true, pointcloud_pred
    )
    normal_loss_ = self._normal_weight * normal_loss(
        pointcloud_true, pointcloud_pred, normals_true, normals_pred
    )
    edge_loss_ = self._edge_weight * edge_loss(
        verts, verts_mask_pred, edges, edges_mask_pred
    )

    total_loss = voxel_loss + chamfer_loss_ + normal_loss_ + edge_loss_
    return total_loss, voxel_loss, chamfer_loss_, normal_loss_, edge_loss_

  def get_config(self) -> dict:
    config = {
        'chamfer_weight': self._chamfer_weight,
        'normal_weight': self._normal_weight,
        'edge_weight': self._edge_weight,
        'voxel_weight': self._voxel_weight,
        'true_num_samples': self._true_num_samples,
        'pred_num_samples': self._pred_num_samples,
    }
    base_config = super(MeshLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
