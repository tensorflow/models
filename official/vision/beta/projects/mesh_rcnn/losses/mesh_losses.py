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


def _compute_square_distances(
    pointcloud_a: tf.Tensor,
    pointcloud_b: tf.Tensor,
) -> tf.Tensor:
  """Compute the squared distance between each point in the input pointclouds.

  Args:
    pointcloud_a: A float `Tensor` of shape [B, num_points_a, 3] holding the
      coordinates of samples points from a batch of meshes. The pointcloud
      for a mesh will be 0 (i.e. pointcloud_a[i, :, :] = 0) if the mesh is
      empty (i.e. verts_mask[i,:] = 0).
    pointcloud_b: A float `Tensor` of shape [B, num_points_b, 3] holding the
      coordinates of samples points from a batch of meshes.

  Returns:
    square_distances: A float `Tensor` of shape [B, num_points_a, num_points_b]
      where the value given by entry [b, i, j] is the squared l2 norm of the
      difference between the vectors `pointcloud_a[b, i, :]` and
      `pointcloud_b[b, j, :]`.
  """
  # FloatTensor[B, num_points_a, num_points_b, 3] where the vector given by
  # entry [b, i, j, :] is the vector
  # `(pointcloud_a[b, i, :] - pointcloud_b[b, j, :])`.
  difference = (tf.expand_dims(pointcloud_a, axis=-2) -
                tf.expand_dims(pointcloud_b, axis=-3))

  return tf.norm(difference, ord=2, axis=-1) ** 2

def _get_features_nearest_neighbors(
    pointcloud_a: tf.Tensor,
    pointcloud_b: tf.Tensor,
    feats_a: tf.Tensor,
    feats_b: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Get two feature `Tensor`s of the nearest neighbors between each pointcloud.

  `feats_a` and `feats_b` can be various features of the points in the
  pointclouds 'a' and 'b'. Here are two example use cases for this function:

  1. If the Tensor passed in for `pointcloud_x` is also the Tensor passed in
  for `feats_x` (x signifying this is done for both pointclouds a and b), then
  the Tensors returned will be the points in `a` nearest to those in `b` and
  the points in `b` nearest to those in `a`. Put simply, this returns the
  nearest neightbors of each pointcloud to the points in the other pointcloud.

  2. If the Tensors passed in for `feats_x` are the normal vectors of the
  corresponding points in `pointcloud_x`, then the Tensors returned will
  be the normals corresponding to the re-ordered points from `normals_a` that
  are the closest to the points in `normals_b` and vice versa.


  Args:
    pointcloud_a: A float `Tensor` of shape [B, num_points_a, 3] holding the
      coordinates of samples points from a batch of meshes. The pointcloud
      for a mesh will be 0 (i.e. pointcloud_a[i, :, :] = 0) if the mesh is
      empty (i.e. verts_mask[i,:] = 0).
    pointcloud_b: A float `Tensor` of shape [B, num_points_b, 3] holding the
      coordinates of samples points from a batch of meshes.
    feats_a: A float `Tensor` of shape [B, num_points_a, 3] holding a
      feature vector corresponding to each sampled point in the pointcloud.
    feats_b: A float `Tensor` of shape [B, num_points_b, 3] holding the
      feature vector corresponding to each sampled point in the pointcloud.

  Returns:
    feats_a_nearest_to_b: A float `Tensor` of shape [B, num_points_b, 3]
      representing the features corresponding to the re-ordered points from
      `feats_a` that are the closest to the points in `feats_b`.
    feats_b_nearest_to_a: A float `Tensor` of shape [B, num_points_a, 3]
      representing the features corresponding to the re-ordered points from
      `feats_b` that are the closest to the points in `feats_a`.
  """
  batch_size = tf.shape(pointcloud_a)[0]
  num_points_a = tf.shape(pointcloud_a)[1]
  num_points_b = tf.shape(pointcloud_b)[1]

  square_distances = _compute_square_distances(pointcloud_a, pointcloud_b)

  # IntTensor[B, num_points_a] where the element i of the vector holds the value j
  # such that point `pointcloud_a[b, i, :]`"s nearest neightbor is
  # `pointcloud_b[b, j, :]`.
  a_nearest_neighbors_in_b = tf.argmin(
      square_distances, axis=-1, output_type=tf.int32
  )
  # IntTensor[B, num_points_b] where the element i of the vector holds the value j
  # such that point `pointcloud_b[b, i, :]`"s nearest neightbor is
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
      [batch_ind_a, tf.expand_dims(a_nearest_neighbors_in_b, axis=-1)], axis=-1
  )
  b_nearest_neighbors_in_a_ind = tf.concat(
      [batch_ind_b, tf.expand_dims(b_nearest_neighbors_in_a, axis=-1)], axis=-1
  )

  feats_a_nearest_to_b = tf.gather_nd(feats_a, b_nearest_neighbors_in_a_ind)
  feats_b_nearest_to_a = tf.gather_nd(feats_b, a_nearest_neighbors_in_b_ind)

  return feats_a_nearest_to_b, feats_b_nearest_to_a

def _cosine_similarity(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  """Computes the cosine similarity between `Tensor`s `a` and `b` along axis=-1.

  Args:
    a: A float `Tensor` with at least 2 dimensions with shape [B, ..., D].
       where D is the dimensionality of the points.
    b: A float `Tensor` with at least 2 dimensions and the same shape as `a`.

  Returns:
    cosine_similarity: A float `Tensor` with shape [B, ...] where the new
      innermost dim is the cosine similarity between of the points (innermost
      dim) in `a` and `b`.
  """
  a_norm = tf.linalg.l2_normalize(a, axis=-1)
  b_norm = tf.linalg.l2_normalize(b, axis=-1)
  return tf.reduce_sum(a_norm * b_norm, axis=-1)

def _add_pointcloud_distances(dist_a: tf.Tensor,
                              dist_b: tf.Tensor,
                              weights: Optional[tf.Tensor],
                              point_reduction: Union[str, None] = "mean",
                              batch_reduction:
                              Union[str, None] = "mean") -> tf.Tensor:
  """Compute a symmetric pointcloud distance from two asymmetric distances.

  Args:
    dist_a: A float `Tensor` with shape [B, num_points_a] representing an
      (asymmetric) distance metric with respect to pointcloud a.
    dist_b: A float `Tensor` with shape [B, num_points_b] representing an
      (asymmetric) distance metric with respect to pointcloud b.
    weights: An optional float `Tensor` with shape [B,] giving weights for
      batch elements for reduction operation.
    point_reduction: Reduction operation to apply for the loss across the
      points, can be one of ["mean", "sum"].
    batch_reduction: Reduction operation to apply for the loss across the
      batch, can be one of ["mean", "sum"] or None.

  Returns:
    distance: A float `Tensor` of shape [B,] if `batch_reduction` is None
      or [] (i.e. scalar) otherwise, representing a reduced (and potentially
      weighted) symmetric pointcloud distance metric.
  """
  batch_size = tf.cast(tf.shape(dist_a)[0], tf.float32)
  num_points_a = tf.cast(tf.shape(dist_a)[1], tf.float32)
  num_points_b = tf.cast(tf.shape(dist_b)[1], tf.float32)

  # Apply batch weights.
  if weights is not None:
    dist_a *= tf.expand_dims(weights, axis=-1)
    dist_b *= tf.expand_dims(weights, axis=-1)

  # Point reduction: sum all dists into one element per batch.
  dist_a = tf.reduce_sum(dist_a, axis=-1)
  dist_b = tf.reduce_sum(dist_b, axis=-1)

  if point_reduction == "mean":
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
                 point_reduction: Union[str, None] = "mean",
                 batch_reduction: Union[str, None] = "mean") -> tf.Tensor:
  """Computes the chamfer distance between two pointclouds a and b.

  Args:
    pointcloud_a: A float `Tensor` of shape [B, num_points_a, 3] holding the
      coordinates of samples points from a batch of meshes. The pointcloud
      for a mesh will be 0 (i.e. pointcloud_a[i, :, :] = 0) if the mesh is
      empty (i.e. verts_mask[i,:] = 0).
    pointcloud_b: A float `Tensor` of shape [B, num_points_b, 3] holding the
      coordinates of samples points from a batch of meshes.
    weights: An optional float `Tensor` with shape [B,] giving weights for
      batch elements for reduction operation.
    point_reduction: Reduction operation to apply for the loss across the
      points, can be one of ["mean", "sum"].
    batch_reduction: Reduction operation to apply for the loss across the
      batch, can be one of ["mean", "sum"] or None.

  Returns:
    chamfer_distance: A float `Tensor` of shape [B,] if `batch_reduction` is
      None or [] (i.e. scalar) otherwise, representing a reduced (and
      potentially weighted) distance between the pointclouds a and b.
  """
  square_distances = _compute_square_distances(pointcloud_a, pointcloud_b)

  # FloatTensor[B, num_points_a] representing the minimum of the squared
  # distance from each point in pointcloud a to each of the points in
  # pointcloud b.
  min_square_dist_a_to_b = tf.reduce_min(input_tensor=square_distances,
                                         axis=-1)
  # FloatTensor[B, num_points_b]
  min_square_dist_b_to_a = tf.reduce_min(input_tensor=square_distances,
                                         axis=-2)

  return _add_pointcloud_distances(
      min_square_dist_a_to_b, min_square_dist_b_to_a,
      weights, point_reduction, batch_reduction
  )

def normal_loss(pointcloud_a: tf.Tensor,
                pointcloud_b: tf.Tensor,
                normals_a: tf.Tensor,
                normals_b: tf.Tensor,
                weights: Optional[tf.Tensor] = None,
                point_reduction: Union[str, None] = "mean",
                batch_reduction: Union[str, None] = "mean") -> tf.Tensor:
  """Computes the absolute normal distance between two pointclouds a and b.

  Args:
    pointcloud_a: A float `Tensor` of shape [B, num_points_a, 3] holding the
      coordinates of samples points from a batch of meshes. The pointcloud
      for a mesh will be 0 (i.e. pointcloud_a[i, :, :] = 0) if the mesh is
      empty (i.e. verts_mask[i,:] = 0).
    pointcloud_b: A float `Tensor` of shape [B, num_points_b, 3] holding the
      coordinates of samples points from a batch of meshes.
    normals_a: A float `Tensor` of shape [B, num_points_a, 3] holding the
      normal vector for each sampled point. Like `pointcloud_x`, an empty
      mesh will correspond to a 0 normals matrix.
    normals_b: A float `Tensor` of shape [B, num_points_b, 3] holding the
      normal vector for each sampled point.
    weights: An optional float `Tensor` with shape [B,] giving weights for
      batch elements for reduction operation.
    point_reduction: Reduction operation to apply for the loss across the
      points, can be one of ["mean", "sum"].
    batch_reduction: Reduction operation to apply for the loss across the
      batch, can be one of ["mean", "sum"] or None.

  Returns:
    normal_distance: A float `Tensor` of shape [B,] if `batch_reduction` is
      None or [] (i.e. scalar) otherwise, representing a reduced (and
      potentially weighted) absolute normal (cosine) distance of normals
      between the pointclouds a and b.
  """
  normals_a_nearest_to_b, normals_b_nearest_to_a = \
    _get_features_nearest_neighbors(pointcloud_a, pointcloud_b,
                                    normals_a, normals_b)

  # FloatTensor[B, num_points_a] representing the absolute normal distance
  # between the normals in pointcloud a and the normals of the closest points
  # in pointcloud b.
  abs_normal_dist_a = 1 - tf.math.abs(
      _cosine_similarity(normals_b_nearest_to_a, normals_a)
  )
  # FloatTensor[B, num_points_b]
  abs_normal_dist_b = 1 - tf.math.abs(
      _cosine_similarity(normals_a_nearest_to_b, normals_b)
  )

  return _add_pointcloud_distances(
      abs_normal_dist_a, abs_normal_dist_b,
      weights, point_reduction, batch_reduction
  )

def edge_loss(verts: tf.Tensor,
              verts_mask: tf.Tensor,
              edges: tf.Tensor,
              edges_mask: tf.Tensor) -> tf.Tensor:
  """Computes mesh edge length regularization loss averaged across all meshes
  in a batch.

  Each mesh contributes equally to the final loss, regardless of
  the number of edges per mesh in the batch by weighting each mesh with the
  inverse number of edges. For example, if mesh 3 (out of N) has only E=4
  edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
  contribute to the final loss.

  Args:
    verts: A float `Tensor` of shape [B, num_verts, 3], where the last dimension
      contains all (x,y,z) vertex coordinates in the initial mesh.
    verts_mask: An int `Tensor` of shape [B, num_verts] representing a mask for
      valid vertices in the watertight mesh.
    edges: A `Tensor` of shape [B, num_faces * 3, 2], where the last dimension
      contain the vertex indices that make up the edge. This may include
      duplicate edges.
    edges_mask: A `Tensor` of shape [B, num_faces * 3], a mask for valid edges
      in the watertight mesh.

  Returns:
    edge_loss: Average edge loss for all meshes (i.e. batch elements). This is
      the normalized and weighted squared l2 norm of the vector differences
      between the vertex pairs for each valid edge.
  """
  shape = tf.shape(edges_mask)
  batch_size, num_edges = tf.cast(shape[0], tf.float32), shape[1]

  num_valid_edges = tf.reduce_sum(edges_mask, axis=-1, keepdims=True)
  # FloatTensor[B, 1] where the element in each batch contains either
  # the number of edges in each mesh or 1 in the case where the mesh is empty.
  num_valid_edges = tf.where(tf.equal(num_valid_edges, 0),
                             tf.ones_like(num_valid_edges),
                             num_valid_edges)
  # FloatTensor[B, num_edges] where each element is the weight for each
  # edge based on the number of edges in the mesh it corresponds to. All edges
  # in a mesh will have the same weight.
  weights = 1 / tf.repeat(num_valid_edges, num_edges, axis=-1)

  # Each FloatTensor[B, num_edges, 3] representing the pairs of valid vertices
  # from `edges`.
  v0, v1 = get_verts_from_indices(
      verts, verts_mask, edges, edges_mask, num_inds_per_set=2
  )
  # FloatTensor[B, num_edges] representing the squared l2 norm of the
  # vector difference between the vertices of each edge.
  sqr_l2_per_edge = tf.norm(v1 - v0, axis=-1) ** 2.0
  norm_sqr_l2_per_edge = sqr_l2_per_edge * weights

  return tf.reduce_sum(norm_sqr_l2_per_edge) / batch_size

class MeshLoss(tf.keras.losses.Loss):
  """Mesh R-CNN losses for the penalizing the predicted voxels and meshes."""

  def __init__(self,
               voxel_weight: float = 0.0,
               chamfer_weight: float = 1.0,
               normal_weight: float = 0.0,
               edge_weight: float = 0.1,
               true_num_samples: int = 5000,
               pred_num_samples: int = 5000) -> None:
    """Mesh Loss Initialization.

    Args:
      voxel_weight: A `float` for weight of the voxel loss.
      chamfer_weight: A `float` for weight of the chamfer loss.
      normal_weight: A `float` for weight of the normal loss.
      edge_weight: A `float` for weight of the edge loss.
      true_num_samples: An `int` for the number of points to sample from the
        ground truth mesh.
      pred_num_samples: An `int` for the number of points to sample from the
        predicted mesh.
    """
    self._voxel_weight = voxel_weight
    self._chamfer_weight = chamfer_weight
    self._normal_weight = normal_weight
    self._edge_weight = edge_weight
    self._true_num_samples = true_num_samples
    self._pred_num_samples = pred_num_samples
    self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

  def __call__(self,
               voxels_true: tf.Tensor,
               voxels_pred: tf.Tensor,
               meshes_true: dict,
               meshes_pred: dict,
               edges_pred: tf.Tensor,
               edges_mask_pred: tf.Tensor
              ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Computes the voxel and mesh losses (chamfer, normal, and edge losses).

    Args:
      voxels_true: An int `Tensor` of shape [B, D, H, W] that contains the voxel
        occupancy ground truth. D, H, and W are equal.
      voxels_pred: An int `Tensor` of shape [B, D, H, W] that contains the voxel
        occupancy prediction. D, H, and W are equal.
      meshes_true: A `dict` containing the `Tensor`s corresponding to the mesh
        verts, faces, and their corresponding masks for the ground truth meshes.
        See `cubify` in `ops/cubify.py` for more information.
      meshes_pred: A `dict` containing the `Tensor`s corresponding to the mesh
        verts, faces, and their corresponding masks for the predicted meshes.
      edges_pred: A `Tensor` of shape [B, num_faces * 3, 2], where the last
        dimension contains the vertex indices that make up the edges in the
        predicted meshes. This may include duplicate edges.
      edges_mask_pred: A `Tensor` of shape [B, num_faces * 3], a mask for valid
        edges in the predicted meshes.

    Returns:
      total_loss: A float scalar `Tensor` representing the sum of the chamfer,
        normal, edge, and voxel losses.
      voxel_loss: A float scalar `Tensor` representing the voxel loss.
      chamfer_loss: A float scalar `Tensor` representing the chamfer loss.
      normal_loss: A float scalar `Tensor` representing the normal loss.
      edge_loss: A float scalar `Tensor` representing the edge loss.
    """
    voxel_loss = 0
    if self._voxel_weight > 0:
      voxel_loss = self._voxel_weight * self._binary_crossentropy(
          voxels_true, voxels_pred
      )

    true_sampler = MeshSampler(self._true_num_samples)
    pred_sampler = MeshSampler(self._pred_num_samples)
    pointcloud_true, normals_true, _ = true_sampler.sample_meshes(
        meshes_true["verts"], meshes_true["verts_mask"],
        meshes_true["faces"], meshes_true["faces_mask"],
    )
    pointcloud_pred, normals_pred, _ = pred_sampler.sample_meshes(
        meshes_pred["verts"], meshes_pred["verts_mask"],
        meshes_pred["faces"], meshes_pred["faces_mask"],
    )

    chamfer_loss_ = self._chamfer_weight * chamfer_loss(
        pointcloud_true, pointcloud_pred
    )
    normal_loss_ = self._normal_weight * normal_loss(
        pointcloud_true, pointcloud_pred, normals_true, normals_pred
    )
    edge_loss_ = self._edge_weight * edge_loss(
        meshes_pred["verts"], meshes_pred["verts_mask"],
        edges_pred, edges_mask_pred
    )

    total_loss = voxel_loss + chamfer_loss_ + normal_loss_ + edge_loss_
    return total_loss, voxel_loss, chamfer_loss_, normal_loss_, edge_loss_

  def get_config(self) -> dict:
    config = {
        "chamfer_weight": self._chamfer_weight,
        "normal_weight": self._normal_weight,
        "edge_weight": self._edge_weight,
        "voxel_weight": self._voxel_weight,
        "true_num_samples": self._true_num_samples,
        "pred_num_samples": self._pred_num_samples,
    }
    base_config = super(MeshLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
