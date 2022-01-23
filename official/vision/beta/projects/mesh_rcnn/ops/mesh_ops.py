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

"""Mesh Ops"""
import tensorflow as tf


def compute_edges(faces: tf.Tensor, faces_mask: tf.Tensor):
  """Computes the edges of a mesh.

  The faces of a mesh consists of the 3 integers (v0, v1, v2) for each vertex,
  the edges for each face are namely (v0, v1), (v1, v2), and (v2, v0).
  The faces mask is used to create an initial mask for the edges. Since
  the initial mask contains duplicate edges (along touching faces), the mask
  is updated to mark only unique valid edges.

  Args:
    faces: A `Tensor` of shape [B, num_faces, 3], where the last dimension
      contain the verts indices that make up the face. This may include
      duplicate faces.
    faces_mask: A `Tensor` of shape [B, num_faces], a mask for valid faces in
      the watertight mesh.

  Returns:
    edges: A `Tensor` of shape [B, num_faces * 3, 2], where the last dimension
      contain the vertex indices that make up the edge. This may include
      duplicate edges.
    edges_mask: A `Tensor` of shape [B, num_faces * 3], a mask for valid edges
      in the watertight mesh.
  """
  # Faces are identical in the batch, only one is needed to create the edges
  shape = tf.shape(faces)
  batch_size, _, _ = shape[0], shape[1], shape[2]
  faces = faces[0, :]

  # Use the 3 vertices of each face to compute the edges
  v0, v1, v2 = tf.split(faces, 3, axis=-1)

  e01 = tf.concat([v0, v1], axis=1)
  e12 = tf.concat([v1, v2], axis=1)
  e20 = tf.concat([v2, v0], axis=1)

  edges = tf.concat([e12, e20, e01], axis=0)

  # Create an initial mask for the edges using faces_mask
  edges_mask = tf.repeat(faces_mask, 3, axis=1)

  # Sort vertex ordering in each edge [v0, v1] so that v0 >= v1
  edges = tf.stack(
      [tf.math.reduce_min(edges, axis=1),
       tf.math.reduce_max(edges, axis=1)],
      axis=-1
  )

  # Convert the edges to scalar values (to be used for sorting)
  edges_max = tf.math.reduce_max(edges) + 1
  edges_hashed = edges[:, 0] * edges_max + edges[:, 1]

  # Sort the edges in increasing order and update the mask accordingly
  sorted_edge_indices = tf.argsort(edges_hashed)
  edges_hashed = tf.gather(edges_hashed, sorted_edge_indices)
  edges = tf.gather(edges, sorted_edge_indices)
  edges_mask = tf.gather(edges_mask, sorted_edge_indices, axis=1)

  # Compare adjacent edges to find the non-unique edges
  unique_edges_mask = tf.concat(
      [[True], edges_hashed[1:] != edges_hashed[:-1]], axis=0)

  # Multiply the masks to create the edges mask for valid and unique edges
  edges_mask = edges_mask * tf.cast(unique_edges_mask, edges_mask.dtype)

  # Re-batch the edges
  edges = tf.expand_dims(edges, axis=-1)
  edges = tf.tile(edges, multiples=[batch_size, 1, 1])
  edges = tf.reshape(edges, shape=[batch_size, -1, 2])

  return edges, edges_mask
