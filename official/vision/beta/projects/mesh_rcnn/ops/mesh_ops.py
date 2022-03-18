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
  faces = faces[0]

  # Use the 3 vertices of each face to compute the edges
  v0, v1, v2 = tf.split(faces, 3, axis=-1)

  e01 = tf.concat([v0, v1], axis=-1)
  e12 = tf.concat([v1, v2], axis=-1)
  e20 = tf.concat([v2, v0], axis=-1)

  edges = tf.concat([e12, e20, e01], axis=0)

  # Create an initial mask for the edges using faces_mask
  edges_mask = tf.tile(faces_mask, [1, 3])

  # Sort vertex ordering in each edge [v0, v1] so that v0 >= v1
  edges = tf.stack(
      [tf.math.reduce_min(edges, axis=-1),
       tf.math.reduce_max(edges, axis=-1)],
      axis=-1
  )

  # Convert the edges to scalar values (to be used for sorting)
  # Multiply the hash by -1 to give valid faces higher priority in sorting
  edges_max = tf.math.reduce_max(edges) + 1

  edges_hashed = (edges[..., 0] * edges_max + edges[..., 1]) * (-1 * tf.cast(edges_mask, edges.dtype))
  
  # Sort the edges in increasing order and update the mask accordingly
  sorted_edge_indices = tf.argsort(edges_hashed, stable=True)
  edges_hashed = tf.gather(edges_hashed, sorted_edge_indices, batch_dims=-1)
  edges = tf.gather(edges, sorted_edge_indices)
  
  edges_mask = tf.gather(edges_mask, sorted_edge_indices, batch_dims=1, axis=-1)
  
  ones = tf.repeat(True, repeats=batch_size)
  unique_edges_mask = tf.concat(
      [tf.reshape(ones, shape=[batch_size, 1]),
       edges_hashed[..., 1:] != edges_hashed[..., :-1]], axis=-1)
  
  # Multiply the masks to create the edges mask for valid and unique edges.
  edges_mask *= tf.cast(unique_edges_mask, edges_mask.dtype)

  return edges, edges_mask

def vert_align(feature_map: tf.Tensor,
               verts: tf.Tensor,
               align_corners: bool = True,
               padding_mode: str = 'border'):
  """Samples features corresponding to mesh's coordinates.

  Each vertex in verts is projected onto the image using its (x, y) coordinates.
  For vertex, a feature is sampled from the feature map is then computed using
  bilinear interpolation.

  Args:
    feature_map: A `Tensor` of shape [B, H, W, C] from which to sample features.
    verts: A `Tensor` of shape [B, V, 3] where the last dimension corresponds
    to the (x, y, z) coordinates of each vertex.
    align_corners: A `bool` that indicates whether the vertex extrema
      coordinates (-1 and 1) will correspond to the corners or centers of the
      pixels. If set to True, the extrema will correspond to the corners.
      Otherwise, they will be set to the centers.
    padding_mode: A `string` that defines the behavior of the sampling for
      vertices that are not within the range [-1, 1]. Can be one of 'zeros',
      'border', or 'reflection'. Only 'border' mode is currently supported.

  Returns:
    verts_features: A `Tensor` of shape [B, V, C], that contains the sampled
      features for each vertex.
  """
  def _get_pixel_value(img, x, y):
    shape = tf.shape(img)
    batch_size = shape[0]
    num_verts = tf.shape(x)[1]

    b = tf.range(0, batch_size)
    batch_repeats = tf.repeat(num_verts, repeats=batch_size)
    b = tf.repeat(b, repeats=batch_repeats)
    b = tf.reshape(b, shape=[batch_size, num_verts])

    indices = tf.stack([b, y, x], axis=2)

    return tf.gather_nd(img, indices)

  height = tf.shape(feature_map)[1]
  width = tf.shape(feature_map)[2]

  max_y = tf.cast(height - 1, dtype=tf.int32)
  max_x = tf.cast(width - 1, dtype=tf.int32)

  x, y = verts[..., 0], verts[..., 1]

  x = tf.cast(x, 'float32')
  y = tf.cast(y, 'float32')

  # Scale coordinates to feature_map dimensions
  if align_corners:
    x = ((x + 1.0) / 2) * tf.cast(max_x, dtype=tf.float32)
    y = ((y + 1.0) / 2) * tf.cast(max_y, dtype=tf.float32)
  else:
    x = ((x + 1.0) * tf.cast(max_x, dtype=tf.float32)) / 2.0
    y = ((y + 1.0) * tf.cast(max_y, dtype=tf.float32)) / 2.0

  # Grab 4 nearest points for each coordinate
  x0 = tf.cast(tf.floor(x), dtype=tf.int32)
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), dtype=tf.int32)
  y1 = y0 + 1

  # Recast as float for delta calculation
  x0 = tf.cast(x0, 'float32')
  x1 = tf.cast(x1, 'float32')
  y0 = tf.cast(y0, 'float32')
  y1 = tf.cast(y1, 'float32')

  # Calculate deltas
  wa = (x1 - x) * (y1 - y)
  wb = (x1 - x) * (y - y0)
  wc = (x - x0) * (y1 - y)
  wd = (x - x0) * (y - y0)

  if padding_mode == 'border':
    x0 = tf.clip_by_value(x0, 0.0, tf.cast(max_x, 'float32'))
    x1 = tf.clip_by_value(x1, 0.0, tf.cast(max_x, 'float32'))
    y0 = tf.clip_by_value(y0, 0.0, tf.cast(max_y, 'float32'))
    y1 = tf.clip_by_value(y1, 0.0, tf.cast(max_y, 'float32'))

  x0 = tf.cast(x0, 'int32')
  x1 = tf.cast(x1, 'int32')
  y0 = tf.cast(y0, 'int32')
  y1 = tf.cast(y1, 'int32')

  # Get pixel value at corner coords
  value_a = _get_pixel_value(feature_map, x0, y0)
  value_b = _get_pixel_value(feature_map, x0, y1)
  value_c = _get_pixel_value(feature_map, x1, y0)
  value_d = _get_pixel_value(feature_map, x1, y1)

  # add dimension for addition
  wa = tf.expand_dims(wa, axis=2)
  wb = tf.expand_dims(wb, axis=2)
  wc = tf.expand_dims(wc, axis=2)
  wd = tf.expand_dims(wd, axis=2)

  verts_features = wa * value_a + wb * value_b + wc * value_c + wd * value_d

  return verts_features

def compute_mesh_shape(batch_size, grid_dims):
  verts_shape = [batch_size, (grid_dims+1)**3, 3]
  verts_mask_shape = [batch_size, (grid_dims+1)**3]
  faces_shape = [batch_size, 12*((grid_dims)**3), 3]
  faces_mask_shape = [batch_size, 12*((grid_dims)**3)]

  return verts_shape, verts_mask_shape, faces_shape, faces_mask_shape
