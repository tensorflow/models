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
import tensorflow as tf

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, grid, align_corners=False):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H, 'int32')
    max_x = tf.cast(W, 'int32')
    zero = tf.zeros([], dtype='int32')

    x = grid[:,0,:,:]
    y = grid[:,1,:,:]

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    if align_corners:
      x = ((x + 1.0) / 2) * tf.cast(max_x-1, 'float32')
      y = ((y + 1.0) / 2) * tf.cast(max_y-1, 'float32')
    else:
      x = ((x + 1.0) * tf.cast(max_x, 'float32') - 1.0) / 2.0
      y = ((y + 1.0) * tf.cast(max_y, 'float32') - 1.0) / 2.0

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # calculate deltas
    wa = (tf.cast(x1, 'float32')-x) * (tf.cast(y1, 'float32')-y)
    wb = (tf.cast(x1, 'float32')-x) * (y-tf.cast(y0, 'float32'))
    wc = (x-tf.cast(x0, 'float32')) * (tf.cast(y1, 'float32')-y)
    wd = (x-tf.cast(x0, 'float32')) * (y-tf.cast(y0, 'float32'))

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    
    x0 = tf.clip_by_value(x0, zero, max_x - 1)
    x1 = tf.clip_by_value(x1, zero, max_x - 1)
    y0 = tf.clip_by_value(y0, zero, max_y - 1)
    y1 = tf.clip_by_value(y1, zero, max_y - 1)
    
    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')


    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def vert_align(feats, verts, return_packed: bool = False, align_corners: bool = True):
  if tf.is_tensor(verts):
    if len(verts.shape) != 3:
      raise ValueError("verts tensor should be 3 dimensional")
    grid = verts
  else:
    raise ValueError(
        "verts must be a tensor.")

  grid = grid[:, None, :, :2]

  if tf.is_tensor(feats):
    feats = [feats]
  for feat in feats:
    if len(feat.shape) != 4:
      raise ValueError("feats.shape (N, C, H, W)")
    if grid.shape[0] != feat.shape[0]:
      raise ValueError("inconsistent batch dimension")

  feats_sampled = []
  for feat in feats:
    feat_sampled = tf.transpose(
        bilinear_sampler(tf.transpose(feat, [0, 2, 3, 1]), tf.transpose(grid, [0, 3, 1, 2]), align_corners=align_corners)
        ,[0, 3, 1, 2])
    feat_sampled = tf.transpose(tf.squeeze(feat_sampled, axis = 2), [0,2,1])
    feats_sampled.append(feat_sampled)
  feats_sampled = tf.concat(feats_sampled, axis = 2)

  return feats_sampled
  
