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

"""Mesh Ops.

Note: The following are used as shorthands for dimensions:
  B: Batch size.
  Nv: Number of vertices. Calculated as `(vox_grid_dim + 1)**3`.
  Nf: Number of faces.  Calculated as `12 * (vox_grid_dim)**3`.
  Ns: Number of points to sample from mesh.
"""

from typing import List, Tuple

import tensorflow as tf



# Number of dimensions in coordinate system used.
COORD_DIM = 3

def compute_edges(faces: tf.Tensor, faces_mask: tf.Tensor) -> tf.Tensor:
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

  edges_hashed = edges[..., 0] * edges_max + edges[..., 1]
  edges_hashed *= -1 * tf.cast(edges_mask, edges.dtype)

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
               padding_mode: str = 'border') -> tf.Tensor:
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

def compute_mesh_shape(batch_size: int,
                       grid_dims: int) -> Tuple[list, list, list, list]:
  """Computes the shape of the mesh tensors given a batch size and voxel size.

  Args:
    batch_size: An `int`, the batch size of the mesh.
    grid_dims: An `int`, the dimension of the voxels.

  Returns:
    verts_shape: An `int`, shape of the mesh vertices.
    verts_mask_shape: An `int`, shape of the mesh vertices mask.
    faces_shape: An `int`, shape of the mesh faces.
    faces_mask_shape: An `int`, shape of the mesh faces mask.
  """
  verts_shape = [batch_size, (grid_dims+1)**3, 3]
  verts_mask_shape = [batch_size, (grid_dims+1)**3]
  faces_shape = [batch_size, 12*((grid_dims)**3), 3]
  faces_mask_shape = [batch_size, 12*((grid_dims)**3)]

  return verts_shape, verts_mask_shape, faces_shape, faces_mask_shape

def get_verts_from_indices(
    verts: tf.Tensor,
    verts_mask: tf.Tensor,
    indices: tf.Tensor,
    indices_mask: tf.Tensor,
    num_inds_per_set: int,
) -> List[tf.Tensor]:
  """Extracts `num_inds_per_set` `Tensors` representing a set of vertices.

  The set of vertices are either mesh faces or edges given by `indices`.

  Args:
    verts: A float `Tensor` of shape [B, Nv, 3], where the last dimension
      contains all (x,y,z) vertex coordinates in the mesh.
    verts_mask: An int `Tensor` of shape [B, Nv] representing a mask for
      valid vertices in the watertight mesh.
    indices: An int `Tensor` of shape [B, num_sets, num_inds_per_set] where
      num_sets = Nf or (Nf * 3) and num_inds_per_set = 3 or 2 for faces or
      edges respectively. For either case, the last dimension contains the
      verts indices that make up the face/edge. This may include duplicate
      faces/edges.
    indices_mask: An int `Tensor` of shape [B, num_sets], representing a mask
      for valid indices (verts that compose faces/edges) in the mesh.
    num_inds_per_set: The size of the inner most dimension (e.g. axis=-1). This
      is the number of indices in each set of indices that compose a face/edge
      in the mesh.

  Returns:
    indexed_verts: A list of `num_inds_per_set` float `Tensor`s, each of shape
      [B, num_sets, 3] where each element represents the ith (0, 1(, or 2))
      vertex for each face/edge of the mesh.
  """
  shape = tf.shape(indices)
  batch_size, num_sets = shape[0], shape[1]

  # Zero out unused vertices and faces/edges
  masked_verts = verts * tf.cast(tf.expand_dims(verts_mask, -1), verts.dtype)
  masked_indices = indices * tf.cast(
      tf.expand_dims(indices_mask, -1), indices.dtype
  )

  # IntTensor[B, num_sets, 1] where the single element in the rows for
  # each batch is the batch idx.
  batch_ind = tf.repeat(
      tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=-1), axis=-1),
      num_sets,
      axis=1,
  )

  indexed_verts = [None] * num_inds_per_set
  for i in range(num_inds_per_set):
    # IntTensor[B, num_sets, 1] where the single element in each row is the
    # ith vertex index from `indices` (i.e. indices[:, :, i]).
    vert_i_index = tf.transpose(
        tf.expand_dims(masked_indices[:, :, i], axis=0), perm=[1, 2, 0]
    )
    # IntTensor[B, num_sets, 2]: Concatenated tensor used to index `verts`.
    vert_ind = tf.concat([batch_ind, vert_i_index], axis=-1)
    # FloatTensor[B, num_sets, num_inds_per_set]: The ith vertex for each
    # face/edge of the mesh.
    indexed_verts[i] = tf.gather_nd(masked_verts, vert_ind)

  return indexed_verts


class MeshSampler():
  """Differential Mesh Sampler to sample a cubified mesh."""
  def __init__(self, num_samples: int):
    """Mesh Sampler Initialization.

    Args:
      num_samples: `int` giving the number of point samples per mesh (Ns above).
    """
    self._num_samples = num_samples

  def sample_meshes(
      self,
      verts: tf.Tensor,
      verts_mask: tf.Tensor,
      faces: tf.Tensor,
      faces_mask: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Uniformly samples points and their normals the surface of meshes.

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

    Returns:
      samples: A float `Tensor` of shape [B, Ns, 3] holding the coordinates
        of sampled points from each mesh in the batch. A samples matrix for a
        mesh will be 0 (i.e. samples[i, :, :] = 0) if the mesh is empty
        (i.e. verts_mask[i,:] all 0).
      normals: A float `Tensor` of shape [B, Ns, 3] holding the normal vector
        for each sampled point. Like `samples`, an empty mesh will correspond
        to a 0 normals matrix.
      sampled_verts_ind: A `Tensor`s of shape [B, Ns, 2] where the first element
        of each row in each batch is the batch index and the second element is
        an index of a face in the mesh to sample from.
    """
    shape = tf.shape(faces)
    batch_size, num_faces, _ = shape[0], shape[1], shape[2]

    v0, v1, v2 = get_verts_from_indices(
        verts, verts_mask, faces, faces_mask, num_inds_per_set=3
    )
    areas, face_normals = self._get_face_areas_and_normals(v0, v1, v2)

    sampled_verts_ind = self._sample_faces(areas, batch_size, num_faces)

    # Each FloatTensor[B, Ns, 3]: The ith (0, 1, or 2) vertices for each face of the
    # sampled mesh.
    smpl_v0 = tf.gather_nd(v0, sampled_verts_ind)
    smpl_v1 = tf.gather_nd(v1, sampled_verts_ind)
    smpl_v2 = tf.gather_nd(v2, sampled_verts_ind)
    # FloatTensor[B, Ns, 3]: The normals for the sampled mesh faces.
    normals = tf.gather_nd(face_normals, sampled_verts_ind)

    w0, w1, w2 = self._get_rand_barycentric_coords(batch_size)

    # FloatTensor[B, Ns, 3]: Each vertex (x,y,z) for each sampled face is
    # elementwise multiplied with the respective barycentric ‘weight’.
    # These are added together to produce the N vertices sampled from the
    # triangle face of each sampled face of each mesh.
    samples = w0 * smpl_v0 + w1 * smpl_v1 + w2 * smpl_v2

    return samples, normals, sampled_verts_ind


  @staticmethod
  def _get_face_areas_and_normals(
      v0: tf.Tensor, v1: tf.Tensor, v2: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the areas and normal vector for the mesh faces.

    Both areas and normal vectors are computed in this single function as they
    both rely on the cross product of two sides of a mesh face.

    Args:
      v0: A float `Tensor` of shape [B, Nf, 3] corresponding to the 0th vertex
        for each face in the (masked) mesh.
      v1: A float `Tensor` of shape [B, Nf, 3] corresponding to the 1th vertex
        for each face in the (masked) mesh.
      v2: A float `Tensor` of shape [B, Nf, 3] corresponding to the 2th vertex
        for each face in the (masked) mesh.

    Returns:
      face_areas, face_normals: Two float `Tensor`s with shape [B, Nf] and
        [B, Nf, 3] corresponding to the areas and normalized normal vectors for
        each face in the mesh.
    """
    vert_normals = tf.linalg.cross((v1 - v0), (v2 - v1))
    # Area of a triangle = (1/2) * (norm of normal vector to triangle face).
    face_areas = tf.norm(vert_normals, axis=-1) / 2
    # Normalize the normal vector calculated from the cross product of the verts.
    vert_normals_norm = tf.repeat(
        tf.expand_dims(tf.norm(vert_normals, axis=-1), axis=-1),
        repeats=3,
        axis=-1
    )
    face_normals = vert_normals / vert_normals_norm
    # Replace `nan` for masked out faces with zero vectors
    face_normals = tf.where(
        tf.math.is_nan(face_normals), tf.zeros_like(face_normals), face_normals
    )
    return face_areas, face_normals


  def _sample_faces(
      self,
      areas: tf.Tensor,
      batch_size: int,
      num_faces: int,
  ) -> tf.Tensor:
    """Computes a `Tensor` containing the indices of the sampled faces.

    Points are uniformly sampled from the surface of the mesh by sampling a face
    f = (v1, v2, v3) from the mesh where the probability distribution of faces
    is given by a multinomial distribution where the unnormalized probabilities
    are given by the area of each face.

    Args:
      areas: A float `Tensor` with shape [B, Nf] corresponding to the areas for
        each face in the mesh.
      batch_size: `int`, specifies the number of batch elements.
      num_faces: `int`, specifies the number of faces in each mesh.

    Returns:
      sampled_verts_ind: A `Tensor`s of shape [B, Ns, 2] where the first element
        of each row in each batch is the batch index and the second element is
        an index of a face in the mesh to sample from. This is used to gather
        the vertices and corresponding normal vectors of the sampled mesh faces.
    """
    # IntTensor[B, Ns, 1] where the single element in the rows for each batch is
    # the batch idx.
    batch_ind = tf.repeat(
        tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=-1), axis=-1),
        self._num_samples,
        axis=1,
    )
    # IntTensor[B, Ns, 1] where the single element in each row is an index of a
    # face of that mesh to sample from.
    sample_faces_ind = tf.expand_dims(
        tf.random.categorical(
            tf.math.log(areas), self._num_samples, dtype=tf.int32
        ),
        axis=-1,
    )
    # If all areas are zero (masked off mesh faces), `tf.random.categorical` will
    # fill each row with the value `Nf` which will cause an out of range error
    # when using this in the `indices` arg of `tf.gather_nd`. So we should set the
    # elements corresponding to the masked off faces to zero.
    sample_faces_ind_mask = tf.cast(sample_faces_ind == num_faces, tf.int32)
    sample_faces_ind = sample_faces_ind - num_faces * sample_faces_ind_mask

    sampled_verts_ind = tf.concat([batch_ind, sample_faces_ind], axis=-1)
    return sampled_verts_ind


  def _get_rand_barycentric_coords(
      self,
      batch_size: int,
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Computes the three `Tensors` corresponding to the random barycentric
      coordinates which are uniformly distributed over a triangle.

    Args:
      batch_size: `int`, specifies the number of batch elements.

    Returns:
      w0, w1, w2: A tuple of three float `Tensor`s each of shape [B, Ns, 1]
        giving random barycentric coordinates.
    """
    eps1 = tf.random.uniform([batch_size, self._num_samples, 1], maxval=1)
    eps2 = tf.random.uniform([batch_size, self._num_samples, 1], maxval=1)
    eps1_sqrt = tf.math.sqrt(eps1)
    w0 = 1.0 - eps1_sqrt
    w1 = (1.0 - eps2) * eps1_sqrt
    w2 = eps2 * eps1_sqrt
    return w0, w1, w2


def convert_obj_to_mesh(obj_filename: str,
                        verts_pad_instances: int,
                        faces_pad_instances: int) -> dict:
  """Reads a .obj file and converts its contents into mesh tensors.

  Args:
    obj_filename: A `str`, .obj file to convert.
    verts_pad_instances: An `int`, specifies  how many additional entries to
      pad verts and verts_mask tensors to.
    faces_pad_instances: An `int`, specifies  how many additional entries to
      pad faces and faces_mask tensors to.

  Returns:
    mesh: A dictinary with the following keys:
      'verts': A `Tensor` of shape [num_verts, 3], where the last dimension
        contains all (x,y,z) vertex coordinates in the mesh.
      'verts_mask': A `Tensor` of shape [num_verts], a mask for valid
        vertices in the mesh.
      'faces': A `Tensor` of shape [num_faces, 3], where the last dimension
        contain the verts indices that make up the face.
      'faces_mask': A `Tensor` of shape [num_faces], a mask for valid faces
        in the mesh.
  """

  with open(obj_filename) as f:
    lines = f.readlines()
  
  verts = []
  faces = []

  # Iterate through each line in the file
  for line in lines:
    line = line.split()

    # Vertex entry
    # Vertices only in the form: v x_coordinate y_coordinate z_coordinate
    if line[0] == 'v':
      verts.append([float(line[1]), float(line[2]), float(line[3])])
    
    # Face entry
    # Faces are indices that point to the faces, but can have other properties
    # delimited by '/' character
    if line[0] == 'f':
      faces.append([face_idx.split('/')[0] for face_idx in line])

  num_verts = len(verts)
  num_faces = len(faces)

  verts = tf.convert_to_tensor(verts, dtype=tf.float32)
  faces = tf.convert_to_tensor(faces, dtype=tf.int32)

  # Apply padding and create masks for verts and faces
  if num_verts < verts_pad_instances:
    num_pad = verts_pad_instances - num_verts
    verts_padding = tf.zeros(shape=[num_pad, 3], dtype=verts.dtype)
    verts = tf.concat([verts, verts_padding], axis=0)
    verts_mask = tf.repeat([1, 0], repeats=[num_verts, num_pad])
  else:
    verts_mask = tf.repeat([1], repeats=[num_verts])
  
  if num_faces < faces_pad_instances:
    num_pad = faces_pad_instances - num_faces
    faces_padding = tf.zeros(shape=[num_pad, 3], dtype=faces.dtype)
    faces = tf.concat([faces, faces_padding], axis=0)
    faces_mask = tf.repeat([1, 0], repeats=[num_faces, num_pad])
  else:
    faces_mask = tf.repeat([1], repeats=[num_faces])
  
  mesh = {
    'verts': verts,
    'faces': faces,
    'verts_mask': verts_mask,
    'faces_mask': faces_mask,
  }

  return mesh


def metric_evaluation(
      self,
      pred_verts: tf.Tensor,
      pred_verts_mask: tf.Tensor,
      pred_faces: tf.Tensor,
      pred_faces_mask: tf.Tensor,
      gt_verts: tf.Tensor,
      gt_verts_mask: tf.Tensor,
      gt_faces: tf.Tensor,
      gt_faces_mask: tf.Tensor,
  )-> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    pred_samples, pred_normals, pred_sampled_verts_ind = self.sample_meshes(pred_verts, pred_verts_mask, pred_faces, pred_faces_mask)
    gt_samples, gt_normals, gt_sampled_verts_ind = self.sample_meshes(gt_verts, gt_verts_mask, gt_faces, gt_faces_mask)

    chamfer_loss = tf.nn.loss.chamfer_distance.evaluate(pred_samples, gt_samples)

    precision = tf.nn.metric.precision.evaluate(gt_verts, pred_verts)

    recall = tf.nn.metric.recall.evaluate(gt_verts, pred_verts)

    f_score = tf.nn.metric.fscore.evaluate(gt_verts, pred_verts)

    normals_a_nearest_to_b, normals_b_nearest_to_a = _get_features_nearest_neighbors(pointcloud_a, pointcloud_b, normals_a, normals_b)
 
    abs_normal_dist_a = 1 - tf.math.abs(
      _cosine_similarity(normals_b_nearest_to_a, normals_a)
    )
    abs_normal_dist_b = 1 - tf.math.abs(
      _cosine_similarity(normals_a_nearest_to_b, normals_b)
    )

    normal_consistency = _add_pointcloud_distances(
      abs_normal_dist_a, abs_normal_dist_b)
    ##cos = tf.keras.losses.cosine_similarity(gt_normals, pred_normals)
    ##cos_sim = cos.mean(dim =1)
    ##normal_consistency = 0.5 * (cos_sim)

    return chamfer_loss, precision, recall, f_score, normal_consistency

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
                              dist_b: tf.Tensor) -> tf.Tensor:
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


  # Point reduction: sum all dists into one element per batch.
  dist_a = tf.reduce_sum(dist_a, axis=-1)
  dist_b = tf.reduce_sum(dist_b, axis=-1)

  dist_a /= num_points_a
  dist_b /= num_points_b


  return dist_a + dist_b

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

