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

"""Utility functions for computing mesh losses.

Note: The following are used as shorthands for dimensions:
  B: Batch size
  Nv: Number of vertices. Calculated as `(vox_grid_dim + 1)**3`.
  Nf: Number of faces.  Calculated as `12 * (vox_grid_dim)**3`.
  Ns: Number of points to sample from mesh
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch

COORD_DIM = 3


def sample_meshes(
  verts: tf.Tensor,
  verts_mask: tf.Tensor,
  faces: tf.Tensor,
  faces_mask: tf.Tensor,
  num_samples: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
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
    num_samples: Integer giving the number of point samples per mesh (Ns above).

  Returns:
    samples: A float `Tensor` of shape [B, Ns, 3] holding the coordinates
      of sampled points from each mesh in the batch. A samples matrix for a
      mesh will be 0 (i.e. samples[i, :, :] = 0) if the mesh is empty
      (i.e. verts_mask[i,:] all 0).
    normals:  A float `Tensor` of shape [B, Ns, 3] holding the normal vector
      for each sampled point. Like `samples`, an empty mesh will correspond
      to a 0 normals matrix.
  """
  verts_shape = tf.shape(verts)
  batch_size, num_verts, = verts_shape[0], verts_shape[1]
  num_faces = tf.shape(faces)[1]

  v0, v1, v2 = _get_face_vertices(verts, faces, batch_size, num_faces)
  areas, normals = _get_face_areas_and_normals(v0, v1, v2)

  sample_face_idxs = _sample_faces(areas, batch_size, num_samples)


  return tf.constant(0), tf.constant(0)


def _get_face_vertices(
  verts: tf.Tensor, faces: tf.Tensor, batch_size: int, num_faces: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Extracts three `Tensors` corresponding to the vertices for each face.

  Args:
    verts: A float `Tensor` of shape [B, Nv, 3], where the last dimension
      contains all (x,y,z) vertex coordinates in the initial mesh.
    faces: An int `Tensor` of shape [B, Nf, 3], where the last dimension
      contains the verts indices that make up the face. This may include
      duplicate faces.
    batch_size: `int`, specifies the number of batch elements.
    batch_size: `int`, specifies the number of faces in each mesh.

  Returns:
    v0, v1, v2: A tuple of three float `Tensor`s each of shape [B, Nf, 3]
      holding the ith (0, 1, or 2) vertex for each face of the mesh.
  """
  # IntTensor[B, Nf, 1] where the single column for each batch is the batch idx.
  batch_ind = tf.repeat(
    tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=-1), axis=-1),
    num_faces,
    axis=1,
  )

  face_verts = [None] * COORD_DIM
  for i in range(COORD_DIM):
    # IntTensor[B, Nf, 1] where the element in each row is the ith (0, 1, or 2)
    # vertex index from `faces`.
    faces_vert_ind = tf.transpose(
      tf.expand_dims(faces[:, :, i], axis=0), perm=[1, 2, 0]
    )
    # IntTensor[B, Nf, 1]: Concatenated tensor used to index into `verts`.
    vert_ind = tf.concat((batch_ind, faces_vert_ind), axis=-1)
    # FloatTensor[B, Nf, 3]: The ith vertex for each face of the mesh.
    face_verts[i] = tf.gather_nd(verts, vert_ind)

  v0, v1, v2 = face_verts
  return v0, v1, v2


def _get_face_areas_and_normals(
  v0: tf.Tensor, v1: tf.Tensor, v2: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes the areas and normal vector for the mesh faces.

  Both areas and normal vectors are computed in this single function as they
  both rely on the cross product of two sides of a mesh face.

  Args:
    v0: A float `Tensor` of shape [B, Nf, 3] corresponding to the 0th vertex
      for each face in the mesh.
    v1: A float `Tensor` of shape [B, Nf, 3] corresponding to the 1th vertex
      for each face in the mesh.
    v2: A float `Tensor` of shape [B, Nf, 3] corresponding to the 2th vertex
      for each face in the mesh.

  Returns:
    face_areas, face_normals: Two float `Tensor`s with shape [B, Nf] and
      [B, Nf, 3] corresponding to the areas and normalized normal vectors for
      each face in the mesh.
  """
  vert_normals = tf.linalg.cross((v1 - v0), (v2 - v1))
  face_areas = tf.norm(vert_normals, axis=-1) / 2
  # Normalize the normal vector calculated from the cross prod of the vertices.
  vert_normals_norm = tf.repeat(
    tf.expand_dims(tf.norm(vert_normals, axis=-1), axis=-1),
    3,
    axis=-1)
  face_normals = vert_normals / vert_normals_norm
  return face_areas, face_normals


def _sample_faces(areas: tf.Tensor, batch_size: int, num_samples: int) -> tf.Tensor:
  """Computes the indices to sample the mesh faces.

  Points are uniformly sampled from the surface of the mesh by sampling a face
  f = (v1, v2, v3) from the mesh where the probability distribution of faces is
  given by a multinomial distribution where the unnormalized probabilities are
  given by the area of each face.

  Args:
    areas: A float `Tensor` with shape [B, Nf]

  Returns:
    sample_face_idxs: TODO
  """
  return tf.constant(0)


def _get_rand_barycentric_coords() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """TODO"""
  return tf.constant(0), tf.constant(0), tf.constant(0)
