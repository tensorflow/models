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

"""Ops needed for computing mesh losses.

Note: The following are used as shorthands for dimensions:
  B: Batch size.
  Nv: Number of vertices. Calculated as `(vox_grid_dim + 1)**3`.
  Nf: Number of faces.  Calculated as `12 * (vox_grid_dim)**3`.
  Ns: Number of points to sample from mesh.
"""

from typing import Tuple

import tensorflow as tf

# Number of dimensions in coordinate system used.
COORD_DIM = 3


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
    batch_size = tf.shape(verts)[0]
    num_faces = tf.shape(faces)[1]

    v0, v1, v2 = get_face_vertices(
      verts, verts_mask, faces, faces_mask, batch_size, num_faces
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
    face_areas = tf.norm(vert_normals, axis=-1) / 2
    # Normalize the normal vector calculated from the cross product of the verts.
    vert_normals_norm = tf.repeat(
      tf.expand_dims(tf.norm(vert_normals, axis=-1), axis=-1), 3, axis=-1
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

    sampled_verts_ind = tf.concat([batch_ind, sample_faces_ind], -1)
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
    eps1 = tf.random.uniform([batch_size, self._num_samples, 1], 0, 1)
    eps2 = tf.random.uniform([batch_size, self._num_samples, 1], 0, 1)
    eps1_sqrt = tf.math.sqrt(eps1)
    w0 = 1.0 - eps1_sqrt
    w1 = (1.0 - eps2) * eps1_sqrt
    w2 = eps2 * eps1_sqrt
    return w0, w1, w2
