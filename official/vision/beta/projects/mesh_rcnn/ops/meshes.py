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

"""Meshes class and operations."""

from typing import List

import tensorflow as tf

from official.vision.beta.projects.mesh_rcnn.ops import mesh_utils


class Meshes():
  """Class for representing and performing computations with meshes.

  Attributes:
    _verts_list: `List` of B tensors, where each tensor represents the
      coordinates for each vertex in the meshes. Each tensor in _verts_list has
      shape [Vn, 3] where Vn = number of vertices in the nth mesh.
    _faces_list: `List` of B tensors, where each tensor represents the
      indices of the vertices in _verts_list for each face in the meshes. Each
      tensor in  _faces_list has shape [Fn, 3] where Fn = number of faces in
      the nth mesh.
    _batch_size: An `int` for the number of meshes contained in this class
    _num_verts_per_mesh: A `Tensor` of shape [B] that contains the number of
        vertices in each mesh.
    _num_faces_per_mesh: A `Tensor` of shape [B] that contains the number of
        faces in each mesh.
    _max_verts_per_mesh: An `int` that gives the maximum number of vertices
        among the meshes.
    _max_faces_per_mesh: An `int` that gives the maximum number of faces
        among the meshes.
    _verts_packed: A `Tensor` of shape [sum(V1, ..., Vn), 3] that represents the
        vertices of the meshes in sequentially packed form.
    _verts_packed_to_mesh_idx: A `Tensor` of shape [sum(V1, ..., Vn)] that gives
        the index of the mesh each vertex in _verts_packed belongs to.
    _mesh_to_verts_packed_first_idx: A `Tensor` of shape [B] that gives
        the index of the first vertex for each mesh (assuming packing is
        sequential).
    _faces_packed: A `Tensor` of shape [sum(F1, ..., Fn), 3] that represents the
        faces of the meshes in sequentially packed form.
    _faces_packed_to_mesh_idx: A `Tensor` of shape [sum(F1, ..., Fn)] that gives
        the index of the mesh each face in _faces_packed belongs to.
    _mesh_to_faces_packed_first_idx: A `Tensor` of shape [B] that gives
        the index of the first face for each mesh (assuming packing is
        sequential).
    _edges_packed = A `Tensor` of shape [sum(E1, ..., En), 2] that represents
        the edges of the meshes in a sequentially packed form, where En is the
        number of edges in the nth mesh.
    _edges_packed_to_mesh_idx = A `Tensor` of shape [sum(E1, ..., En)] that
        gives the index of the mesh each edge in _edges_packed belongs to.
    _mesh_to_edges_packed_first_idx = A `Tensor` of shape [B] that gives
        the index of the first edge for each mesh (assuming packing is
        sequential).
    self._num_edges_per_mesh: A `Tensor` of shape [B] contains the number of
        edges in each mesh.
    self._faces_packed_to_edges_packed A `Tensor` of shape [sum(F1, ..., F2), 3]
        that maps each face in the meshes to the indices of the edges in
        _edges_packed that belong to it.
    self._verts_padded: A `Tensor` of shape [B, _max_verts_per_mesh, 3] for that
        represents the padded form of the vertices. Meshes with less vertices
        than _max_verts_per_mesh are padded at the end with 0.
    self._faces_padded: A `Tensor` of shape [B, _max_faces_per_mesh, 3] for that
        represents the padded form of the faces. Meshes with less faces
        than _max_faces_per_mesh are padded at the end with -1.
    self._face_normals_packed: A `Tensor` of shape [sum(F1, ..., Fn), 3] that
        contains the unit normal vectors for each face in the meshes.
    self._face_normals_padded: # TODO
    self._face_areas_packed: A `Tensor` of shape [sum(F1, ..., Fn)] that
        contains the areas for each face in the meshes.
  """
  def __init__(self, verts_list: List[tf.Tensor], faces_list: List[tf.Tensor]):
    """Meshes Initialization.

    Args:
        verts_list: `List` of tensors that give the x, y, z coordinates of each
            vertex in the meshes.
        faces_list: `List` of tensors that gives the indices of the vertices in
            the corresponding mesh in verts_list that make up the face.
    """
    self._verts_list = verts_list
    self._faces_list = faces_list

    self._batch_size = len(verts_list)

    self._num_verts_per_mesh = tf.constant([len(v) for v in self._verts_list])
    self._num_faces_per_mesh = tf.constant([len(f) for f in self._faces_list])

    self._max_verts_per_mesh = tf.math.reduce_max(self._num_verts_per_mesh)
    self._max_faces_per_mesh = tf.math.reduce_max(self._num_faces_per_mesh)

    self._verts_packed = None
    self._verts_packed_to_mesh_idx = None

    self._mesh_to_verts_packed_first_idx = None

    self._faces_packed = None
    self._faces_packed_to_mesh_idx = None
    self._mesh_to_faces_packed_first_idx = None

    self._edges_packed = None
    self._edges_packed_to_mesh_idx = None
    self._mesh_to_edges_packed_first_idx = None

    self._num_edges_per_mesh = None
    self._faces_packed_to_edges_packed = None

    self._verts_padded = None
    self._faces_padded = None

    self._face_normals_packed = None
    self._face_normals_padded = None

    self._face_areas_packed = None

    # Compute mesh representations/properties
    self.compute_packed()
    self.compute_padded()

    self.compute_edges_packed()
    self.compute_face_areas_normals()

  @property
  def verts_list(self):
    return self._verts_list

  @property
  def faces_list(self):
    return self._faces_list

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def max_verts_per_mesh(self):
    return self._max_verts_per_mesh

  @property
  def max_faces_per_mesh(self):
    return self._max_faces_per_mesh

  @property
  def verts_packed(self):
    return self._verts_packed

  @property
  def verts_packed_to_mesh_idx(self):
    return self._verts_packed_to_mesh_idx

  @property
  def mesh_to_verts_packed_first_idx(self):
    return self._mesh_to_verts_packed_first_idx

  @property
  def num_verts_per_mesh(self):
    return self._num_verts_per_mesh

  @property
  def num_faces_per_mesh(self):
    return self._num_faces_per_mesh

  @property
  def faces_packed(self):
    return self._faces_packed

  @property
  def faces_packed_to_mesh_idx(self):
    return self._faces_packed_to_mesh_idx

  @property
  def mesh_to_faces_packed_first_idx(self):
    return self._mesh_to_faces_packed_first_idx

  @property
  def verts_padded(self):
    return self._verts_padded

  @property
  def faces_padded(self):
    return self._faces_padded

  @property
  def edges_packed(self):
    return self._edges_packed

  @property
  def edges_packed_to_mesh_idx(self):
    return self._edges_packed_to_mesh_idx

  @property
  def mesh_to_edges_packed_first_idx(self):
    return self._mesh_to_edges_packed_first_idx

  @property
  def faces_packed_to_edges_packed(self):
    return self._faces_packed_to_edges_packed

  @property
  def num_edges_per_mesh(self):
    return self._num_edges_per_mesh

  @property
  def verts_padded_to_packed_idx(self):
    # TODO: implement
    pass

  @property
  def face_normals_packed(self):
    return self._face_normals_packed

  @property
  def face_areas_packed(self):
    return self._face_areas_packed

  @property
  def face_normals_padded(self):
    return self._face_normals_padded

  def compute_padded(self):
    """Computes the padded representations of faces and edges of the meshes."""
    self._verts_padded = mesh_utils.list_to_padded_tensor(self._verts_list, 0)
    self._faces_padded = mesh_utils.list_to_padded_tensor(self._faces_list, -1)

  def compute_packed(self):
    """Computes the packed representation of faces and edges of the meshes."""
    # Compute packed vertices
    verts_packed_output = mesh_utils.list_to_packed_tensor(self._verts_list)

    self._verts_packed = verts_packed_output[0]
    self._mesh_to_verts_packed_first_idx = verts_packed_output[2]
    self._verts_packed_to_mesh_idx = verts_packed_output[3]

    # Compute packed faces
    # Note that there may be duplicate faces in the packed tensor. Those
    # duplicate faces may contain vertices that no longer point to the correct
    # indices in verts_packed. To ensure that faces from different meshes will
    # have correct and unique vertices, we will offset them based on the number
    # of vertices in each mesh
    faces_packed_output = mesh_utils.list_to_packed_tensor(self._faces_list)

    self._faces_packed = faces_packed_output[0]
    self._mesh_to_faces_packed_first_idx = faces_packed_output[2]
    self._faces_packed_to_mesh_idx = faces_packed_output[3]

    # Here we offset the faces
    faces_packed_offset = tf.gather_nd(
        self._mesh_to_verts_packed_first_idx,
        tf.expand_dims(self._faces_packed_to_mesh_idx, axis=-1))
    faces_packed_offset = tf.reshape(faces_packed_offset, [-1, 1])

    self._faces_packed += faces_packed_offset

  def compute_edges_packed(self):
    """Computes a packed representation of the edges of the meshes."""
    # get vertices for each face
    v0, v1, v2 = tf.split(self._faces_packed, 3, axis=-1)

    # stack vertices to get each edge, which may include duplicates
    e01 = tf.concat([v0, v1], axis=1)
    e12 = tf.concat([v1, v2], axis=1)
    e20 = tf.concat([v2, v0], axis=1)

    # combine edges together
    edges = tf.concat([e12, e20, e01], axis=0)
    edge_to_mesh = tf.concat(
        [self._faces_packed_to_mesh_idx,
         self._faces_packed_to_mesh_idx,
         self._faces_packed_to_mesh_idx], axis=-1)

    # ensure that each edge [v1, v2] satisfies v1 <= v2
    verts_min = tf.math.reduce_min(edges, axis=-1, keepdims=True)
    verts_max = tf.math.reduce_max(edges, axis=-1, keepdims=True)
    edges = tf.concat([verts_min, verts_max], axis=-1)

    # To remove duplicate edges, we are hashing them into scalar values
    # in order to use tf.unique()
    num_verts = tf.shape(self._verts_packed)[0]
    edges_hash = num_verts * edges[:, 0] + edges[:, 1]

    # torch.unique() automatically sorts the input
    u, _ = tf.unique(tf.sort(edges_hash))

    # We compute the inverse indexes across tf.sort() and tf.unique() here
    # such that inverse_idxs[i] == j means that edges[i] == unique_edges[j]
    inverse_idxs_map = tf.where(tf.expand_dims(u, axis=-1) == edges_hash)
    locations, values = tf.unstack(inverse_idxs_map, axis=-1)
    inverse_idxs = tf.scatter_nd(tf.expand_dims(
        values, axis=-1), locations, shape=tf.shape(locations, tf.int64))

    # We get the sorted hash and the index of each element in the sorted hash
    # mapped to the original array
    sorted_hash, sort_idx = tf.sort(edges_hash), tf.argsort(edges_hash) #TODO dont need sort, just argsort

    # Since the hash is sorted, we know that we have non-duplicate hashes where:
    # sorted_hash[1:] != sorted_hash[:-1] (comparing adjacent hash values)
    unique_mask = sorted_hash[1:] != sorted_hash[:-1]

    # First element is always going to be a non-duplicate
    unique_mask = tf.concat(
        [tf.constant([True], unique_mask.dtype), unique_mask], axis=-1)

    # Gives the indices of unique values in sorted_hash
    unique_idx = sort_idx[unique_mask]

    # Recompute the unique packed edges by reversing the hash
    self._edges_packed = tf.stack([u // num_verts, u % num_verts], axis=-1)

    # Maps each edge in edges_packed to the mesh id it belongs to
    self._edges_packed_to_mesh_idx = tf.gather_nd(
        edge_to_mesh, tf.expand_dims(unique_idx, -1))

    # Maps the each face to the 3 indices in edges_packed that contain its edges
    self._faces_packed_to_edges_packed = tf.transpose(
        tf.reshape(inverse_idxs, [3, tf.shape(self._faces_packed)[0]]))

    # Compute number of edges per mesh
    num_edges_per_mesh = tf.zeros(self._batch_size, dtype=tf.int32)
    ones = tf.ones(shape=tf.shape(
        self._edges_packed_to_mesh_idx), dtype=tf.int32)

    self._num_edges_per_mesh = tf.tensor_scatter_nd_add(
        num_edges_per_mesh, tf.expand_dims(
            self._edges_packed_to_mesh_idx, -1), ones)

    # Compute first idx of each mesh in edges_packed
    num_edges_cumsum = tf.cumsum(self._num_edges_per_mesh)

    self._mesh_to_edges_packed_first_idx = tf.concat(
        [tf.zeros(shape=[1], dtype=tf.int32), num_edges_cumsum[:-1]], axis=0)

  def compute_face_areas_normals(self):
    """Computes the face areas and face normals of the meshes."""
    v0, v1, v2 = tf.split(self._faces_packed, 3, axis=-1)

    # Get the (x, y, z) coordinates of the vertices in the faces
    coords_v0 = tf.gather_nd(self.verts_packed, v0)
    coords_v1 = tf.gather_nd(self.verts_packed, v1)
    coords_v2 = tf.gather_nd(self.verts_packed, v2)

    # Get distance vectors
    d1 = (coords_v1 - coords_v0)
    d2 = (coords_v2 - coords_v0)

    # Compute the cross product and normalize
    # NOTE: see if we need to pre-compute the mags in case of dividing by 0
    cross = tf.linalg.cross(d1, d2)
    normals, mags = tf.linalg.normalize(cross, ord="euclidean", axis=1)

    self._face_normals_packed = normals

    # Magnitude of the faces is the same as the area of the parallelogram
    # that spans the vectors, dividing by 2 gives the triangle area
    areas = mags / 2.0
    self._face_areas_packed = tf.reshape(areas, shape=[-1])
