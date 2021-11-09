import tensorflow as tf

from official.vision.beta.projects.yolo.ops.preprocessing_ops import \
    pad_max_instances


class Meshes():
  """Class for representing and performing computations with meshes.

  Mesh attributes in this class (vertices, faces, edges) have packed and padded
  forms, but all have static shapes and are padded to max instances. The number
  of instances should be statically defined.
  
  For voxel grid dimensions of 48 x 48 x 48, where each voxel can have up to 
  8 vertices and 12 faces, and a batch size of 64:

  verts_max_instances = 48^3 * 8 = 884736
  faces_max_instances = 48^3 * 12 = 1327104
  edges_max_instances = 48^3 * 12 * 3 = 3981312

  The shape of the packed variables with are as follows:
  _verts_packed = [verts_max_instances * batch_size, 3]
  _faces_packed = [faces_max_instances * batch_size, 3]
  _edges_packed = [edges_max_instances * batch_size, 2]

  The shape of the padded variables are as follows:
  _verts_packed = [batch_size, verts_max_instances, 3]
  _faces_packed = [batch_size, faces_max_instances, 3]
  _edges_packed = [batch_size, edges_max_instances, 2]

  Attributes:
    TODO

  """
  def __init__(self, 
               verts_list,
               faces_list,
               verts_max_instances=884736,
               faces_max_instances=1327104,
               edges_max_instances=110592):
    self._verts_list = verts_list
    self._faces_list = faces_list
    self._batch_size = len(verts_list)

    self._verts_max_instances = verts_max_instances
    self._faces_max_instances = faces_max_instances
    self._edges_max_instances = edges_max_instances

    (self._verts_packed, 
     self._num_verts_per_mesh, 
     self._mesh_to_verts_packed_first_idx, 
     self._verts_packed_to_mesh_idx) = self.compute_verts_packed(
        self._verts_list)

    (self._faces_packed,
     self._num_faces_per_mesh,
     self._mesh_to_faces_packed_first_idx,
     self._faces_packed_to_mesh_idx) = self.compute_faces_packed(
        self._faces_list, self._mesh_to_verts_packed_first_idx)
    
    self._edges_packed = self.compute_edges_packed(self._faces_packed, 
        self._faces_packed_to_mesh_idx, self._num_verts_per_mesh)

  @tf.function
  def compute_verts_packed(self, verts_list):
    # Pack and pad vertices
    packed = tf.concat(verts_list, axis=0)
    verts_packed = pad_max_instances(packed, self._verts_max_instances, 
        pad_value=0, pad_axis=0)
    
    # For each mesh, get the number of vertices it contains
    num_verts_per_mesh = tf.convert_to_tensor([tf.shape(x)[0] for x in verts_list])

    # Maps each mesh to its starting vertex index in the verts_packed
    mesh_to_verts_packed_first_idx = tf.cumsum(num_verts_per_mesh, axis=0) - num_verts_per_mesh

    # Maps each vertex to the mesh it belongs to
    verts_packed_to_mesh_idx = tf.repeat(tf.range(self._batch_size), num_verts_per_mesh)
    verts_packed_to_mesh_idx = pad_max_instances(verts_packed_to_mesh_idx, 
        self._verts_max_instances, pad_value=-1, pad_axis=0)
    
    return verts_packed, num_verts_per_mesh, mesh_to_verts_packed_first_idx, verts_packed_to_mesh_idx

  @tf.function
  def compute_faces_packed(self, faces_list, mesh_to_verts_packed_first_idx):
    # Pack the faces
    packed = tf.concat(faces_list, axis=0)
    
    # For each mesh, get the number of faces it contains
    num_faces_per_mesh = tf.convert_to_tensor([tf.shape(x)[0] for x in faces_list])

    # Maps each mesh to its starting face index in the faces_packed
    mesh_to_faces_packed_first_idx = tf.cumsum(num_faces_per_mesh, axis=0) - num_faces_per_mesh

    # Maps each vertex to the mesh it belongs to
    faces_packed_to_mesh_idx = tf.repeat(tf.range(self._batch_size), num_faces_per_mesh)
    
    # Offset the faces so that the vertices specific in the faces are consistent
    # with the indices in faces_packed
    faces_packed_offset = tf.gather_nd(
        mesh_to_verts_packed_first_idx,
        tf.expand_dims(faces_packed_to_mesh_idx, axis=-1))
    
    faces_packed_offset = tf.reshape(faces_packed_offset, [-1, 1])

    packed += faces_packed_offset
    faces_packed = pad_max_instances(packed, self._faces_max_instances, 
        pad_value=-1, pad_axis=0)

    faces_packed_to_mesh_idx = pad_max_instances(faces_packed_to_mesh_idx, 
        self._faces_max_instances, pad_value=-1, pad_axis=0)
    
    return faces_packed, num_faces_per_mesh, mesh_to_faces_packed_first_idx, faces_packed_to_mesh_idx
  
  @tf.function
  def compute_edges_packed(self, faces_packed, faces_packed_to_mesh_idx, num_verts_per_mesh):
    # Get each face's vertices
    v0, v1, v2 = tf.split(faces_packed, 3, axis=-1)

    # Stack adjacent vertices to create edges (may contain duplicates)
    e01 = tf.concat([v0, v1], axis=1)
    e12 = tf.concat([v1, v2], axis=1)
    e20 = tf.concat([v2, v0], axis=1)

    # Combine edges together
    edges = tf.concat([e12, e20, e01], axis=0)
    edge_to_mesh = tf.concat(
        [faces_packed_to_mesh_idx,
         faces_packed_to_mesh_idx,
         faces_packed_to_mesh_idx], axis=-1)
    
    # Ensure that each edge [v1, v2] satisfies v1 <= v2
    verts_min = tf.math.reduce_min(edges, axis=-1, keepdims=True)
    verts_max = tf.math.reduce_max(edges, axis=-1, keepdims=True)
    edges = tf.concat([verts_min, verts_max], axis=-1)

    # Hash the edges into scalar values so that we can sort them and identify
    # where the unique edges are
    num_verts = tf.reduce_sum(num_verts_per_mesh)
    edges_hashed = num_verts * edges[:, 0] + edges[:, 1]

    # Sort the scalar hashed edges
    sorted_idx = tf.argsort(edges_hashed)
    sorted_edges_hashed = tf.gather_nd(edges_hashed, tf.expand_dims(sorted_idx, axis=-1))

    # unique_mask is True for each first occurance of an edge, these are where
    # the unique edges are
    unique_mask = sorted_edges_hashed[1:] != sorted_edges_hashed[:-1]
    unique_mask = tf.concat(
        [tf.constant([True], unique_mask.dtype), unique_mask], axis=0)
    
    # Get indices of unique and non-unique edges
    unique_idx = tf.where(unique_mask)
    non_unique_idx = tf.where(1-unique_mask)

    # Replace non-unique edge hashes with num_verts * -1 - 1
    updates = tf.fill(dims=[tf.shape(non_unique_idx)[0], 1], value=num_verts * -1 - 1)
    sorted_edges_hashed = tf.tensor_scatter_nd_update(sorted_edges_hashed, non_unique_idx, updates)

    # Unhash the edges (these shuold all)
    unique_edges = tf.stack([sorted_edges_hashed // num_verts, sorted_edges_hashed % num_verts], axis=-1)
    

    
    
