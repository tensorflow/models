import tensorflow as tf

from typing import Tuple

# Vertices in the unit cube (x, y, z)
UNIT_CUBOID_VERTS = tf.constant(
    [
        [0, 0, 0], # left-top-front
        [0, 0, 1], # left-top-back
        [0, 1, 0], # left-bottom-front
        [0, 1, 1], # left-bottom-back
        [1, 0, 0], # right-top-front
        [1, 0, 1], # right-top-back
        [1, 1, 0], # right-bottom-front
        [1, 1, 1], # right-bottom-back
    ])

NUM_VERTS_PER_CUBOID = tf.shape(UNIT_CUBOID_VERTS)[0]

# Indices for vertices that make up each face in the unit cuboid 
UNIT_CUBOID_FACES = tf.constant(
    [
        [0, 1, 2],
        [1, 3, 2],  # left face: 0, 1
        [2, 3, 6],
        [3, 7, 6],  # bottom face: 2, 3
        [0, 2, 6],
        [0, 6, 4],  # front face: 4, 5
        [1, 5, 4],
        [1, 0, 4],  # up face: 6, 7
        [6, 7, 5],
        [6, 5, 4],  # right face: 8, 9
        [1, 7, 3],
        [1, 5, 7],  # back face: 10, 11
    ])
NUM_FACES_PER_CUBOID = tf.shape(UNIT_CUBOID_FACES)[0]

def hash_flatenned_3d_coords(x: tf.Tensor) -> Tuple[tf.Tensor, int]:
  """Hashes a rank 2 int Tensor with a last dimension of 3 into a rank 1 Tensor.
  
  This hashing scheme only works where there is an upper bound max_val on all of
  the values in the tensor. The input must not contain any negative values.
  Suppose the input tensor has shape [10, 3], the hashed output will have
  shape [10].

  Args:
    x: `Tensor` with rank 2, and 3 as the last dimension.
  Returns:
    hashed_x: `Tensor` with rank 1 for the hashed output.
    max_val: `int` that was used to hash the input tensor.
  """
  max_val = tf.math.reduce_max(x) + 1
  hashed_x = x[:, 0] * max_val ** 2 + x[:, 1] * max_val + x[:, 2]
  return hashed_x, max_val

def unhash_flattened_3d_coords(x: tf.Tensor, max_val: int) -> tf.Tensor:
  """Undos the hash on a rank 1 Tensor and converts it back ot a rank 2 Tensor.
  
  Args:
    x: `Tensor` with rank 1, the hashed tensor.
    max_val: `int` that was used to hash the tensor.
  Returns:
    unhashed_x: `Tensor` with rank 1, and 3 as the last dimension.
  """
  max_val = tf.cast(max_val, x.dtype)
  unhashed_x = tf.stack(
      [x // (max_val ** 2), (x // max_val) % max_val, x % max_val], axis=1)
  return unhashed_x

def generate_3d_coords(x_max: int, y_max: int, z_max:int,
                       flatten_output: bool=False) -> tf.Tensor:
  """Generates a tensor that containing cartesian coordinates.

  This function has two modes. By default, the returned tensor has shape
  [x_max+1, y_max+1, z_max+1, 3], where last dimension contains the x, y, and z
  values at the specified coordinate (i.e. coords[x, y, z] == [x, y, z]).
  If flatten_output is True, the output is flattened to a 2D Tensor and the 
  inner dimension is 3.

  Args:
    x_max: `int`, specifies the maximum x value in the coordinate grid.
    y_max: `int`, specifies the maximum y value in the coordinate grid.
    z_max: `int`, specifies the maximum z value in the coordinate grid.
    flatten_output: `bool`, whether to flatten the output to a 2D Tensor.
  
  Returns:
    coords: `Tensor` that contains the coordinate values with shape 
      [x_max+1, y_max+1, z_max+1, 3] when flatten_output is False and shape
      [(x_max+1)*(y_max+1)*(z_max+1), 3] when flatten_output is True. The values
      in the last dimension are the x, y, and z values at a coordinate location.
  """
  x = tf.range(0, x_max+1, dtype=tf.int32)
  y = tf.range(0, y_max+1, dtype=tf.int32)
  z = tf.range(0, z_max+1, dtype=tf.int32)

  x_coords, y_coords, z_coords = tf.meshgrid(x, y, z, indexing='ij')
  coords = tf.stack([x_coords, y_coords, z_coords], axis=-1)

  if flatten_output:
    coords = tf.reshape(coords, shape=[-1, 3])

  return coords

def initialize_mesh(grid_dims: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Initializes the vertices and faces for a complete cuboid mesh.

  This function generates 2 rank 2 Tensors, one for vertices and another for
  faces. The vertices are the initial (x,y,z) coordinates of each vertex
  in the mesh normalized between 0 and 1. The faces are the indices of the 3 
  vertices that define the face. Note that there may be duplicate faces.

  Args:
    grid_dims: `int`, specifies the length, width, and depth of the mesh.
  
  Returns:
    verts: `Tensor` of shape [num_verts, 3], where num_verts = (grid_dim+1)**3.
    faces: `Tensor` of shape [num_faces, 3], where num_verts = 12*(grid_dim)**3.
  """
  num_cuboids = (grid_dims) ** 3

  # Generate the vertex locations
  coords = generate_3d_coords(grid_dims, grid_dims, grid_dims, True)

  # Normalize the verts so that they fall between 0 and 1
  verts = coords / grid_dims

  # Generate offsets for the unit cube verts for the grid
  offsets = generate_3d_coords(grid_dims-1, grid_dims-1, grid_dims-1)
  offsets = tf.expand_dims(offsets, axis=-2)
  
  # Create unit cubes for each cuboid
  cuboid_verts = tf.tile(UNIT_CUBOID_VERTS, multiples=[num_cuboids, 1])
  cuboid_verts = tf.reshape(
      cuboid_verts,
      shape=[grid_dims, grid_dims, grid_dims, NUM_VERTS_PER_CUBOID, 3])
  
  # Add the offsets so that each cube in the grid has the correct vertices
  cuboid_verts += offsets
  
  # Map the vertices of each cuboid in the grid to the predefined coords
  cuboid_verts = tf.reshape(cuboid_verts, shape=[-1, 3])
  
  # Converting to scalar values to save memory when doing tf.equal
  coords_hashed, _ = hash_flatenned_3d_coords(coords)
  cuboid_verts_hashed, _ = hash_flatenned_3d_coords(
      cuboid_verts)
  cuboid_verts_hashed = tf.reshape(cuboid_verts_hashed, shape=[-1, 1, 1])
  mask = tf.equal(coords_hashed, cuboid_verts_hashed)
  cuboid_verts = tf.where(mask)[:, -1]

  # cuboid_verts is a tensor with shape [num_cuboids, 8], where each entry 
  # contains the indices of the 8 vertices in verts that belong to it
  # The ordering of the vertices are:
  # 0. left-up-front
  # 1. left-up-back
  # 2. left-down-front
  # 3. left-down-back
  # 4. right-up-front
  # 5. right-up-back
  # 6. right-down-front
  # 7. right-down-back
  cuboid_verts = tf.reshape(
      cuboid_verts, shape=[num_cuboids, NUM_VERTS_PER_CUBOID])
  
  # Map each cuboids face's vertex indices to match the actual index of the
  # vertex in cuboid_verts
  cuboid_range = tf.expand_dims(
      tf.repeat(tf.range(num_cuboids), [NUM_FACES_PER_CUBOID]), -1)
  cuboid_face_indices = tf.tile(UNIT_CUBOID_FACES, multiples=[num_cuboids, 1])
  cuboid_face_indices = tf.stack(
      [tf.concat(
          [cuboid_range, tf.expand_dims(cuboid_face_indices[:, 0], -1)], 1), 
       tf.concat(
          [cuboid_range, tf.expand_dims(cuboid_face_indices[:, 1], -1)], 1),
       tf.concat(
          [cuboid_range, tf.expand_dims(cuboid_face_indices[:, 2], -1)], 1)],
      axis=1)
  faces = tf.gather_nd(cuboid_verts, cuboid_face_indices)

  return verts, faces

def create_face_mask(voxels, kernel_weights, axis):
  conv_input = tf.expand_dims(voxels, axis=-1)

  mask = tf.nn.conv3d(conv_input, kernel_weights, [1, 1, 1, 1, 1], "VALID")
  mask = tf.cast(mask > 0.5, mask.dtype)

  mask = tf.squeeze(mask)
  batch_size, depth, height, width = tf.shape(voxels)

  if axis == 3:
    width -= 1
    concat_shape = [batch_size, depth, height, 1]
  elif axis == 2:
    height -= 1
    concat_shape = [batch_size, depth, 1, width]
  elif axis == 1:
    depth -= 1
    concat_shape = [batch_size, 1, height, width]

  mask = tf.reshape(mask, shape=[batch_size, depth, height, width])

  lower = tf.concat(
      [tf.ones(concat_shape),
       1-mask], axis=axis)
  upper = tf.concat(
      [1-mask,
       tf.ones(concat_shape)], axis=axis)
  
  lower *= voxels
  upper *= voxels

  return lower, upper


def cubify(voxels: tf.Tensor,
           thresh: float,
           align: str='topleft'):
  """Converts a voxel occupancy grid into a mesh.

  Args:
    voxels: A `Tensor` of shape [B, D, H, W] that contains the voxel occupancy
      prediction.
    thresh: A `float` that specifies the threshold value of a valid occupied
      voxel. 
    align: A string of either 'topleft', 'corner', or 'center'. Currently only
      'topleft' is supported, and is the default behavior.
  
  Returns:
    vertices:
    vertices_mask:
    faces:
    faces_mask:
  """
  batch_size, depth, _, _ = tf.shape(voxels)
  voxels = tf.cast(voxels > thresh, voxels.dtype)
  wz = tf.constant(value=0.5, shape=[2, 1, 1, 1, 1])
  wy = tf.constant(value=0.5, shape=[1, 2, 1, 1, 1])
  wx = tf.constant(value=0.5, shape=[1, 1, 2, 1, 1])

  z_front_updates, z_back_updates = create_face_mask(voxels, wz, axis=1)
  y_top_updates, y_bot_updates = create_face_mask(voxels, wy, axis=2)
  x_left_updates, x_right_updates = create_face_mask(voxels, wx, axis=3)

  updates = [
      x_left_updates,
      x_left_updates,

      y_bot_updates,
      y_bot_updates,

      z_front_updates,
      z_front_updates,
      
      y_top_updates,
      y_top_updates,

      x_right_updates,
      x_right_updates,

      z_back_updates,
      z_back_updates]
  
  faces_idx = tf.stack(updates, axis=0)

  # faces_idx cuboid order: (x, y, z) because this will be used to generate
  # the mask to mark valid faces. The faces were generated in x, y, z order
  faces_idx = tf.transpose(faces_idx, perm=[1, 4, 3, 2, 0])
  faces_idx = tf.reshape(faces_idx, [batch_size, -1, 12])
  # Boolean mask for valid faces
  faces_mask = tf.reshape(faces_idx, shape=[batch_size, -1])

  grid_dim = depth
  # All verts and faces in the mesh
  verts, faces = initialize_mesh(grid_dim)

  # Batch the initial verts and faces
  verts = tf.expand_dims(verts, axis=0)
  faces = tf.expand_dims(faces, axis=0)
  verts = tf.tile(verts, multiples=[batch_size, 1, 1])
  faces = tf.tile(faces, multiples=[batch_size, 1, 1])

  # Offset the faces so that each vertex index is positive
  masked_faces = faces + 1

  # Zero out unused faces
  masked_faces = masked_faces * tf.cast(
      tf.expand_dims(faces_mask, -1), faces.dtype) 

  # Create a mask based on whether or not a vertex appeared in any of the faces
  num_verts = (depth+1)**3
  verts_mask = tf.math.bincount(
      tf.cast(tf.reshape(masked_faces, [batch_size, -1]), tf.int32),
      minlength=num_verts+1, binary_output=True, axis=-1)
  
  # The first index was used for any 0s, which can be ignored
  verts_mask = verts_mask[:, 1:]

  return verts, faces, verts_mask, faces_mask