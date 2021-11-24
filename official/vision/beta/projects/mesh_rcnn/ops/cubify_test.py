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

"""Test for meshes and mesh operations."""

import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.cubify import initialize_mesh
from official.vision.beta.projects.mesh_rcnn.ops.cubify import generate_3d_coords
from official.vision.beta.projects.mesh_rcnn.ops.cubify import hash_flatenned_3d_coords
from official.vision.beta.projects.mesh_rcnn.ops.cubify import unhash_flattened_3d_coords

class CubifyTest(parameterized.TestCase, tf.test.TestCase):
  def create_voxels(self, grid_dims, batch_size, occupancy_locs):
    ones = tf.ones(shape=[len(occupancy_locs)])
    voxels = tf.scatter_nd(
        indices=tf.convert_to_tensor(occupancy_locs, tf.int32),
        updates=ones,
        shape=[batch_size, grid_dims, grid_dims, grid_dims])
      
    return voxels

  @parameterized.named_parameters(
    {'testcase_name': 'unit_coord',
    'coord_dim': (1, 1, 1),
    'perform_flatten': False},
    {'testcase_name': 'equisized_cords',
    'coord_dim': (5, 5, 5),
    'perform_flatten': False},
    {'testcase_name': 'unequisized_cords',
    'coord_dim': (3, 4, 5),
    'perform_flatten': False},
    {'testcase_name': 'flatten_unequisized_cords',
    'coord_dim': (3, 4, 5),
    'perform_flatten': True}
  )
  def test_generate_3d_coords(self, coord_dim, perform_flatten):
    output = generate_3d_coords(coord_dim[0], coord_dim[1], coord_dim[2], perform_flatten)

    self.assertEqual(tf.reduce_max(output), max(coord_dim))
    self.assertEqual(tf.reduce_min(output), 0)

    if perform_flatten:
      self.assertAllEqual(
          tf.shape(output), 
          [(coord_dim[0]+1) * (coord_dim[1]+1) * (coord_dim[2]+1), 3])
      
      i = 0
      for x in range(coord_dim[0]+1):
        for y in range(coord_dim[1]+1):
          for z in range(coord_dim[2]+1):
            self.assertAllEqual(output[i], [x, y, z])
            i += 1
      
    else:
      self.assertAllEqual(
          tf.shape(output), [coord_dim[0]+1, coord_dim[1]+1, coord_dim[2]+1, 3]
      )
      for x in range(coord_dim[0]+1):
        for y in range(coord_dim[1]+1):
          for z in range(coord_dim[2]+1):
            self.assertAllEqual(output[x, y, z], [x, y, z])

  @parameterized.named_parameters(
    {'testcase_name': 'hash_and_unhash',
     'tensor_length': 100,
     'max_value': 100},
  )
  def test_hash_3d_coords(self, tensor_length, max_value):
    self.skipTest("skipping")
    input_tensor = tf.random.uniform([tensor_length, 3], 0, max_value, dtype=tf.int64)
    hashed_tensor = hash_flatenned_3d_coords(input_tensor, max_value)
    self.assertEqual(len(tf.shape(hashed_tensor)), 1)
    self.assertEqual(tf.shape(hashed_tensor)[0], tf.shape(input_tensor)[0])

    unhashed_tensor = unhash_flattened_3d_coords(hashed_tensor, max_value)
    self.assertAllEqual(unhashed_tensor, input_tensor)


  @parameterized.named_parameters(
      {'testcase_name': 'unit_mesh',
       'grid_dim': 1},
      {'testcase_name': 'large_mesh',
       'grid_dim': 24}
  )
  def test_initialize_mesh(self, grid_dim):
    verts, faces = initialize_mesh(grid_dim)
  
    self.assertAllEqual(tf.shape(verts), [(grid_dim+1) ** 3, 3])
    self.assertAllEqual(tf.shape(faces), [12*(grid_dim) ** 3, 3])

  @parameterized.named_parameters(
      {'testcase_name': 'unit_mesh',
       'grid_dims': 1,
       'batch_size': 1,
       'occupancy_locs': 
           [
               [0, 0, 0, 0] # coordinates given in (z, y, x)
           ],
       'expected_num_verts': [8],
       'expected_num_faces': [12],
      },
      {'testcase_name': 'batched_unit_mesh',
       'grid_dims': 1,
       'batch_size': 2,
       'occupancy_locs': 
           [
               [0, 0, 0, 0],
               [1, 0, 0, 0]
           ],
       'expected_num_verts': [8, 8],
       'expected_num_faces': [12, 12]},
      {'testcase_name': 'batched_small_mesh',
       'grid_dims': 2,
       'batch_size': 2,
       'occupancy_locs': 
           [
               [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
               [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
           ],
       'expected_num_verts': [16, 20],
       'expected_num_faces': [28, 36]},
      {'testcase_name': 'batched_large_mesh_with_empty_samples',
       'grid_dims': 10,
       'batch_size': 5,
       'occupancy_locs': 
           [
               [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
               [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
               [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
               [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
           ],
       'expected_num_verts': [16, 20, 0, 26, 0],
       'expected_num_faces': [28, 36, 0, 48, 0]}
  )
  def test_cubify(self, grid_dims, batch_size, occupancy_locs, expected_num_faces, expected_num_verts):
    voxels = self.create_voxels(grid_dims, batch_size, occupancy_locs)
    verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)
  
    self.assertAllEqual(tf.shape(verts), [batch_size, (grid_dims+1)**3, 3])
    self.assertAllEqual(tf.shape(faces), [batch_size, (grid_dims**3)*12, 3])
    self.assertAllEqual(tf.shape(verts_mask), [batch_size, (grid_dims+1)**3])
    self.assertAllEqual(tf.shape(faces_mask), [batch_size, (grid_dims**3)*12])

    verts_mask_list = tf.unstack(verts_mask)
    faces_mask_list = tf.unstack(faces_mask)

    for i, v in enumerate(verts_mask_list):
      self.assertEqual(tf.reduce_sum(v), expected_num_verts[i])
    for i, f in enumerate(faces_mask_list):
      self.assertEqual(tf.reduce_sum(f), expected_num_faces[i])
    
if __name__ == "__main__":
  tf.test.main()
