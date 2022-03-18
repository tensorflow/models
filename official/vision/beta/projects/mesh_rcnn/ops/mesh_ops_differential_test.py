"""Differential Test for Mesh Ops"""

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized
from pytorch3d.ops import vert_align as torch_vert_align
from pytorch3d.ops import cubify as torch_cubify

from collections import Counter

from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
    vert_align as tf_vert_align

from official.vision.beta.projects.mesh_rcnn.ops.cubify import \
    cubify as tf_cubify


from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import compute_edges

class MeshOpsDifferentialTest(parameterized.TestCase, tf.test.TestCase):

  def create_np_voxels(self, grid_dims, batch_size, occupancy_locs):
    voxels = np.zeros(shape=[batch_size, grid_dims, grid_dims, grid_dims],
                      dtype=np.float32)
    for loc in occupancy_locs:
      voxels[loc[0]][loc[1]][loc[2]][loc[3]] += 0.5

    return voxels

  def test_vert_align(self):
    verts = np.random.uniform(low=-2.0, high=2.0, size=(2, 5, 3))
    feature_map = np.random.uniform(low=-1.0, high=1.0, size=(2, 10, 10, 15))

    tf_verts = tf.convert_to_tensor(verts, tf.float32)
    torch_verts = torch.as_tensor(verts)

    tf_feature_map = tf.convert_to_tensor(feature_map, tf.float32)
    torch_feature_map = torch.as_tensor(feature_map)
    torch_feature_map = torch.permute(torch_feature_map, (0, 3, 1, 2))

    torch_outputs = torch_vert_align(
        torch_feature_map, torch_verts, padding_mode='border')
    tf_outputs = tf_vert_align(tf_feature_map, tf_verts)

    torch_outputs = torch_outputs.numpy()
    tf_outputs = tf_outputs.numpy()

    self.assertAllClose(torch_outputs, tf_outputs)
  
  @parameterized.named_parameters(
      {'testcase_name': 'batched_mesh',
       'grid_dims': 2,
       'batch_size': 4,
       'occupancy_locs':
           [
               [0, 0, 0, 0],
               [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], 
               [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
               [3, 0, 0, 1], [3, 0, 1, 1], [3, 1, 0, 1], [3, 1, 1, 1], 
               [3, 1, 1, 0], 
           ]
      }
  )
  def test_compute_edges(self, grid_dims, batch_size, occupancy_locs):
    voxels = self.create_np_voxels(grid_dims, batch_size, occupancy_locs)
    tf_voxels = tf.convert_to_tensor(voxels)
    torch_voxels = torch.tensor(voxels)

    # Run cubify
    torch_mesh = torch_cubify(torch_voxels, thresh=0.2, align='topleft')
    tf_mesh = tf_cubify(tf_voxels, thresh=0.2, align='topleft')

    tf_verts = tf_mesh['verts'].numpy()
    tf_faces = tf_mesh['faces']
    tf_faces_mask = tf_mesh['faces_mask']

    torch_verts = torch_mesh.verts_packed()
    torch_verts = torch_verts.detach().numpy()

    tf_edges, tf_edges_mask = compute_edges(tf_faces, tf_faces_mask)
    packed_torch_edges = torch_mesh.edges_packed().cpu().detach().numpy()
    num_edges_per_mesh = torch_mesh.num_edges_per_mesh()

    tf_edges = tf_edges.numpy()
    tf_edges_mask = tf_edges_mask.numpy()
    tf_edges = [e[m == 1] for e, m in zip(tf_edges, tf_edges_mask)]

    torch_edges = []
    start_idx = 0
    for i in range(batch_size):
      torch_edges.append(packed_torch_edges[start_idx:start_idx+num_edges_per_mesh[i]])
      start_idx += num_edges_per_mesh[i]
    
    for edges_1, edges_2, verts_1 in zip(tf_edges, torch_edges, tf_verts):
      edges_to_vert_coords_1 = set()
      edges_to_vert_coords_2 = set()

      for e1 in edges_1:
        coords_1 = tuple([tuple(verts_1[e1[0]]), tuple(verts_1[e1[1]])])
        edges_to_vert_coords_1.add(coords_1)
      
      for e2 in edges_2:
        coords_2 = tuple([tuple(torch_verts[e2[0]]), tuple(torch_verts[e2[1]])])
        edges_to_vert_coords_2.add(coords_2)
      
      self.assertSetEqual(edges_to_vert_coords_1, edges_to_vert_coords_2)

if __name__ == "__main__":
  tf.test.main()
