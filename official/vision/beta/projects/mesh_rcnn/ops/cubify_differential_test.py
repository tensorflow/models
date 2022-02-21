"""Differential Test for Cubify."""

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized
from pytorch3d.ops import cubify as torch_cubify

from official.vision.beta.projects.mesh_rcnn.ops.cubify import \
    cubify as tf_cubify


class CubifyDifferentialTest(parameterized.TestCase, tf.test.TestCase):
  """Differential Test for Cubify."""

  def create_np_voxels(self, grid_dims, batch_size, occupancy_locs):
    voxels = np.zeros(shape=[batch_size, grid_dims, grid_dims, grid_dims],
                      dtype=np.float32)
    for loc in occupancy_locs:
      voxels[loc[0]][loc[1]][loc[2]][loc[3]] += 1

    return voxels

  @parameterized.named_parameters(
      {'testcase_name': 'batched_large_mesh_with_empty_samples',
       'grid_dims': 5,
       'batch_size': 5,
       'occupancy_locs':
           [
               [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 3],
               [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
               [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
               [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
           ]
      }
  )
  def test_cubify_differential(self, grid_dims, batch_size, occupancy_locs):
    # Create identical voxels for both cubify functions
    voxels = self.create_np_voxels(grid_dims, batch_size, occupancy_locs)
    tf_voxels = tf.convert_to_tensor(voxels)
    torch_voxels = torch.tensor(voxels)

    # Run cubify
    torch_mesh = torch_cubify(torch_voxels, thresh=0.5)
    tf_mesh = tf_cubify(tf_voxels, thresh=0.5)

    # Extract the verts and faces from both meshes
    torch_verts = torch_mesh.verts_list()
    torch_verts = [v.cpu().detach().numpy() for v in torch_verts]
    torch_faces = torch_mesh.faces_list()
    torch_faces = [f.cpu().detach().numpy() for f in torch_faces]

    tf_all_verts = tf_mesh[0].numpy()
    tf_verts_mask = tf_mesh[2].numpy()
    tf_verts = [v[m == 1] for v, m in zip(tf_all_verts, tf_verts_mask)]

    tf_all_faces = tf_mesh[1].numpy()
    tf_faces_mask = tf_mesh[3].numpy()
    tf_faces = [f[m == 1] for f, m in zip(tf_all_faces, tf_faces_mask)]

    # Test each set of vertices in the batch
    for verts_1, verts_2 in zip(tf_verts, torch_verts):
      ind_1 = np.lexsort((verts_1[:, 0], verts_1[:, 1], verts_1[:, 2]))
      ind_2 = np.lexsort((verts_2[:, 0], verts_2[:, 1], verts_2[:, 2]))
      verts_1 = verts_1[ind_1]
      verts_2 = verts_2[ind_2]
      self.assertAllEqual(verts_1, verts_2)

    # Faces not differentially tested because pytorch3d has inconsistent
    # top and bottom cube faces

if __name__ == "__main__":
  tf.test.main()
