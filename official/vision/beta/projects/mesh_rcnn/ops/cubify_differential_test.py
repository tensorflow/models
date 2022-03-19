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
      voxels[loc[0]][loc[1]][loc[2]][loc[3]] += 0.5

    return voxels

  @parameterized.named_parameters(
      {'testcase_name': 'unit_mesh',
       'grid_dims': 2,
       'batch_size': 1,
       'occupancy_locs':
           [
               [0, 0, 0, 0]
           ]
      },

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
  def test_cubify_differential(self, grid_dims, batch_size, occupancy_locs):
    # Create identical voxels for both cubify functions
    voxels = self.create_np_voxels(grid_dims, batch_size, occupancy_locs)
    tf_voxels = tf.convert_to_tensor(voxels)
    torch_voxels = torch.tensor(voxels)

    # Run cubify
    torch_mesh = torch_cubify(torch_voxels, thresh=0.2, align='topleft')
    tf_mesh = tf_cubify(tf_voxels, thresh=0.2, align='topleft')

    # Extract the verts and faces from both meshes
    torch_verts = torch_mesh.verts_list()
    torch_verts = [v.cpu().detach().numpy() for v in torch_verts]
    torch_faces = torch_mesh.faces_list()
    torch_faces = [f.cpu().detach().numpy() for f in torch_faces]

    tf_all_verts = tf_mesh['verts'].numpy()
    tf_verts_mask = tf_mesh['verts_mask'].numpy()
    tf_verts = [v[m == 1] for v, m in zip(tf_all_verts, tf_verts_mask)]

    tf_all_faces = tf_mesh['faces'].numpy()
    tf_faces_mask = tf_mesh['faces_mask'].numpy()
    tf_faces = [f[m == 1] for f, m in zip(tf_all_faces, tf_faces_mask)]
    
    # Test each set of vertices in the batch
    for verts_1, verts_2 in zip(tf_verts, torch_verts):
      ind_1 = np.lexsort((verts_1[:, 0], verts_1[:, 1], verts_1[:, 2]))
      ind_2 = np.lexsort((verts_2[:, 0], verts_2[:, 1], verts_2[:, 2]))

      verts_1 = verts_1[ind_1]
      verts_2 = verts_2[ind_2]

      self.assertAllEqual(verts_1, verts_2)

    # Test the face coordinates in each batch
    for faces_1, faces_2, verts_1, verts_2 in zip(tf_faces, torch_faces, tf_all_verts, torch_verts):
      # At this point, the faces mask has been applied so only unique faces
      # exist. We can make 2 sets of faces, but replace the vertex indices
      # with the actual vertex coordinate. Then make sure that the sets are 
      # identical
      faces_to_vert_coords_1 = set()
      faces_to_vert_coords_2 = set()

      # Grab the coordinates that make up each face
      for f1, f2 in zip(faces_1, faces_2):
        coords_1 = tuple([tuple(verts_1[f1[0]]), tuple(verts_1[f1[1]]), tuple(verts_1[f1[2]])])
        faces_to_vert_coords_1.add(coords_1)

        coords_2 = tuple([tuple(verts_2[f2[0]]), tuple(verts_2[f2[1]]), tuple(verts_2[f2[2]])])
        faces_to_vert_coords_2.add(coords_2)

      self.assertSetEqual(faces_to_vert_coords_1, faces_to_vert_coords_2)
              
if __name__ == "__main__":
  tf.test.main()
