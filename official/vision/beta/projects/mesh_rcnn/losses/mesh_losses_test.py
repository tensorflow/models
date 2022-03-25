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

"""Tests for Mesh Losses."""

from typing import List, Optional, Tuple
from absl.testing import parameterized

import tensorflow as tf

from official.vision.beta.projects.mesh_rcnn.losses.mesh_losses import (
    MeshLoss, chamfer_loss, edge_loss, normal_loss)
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import compute_edges
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import create_voxels

CUBIFY_THRESH = 0.5
ERR_TOL = 1e-6

true_pointcloud = {
    'samples':
        [
            [[-0.194013268, -1.01414645, -0.631096303],
             [-0.601892471, -0.0221258588, -0.379562378],
             [-1.02934957, -0.85288167, -0.945664],
             [-0.0156115796, -1.0473069, -0.542341053],
             [0.959001601, -0.0720413253, -0.790964246]],

            [[0.115194112, 0.428069532, 0.798253298],
             [-0.0625587329, -0.0602969117, -0.976883173],
             [-0.898269057, 0.626779079, 1.10757041],
             [-1.00582552, -0.325456798, -0.169963032],
             [0.920479834, -0.813576579, 0.466118306]],

            [[0, 0, 0] for i in range(5)],
        ],
    'normals':
        [
            [[-0.116013758, -0.984100878, -0.134485632],
             [0.00226397254, 0.999980152, -0.0058719418],
             [-0.248364881, -0.968637884, -0.00745504908],
             [-0.116013758, -0.984100878, -0.134485632],
             [0.981007874, 0.0940456912, -0.169643626]],

            [[0.933110416, 0.303228378, 0.193280756],
             [-0.3285456, 0.178630605, -0.927442133],
             [0.315824628, 0.0186190233, 0.948634923],
             [-0.995064318, -0.0853243172, 0.05066403],
             [0.960352242, 0.0544978492, 0.273411155]],

            [[0, 0, 0] for i in range(5)],
        ]
}
pred_pointcloud = {
    'samples':
        [
            [[0.229866192, -0.957541525, -0.223863363],
             [-1.01462829, -0.937355816, -0.0296658948],
             [0.79906404, -0.0858851224, -0.427685171],
             [0.439470291, -1.00555849, -0.77687192],
             [0.951509595, -0.815054476, -1.08535147]],

            [[1.08359015, -0.307075381, 0.943411946],
             [-1.08097625, -0.0384020805, 0.197004],
             [-0.913895428, -1.16104865, -0.950149834],
             [0.582187772, -0.387396961, 1.11741817],
             [0.667381763, -0.0281033106, 0.449858427]],

            [[0, 0, 0] for i in range(5)],
        ],
    'normals':
        [
            [[0.0500510931, -0.996487379, 0.0671402588],
             [-0.996427178, -0.0723610222, 0.0435518362],
             [0.12468347, 0.991397, 0.039824117],
             [-0.113386542, -0.961690903, 0.249587893],
             [0.941307545, -0.260884136, -0.214195102]],

            [[0.985631943, -0.0592056066, -0.158190921],
             [-0.995837271, -0.085104771, 0.0326405838],
             [0.0747440904, -0.960672, 0.267437339],
             [-0.00552498689, -0.0495253392, 0.998757601],
             [0.0723976716, 0.997179806, 0.0197746]],

            [[0, 0, 0] for i in range(5)],
        ],
}

class MeshLossesTest(parameterized.TestCase, tf.test.TestCase):
  """Unit Test Mesh Loss Function(s)."""

  def _get_voxels_verts_and_faces(
      self, grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Create the ground truth and predicted meshes: verts, faces, and
       respective masks."""
    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    mesh = cubify(voxels, thresh=CUBIFY_THRESH)

    verts = tf.cast(mesh['verts'], tf.float32)
    faces = tf.cast(mesh['faces'], tf.int32)
    verts_mask = tf.cast(mesh['verts_mask'], tf.int8)
    faces_mask = tf.cast(mesh['faces_mask'], tf.int8)

    return voxels, verts, verts_mask, faces, faces_mask

  @parameterized.named_parameters({
      'testcase_name': 'batched_mesh_with_empty_samples',
      'true_samples': true_pointcloud['samples'],
      'pred_samples': pred_pointcloud['samples'],
      'weights_list': [None, tf.constant([0.75, 2, 0.1])],
      'expected_losses': [0.849560857, 1.37367523],
  })
  def test_chamfer_loss(
      self,
      true_samples: List[List[List[float]]],
      pred_samples: List[List[List[float]]],
      weights_list: List[Optional[tf.Tensor]],
      expected_losses: List[float],
  ) -> None:
    true_samples = tf.convert_to_tensor(true_samples, tf.float32)
    pred_samples = tf.convert_to_tensor(pred_samples, tf.float32)

    for weights, expected_loss in zip(weights_list, expected_losses):
      actual_loss = chamfer_loss(
          true_samples, pred_samples, weights=weights
      )
      self.assertNear(actual_loss, expected_loss, err=ERR_TOL)

  @parameterized.named_parameters({
      'testcase_name': 'batched_mesh_with_empty_samples',
      'true_samples': true_pointcloud['samples'],
      'true_normals': true_pointcloud['normals'],
      'pred_samples': pred_pointcloud['samples'],
      'pred_normals': pred_pointcloud['normals'],
      'weights_list': [None, tf.constant([0.75, 2, 0.1])],
      'expected_losses': [1.23676932, 0.909479141],
  })
  def test_normal_loss(
      self,
      true_samples: List[List[List[float]]],
      true_normals: List[List[List[float]]],
      pred_samples: List[List[List[float]]],
      pred_normals: List[List[List[float]]],
      weights_list: List[Optional[tf.Tensor]],
      expected_losses: List[float],
  ) -> None:
    true_samples = tf.convert_to_tensor(true_samples, tf.float32)
    true_normals = tf.convert_to_tensor(true_normals, tf.float32)
    pred_samples = tf.convert_to_tensor(pred_samples, tf.float32)
    pred_normals = tf.convert_to_tensor(pred_normals, tf.float32)

    for weights, expected_loss in zip(weights_list, expected_losses):
      actual_loss = normal_loss(
          true_samples, pred_samples, true_normals,
          pred_normals, weights=weights
      )
      self.assertNear(actual_loss, expected_loss, err=ERR_TOL)

  @parameterized.named_parameters({
      'testcase_name': 'batched_large_mesh_with_empty_samples',
      'grid_dims': 4,
      'batch_size': 4,
      'occupancy_locs':
          [
              [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 3],
              [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
              [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
              [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
          ],
      'expected_loss': 0.444444537,
  })
  def test_edge_loss(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      expected_loss: float,
  ) -> None:
    _, verts, verts_mask, faces, faces_mask = self._get_voxels_verts_and_faces(
        grid_dims, batch_size, occupancy_locs
    )
    edges, edges_mask = compute_edges(faces, faces_mask)

    actual_loss = edge_loss(verts, verts_mask, edges, edges_mask)

    self.assertNear(actual_loss, expected_loss, err=ERR_TOL)

  @parameterized.named_parameters({
      'testcase_name': 'batched_large_mesh_with_empty_samples',
      'grid_dims': 4,
      'batch_size': 4,
      'voxel_weight': 1.0,
      'chamfer_weight': 1.0,
      'normal_weight': 1.0,
      'edge_weight': 1.0,
      'true_num_samples': 50,
      'pred_num_samples': 50,
      'true_occupancy_locs':
          [
              [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 3],
              [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
              [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
              [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
          ],
      'pred_occupancy_locs':
          [
              [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 3],
              [1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1],
              [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
              [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 2, 1, 1],
          ],
  })
  def test_mesh_loss(
      self,
      grid_dims: int,
      batch_size: int,
      voxel_weight: float,
      chamfer_weight: float,
      normal_weight: float,
      edge_weight: float,
      true_num_samples: int,
      pred_num_samples: int,
      true_occupancy_locs: List[List[int]],
      pred_occupancy_locs: List[List[int]],
  ) -> None:

    voxels_true, verts_true, verts_mask_true, faces_true, faces_mask_true = \
        self._get_voxels_verts_and_faces(grid_dims,
                                         batch_size,
                                         true_occupancy_locs)
    meshes_true = {
        'verts': verts_true,
        'verts_mask': verts_mask_true,
        'faces': faces_true,
        'faces_mask': faces_mask_true,
    }
    voxels_pred, verts_pred, verts_mask_pred, faces_pred, faces_mask_pred = \
        self._get_voxels_verts_and_faces(grid_dims,
                                         batch_size,
                                         pred_occupancy_locs)
    meshes_pred = {
        'verts': verts_pred,
        'verts_mask': verts_mask_pred,
        'faces': faces_pred,
        'faces_mask': faces_mask_pred,
    }
    edges_pred, edges_mask_pred = compute_edges(faces_pred, faces_mask_pred)
    mesh_loss = MeshLoss(voxel_weight,
                         chamfer_weight,
                         normal_weight,
                         edge_weight,
                         true_num_samples,
                         pred_num_samples)

    total_loss, voxel_loss, chamfer_loss_, normal_loss_, edge_loss_ = \
        mesh_loss(
            voxels_true,
            voxels_pred,
            meshes_true,
            meshes_pred,
            edges_pred,
            edges_mask_pred,
        )

    self.assertGreaterEqual(total_loss, 0)
    self.assertGreaterEqual(voxel_loss, 0)
    self.assertGreaterEqual(chamfer_loss_, 0)
    self.assertGreaterEqual(normal_loss_, 0)
    self.assertGreaterEqual(edge_loss_, 0)

if __name__ == "__main__":
  tf.random.set_seed(1)
  tf.test.main()
