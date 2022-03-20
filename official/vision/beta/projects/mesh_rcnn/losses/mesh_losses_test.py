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

from typing import Any, Dict, List, Optional, Tuple
from absl.testing import parameterized

import tensorflow as tf

from official.vision.beta.projects.mesh_rcnn.losses.mesh_losses import (
    chamfer_loss, edge_loss, normal_loss)
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import (
    MeshSampler, compute_edges)
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import create_voxels

CUBIFY_THRESH = 0.5
ERR_TOL = 1e-7

shared_test_cases: List[Dict[str, Any]] = [
    {'testcase_name': 'batched_large_mesh_with_empty_samples',
     'grid_dims': 4,
     'batch_size': 4,
     'occupancy_locs':
         [
             [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 3],
             [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
             [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
             [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
         ]
    }
]
chamfer_and_normal_params: Dict[str, Any] = {
    'true_num_samples': 50,
    'pred_num_samples': 50,
    'weights_list': [None, tf.constant([1, 3, 0, 0.75])]
}
expected_loss_dict: Dict[str, dict] = {
    'chamfer': {
        'batched_large_mesh_with_empty_samples': [0.0849633664, 0.107685894]
    },
    'normal': {
        'batched_large_mesh_with_empty_samples': [1.10514593, 0.844003677]
    },
    'edge': {
        'batched_large_mesh_with_empty_samples': 0.486214489
    }
}
class MeshLossesTest(parameterized.TestCase, tf.test.TestCase):
  """Unit Test Mesh Loss Function(s)."""

  def _get_verts_and_faces(
      self, grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Create the ground truth and predicted meshes (verts, faces, and
       respective masks)"""
    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    mesh = cubify(voxels, thresh=CUBIFY_THRESH)

    verts = tf.cast(mesh['verts'], tf.float32)
    faces = tf.cast(mesh['faces'], tf.int32)
    verts_mask = tf.cast(mesh['verts_mask'], tf.int8)
    faces_mask = tf.cast(mesh['faces_mask'], tf.int8)

    # Randomly perturb the vert positions so we get more realistic losses.
    true_verts_perturb = tf.random.uniform(verts.shape, minval=-0.2, maxval=0.2,
                                           seed=1)
    pred_verts_perturb = tf.random.uniform(verts.shape, minval=-0.2, maxval=0.2,
                                           seed=2)

    true_verts = verts + true_verts_perturb
    pred_verts = verts + pred_verts_perturb

    return true_verts, pred_verts, verts_mask, faces, faces_mask

  def _get_pointclouds_and_normals(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      true_num_samples: int,
      pred_num_samples: int,
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Get the ground truth and predicted pointclouds (samples) and
       corresponding normals"""
    true_verts, pred_verts, verts_mask, faces, faces_mask = \
        self._get_verts_and_faces(grid_dims, batch_size, occupancy_locs)

    true_sampler = MeshSampler(true_num_samples)
    pred_sampler = MeshSampler(pred_num_samples)
    true_samples, true_normals, _ = true_sampler.sample_meshes(
        true_verts, verts_mask, faces, faces_mask,
    )
    pred_samples, pred_normals, _ = pred_sampler.sample_meshes(
        pred_verts, verts_mask, faces, faces_mask,
    )

    return true_samples, true_normals, pred_samples, pred_normals

  @parameterized.named_parameters(*[
      {**test, **chamfer_and_normal_params,
       'expected_losses': expected_loss_dict['chamfer'][test['testcase_name']]
      } for test in shared_test_cases
  ])
  def test_chamfer_loss_differential(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      true_num_samples: int,
      pred_num_samples: int,
      weights_list: List[Optional[tf.Tensor]],
      expected_losses: List[float],
  ) -> None:
    true_samples, _, pred_samples, _, = self._get_pointclouds_and_normals(
        grid_dims, batch_size, occupancy_locs,
        true_num_samples, pred_num_samples
    )

    for weights, expected_loss in zip(weights_list, expected_losses):
      actual_loss = chamfer_loss(
          true_samples, pred_samples, weights=weights
      )
      self.assertNear(actual_loss, expected_loss, err=ERR_TOL)

  @parameterized.named_parameters(*[
      {**test, **chamfer_and_normal_params,
       'expected_losses': expected_loss_dict['normal'][test['testcase_name']]
      } for test in shared_test_cases
  ])
  def test_normal_loss_differential(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      true_num_samples: int,
      pred_num_samples: int,
      weights_list: List[Optional[tf.Tensor]],
      expected_losses: List[float],
  ) -> None:
    (true_samples, true_normals, pred_samples, pred_normals
    ) = self._get_pointclouds_and_normals(grid_dims,
                                          batch_size,
                                          occupancy_locs,
                                          true_num_samples,
                                          pred_num_samples)

    for weights, expected_loss in zip(weights_list, expected_losses):
      actual_loss = normal_loss(
          true_samples, pred_samples, true_normals,
          pred_normals, weights=weights
      )
      self.assertNear(actual_loss, expected_loss, err=ERR_TOL)

  @parameterized.named_parameters(*[
      {**test,
       'expected_loss': expected_loss_dict['edge'][test['testcase_name']]
      } for test in shared_test_cases
  ])
  def test_edge_loss_differential(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      expected_loss: float,
  ) -> None:
    _, verts, verts_mask, faces, faces_mask = self._get_verts_and_faces(
        grid_dims, batch_size, occupancy_locs
    )
    edges, edges_mask = compute_edges(faces, faces_mask)

    actual_loss = edge_loss(verts, verts_mask, edges, edges_mask)

    self.assertNear(actual_loss, expected_loss, err=ERR_TOL)

if __name__ == "__main__":
  tf.random.set_seed(1)
  tf.test.main()
