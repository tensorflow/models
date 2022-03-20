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

"""Differential Tests for Mesh Losses."""

from typing import List, Optional, Tuple
from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import torch

from pytorch3d.loss import chamfer_distance as torch_chamfer_and_normal_loss
from pytorch3d.loss import mesh_edge_loss as torch_edge_loss
from pytorch3d.structures import Meshes as TorchMeshes

from official.vision.beta.projects.mesh_rcnn.losses.mesh_losses import \
    chamfer_loss as tf_chamfer_loss
from official.vision.beta.projects.mesh_rcnn.losses.mesh_losses import \
    edge_loss as tf_edge_loss
from official.vision.beta.projects.mesh_rcnn.losses.mesh_losses import \
    normal_loss as tf_normal_loss
from official.vision.beta.projects.mesh_rcnn.ops.cubify import \
    cubify as tf_cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
    MeshSampler as TfMeshSampler
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import \
    compute_edges as tf_compute_edges
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import \
    create_voxels as tf_create_voxels

CUBIFY_THRESH = 0.5

shared_test_cases = [
    {'testcase_name': 'batched_large_mesh_with_empty_samples',
     'grid_dims': 4,
     'batch_size': 5,
     'occupancy_locs':
         [
             [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 3],
             [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1],
             [3, 0, 0, 0], [3, 0, 0, 1], [3, 0, 1, 0], [3, 0, 1, 1],
             [3, 1, 0, 0], [3, 1, 0, 1], [3, 1, 1, 0], [3, 1, 1, 1],
         ]
    }
]
chamfer_and_normal_params = {
    'true_num_samples': 5000,
    'pred_num_samples': 5000,
    'weights_list': [None, np.array([1, 3, 0, 0.75, 0.5])]
}

class MeshLossesDifferentialTest(parameterized.TestCase, tf.test.TestCase):
  """Differential Test Mesh Loss Function(s)."""

  def _get_tf_verts_and_faces(
      self, grid_dims: int, batch_size: int, occupancy_locs: List[List[int]]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Create the ground truth and predicted meshes (verts and faces)
    # using the TF cubify.
    voxels = tf_create_voxels(grid_dims, batch_size, occupancy_locs)
    mesh = tf_cubify(voxels, thresh=CUBIFY_THRESH)

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
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
             tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Get the ground truth and predicted pointclouds (samples) and
    # corresponding normals using the TF cubify and TF mesh sampler.
    true_verts, pred_verts, verts_mask, faces, faces_mask = \
        self._get_tf_verts_and_faces(grid_dims, batch_size, occupancy_locs)

    true_sampler = TfMeshSampler(true_num_samples)
    pred_sampler = TfMeshSampler(pred_num_samples)
    tf_true_samples, tf_true_normals, _ = true_sampler.sample_meshes(
        true_verts, verts_mask, faces, faces_mask,
    )
    tf_pred_samples, tf_pred_normals, _ = pred_sampler.sample_meshes(
        pred_verts, verts_mask, faces, faces_mask,
    )

    # Create the torch pointclouds and normals.
    torch_true_samples = torch.tensor(tf_true_samples.numpy())
    torch_true_normals = torch.tensor(tf_true_normals.numpy())
    torch_pred_samples = torch.tensor(tf_pred_samples.numpy())
    torch_pred_normals = torch.tensor(tf_pred_normals.numpy())

    return (tf_true_samples, tf_true_normals, tf_pred_samples, tf_pred_normals,
            torch_true_samples, torch_true_normals, torch_pred_samples,
            torch_pred_normals)

  @parameterized.named_parameters(*[
      {**test, **chamfer_and_normal_params} for test in shared_test_cases
  ])
  def test_chamfer_loss_differential(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      true_num_samples: int,
      pred_num_samples: int,
      weights_list: List[Optional[np.array]],
  ) -> None:
    tf_true_samples, _, tf_pred_samples, _, torch_true_samples, _, \
    torch_pred_samples, _ = self._get_pointclouds_and_normals(
        grid_dims, batch_size, occupancy_locs,
        true_num_samples, pred_num_samples
    )

    for weights in weights_list:
      tf_weights, torch_weights = None, None
      if weights is not None:
        tf_weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        torch_weights = torch.tensor(weights, dtype=torch.float32)

      tf_cham_dist = tf_chamfer_loss(
          tf_true_samples, tf_pred_samples, weights=tf_weights
      )
      torch_cham_dist, _ = torch_chamfer_and_normal_loss(
          torch_true_samples, torch_pred_samples, weights=torch_weights
      )
      self.assertNear(tf_cham_dist, torch_cham_dist, err=1e-5)

  @parameterized.named_parameters(*[
      {**test, **chamfer_and_normal_params} for test in shared_test_cases
  ])
  def test_normal_loss_differential(
      self,
      grid_dims: int,
      batch_size: int,
      occupancy_locs: List[List[int]],
      true_num_samples: int,
      pred_num_samples: int,
      weights_list: List[Optional[np.array]],
  ) -> None:
    tf_true_samples, tf_true_normals, tf_pred_samples, tf_pred_normals, \
    torch_true_samples, torch_true_normals, torch_pred_samples,         \
    torch_pred_normals = self._get_pointclouds_and_normals(
        grid_dims, batch_size, occupancy_locs,
        true_num_samples, pred_num_samples
    )

    for weights in weights_list:
      tf_weights, torch_weights = None, None
      if weights is not None:
        tf_weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        torch_weights = torch.tensor(weights, dtype=torch.float32)

      tf_normal_dist = tf_normal_loss(
          tf_true_samples, tf_pred_samples, tf_true_normals,
          tf_pred_normals, weights=tf_weights
      )
      _, torch_normal_dist = torch_chamfer_and_normal_loss(
          torch_true_samples, torch_pred_samples, x_normals=torch_true_normals,
          y_normals=torch_pred_normals, weights=torch_weights
      )
      self.assertNear(tf_normal_dist, torch_normal_dist, err=1e-5)

  @parameterized.named_parameters(*shared_test_cases)
  def test_edge_loss_differential(
      self, grid_dims: int, batch_size: int, occupancy_locs: List[List[int]],
  ) -> None:
    ## Get the TF meshes (verts, faces, and edges tensors).
    _, tf_verts, tf_verts_mask, tf_faces, tf_faces_mask = \
        self._get_tf_verts_and_faces(grid_dims, batch_size, occupancy_locs)
    tf_edges, tf_edges_mask = tf_compute_edges(tf_faces, tf_faces_mask)

    ## Create the equivalent pytorch3d `Meshes` instance.
    np_verts = tf_verts.numpy()
    np_verts_mask = tf_verts_mask.numpy()
    np_faces = tf_faces.numpy()
    np_faces_mask = tf_faces_mask.numpy()

    torch_faces_list = [torch.tensor(f[m == 1]) for f, m, in
                        zip(np_faces, np_faces_mask)]
    torch_faces_updates = [torch.zeros(t.shape, dtype=torch.int32)
                           for t in torch_faces_list]

    # Compute `torch_verts_list` and `torch_faces_updates`.
    torch_verts_list = []
    # pylint: disable=too-many-nested-blocks
    for batch_idx, (verts, verts_masks) in enumerate(
        zip(np_verts, np_verts_mask)
    ):
      # `masked_verts` holds all the vertices that are not masked off/away for
      # the current mesh (batch element).
      masked_verts = []
      for vert_idx, (vert, mask) in enumerate(zip(verts, verts_masks)):
        if mask:
          masked_verts.append(vert)
        else:
          # Subtract one from a vert index (indices within each face) every time
          # that index is greater than the index of a vert that was masked off.
          for face_idx in range(torch_faces_list[batch_idx].shape[0]):
            for face_vert_idx in range(torch_faces_list[batch_idx].shape[1]):
              cur = torch_faces_list[batch_idx][face_idx, face_vert_idx]
              if cur > vert_idx:
                torch_faces_updates[batch_idx][face_idx, face_vert_idx] -= 1
      torch_verts_list.append(
          torch.tensor(masked_verts) if masked_verts else torch.ones((0, 3))
      )
    # Correct the vert indices based on the upate tensor.
    torch_faces_list = [orig + update for orig, update in
                        zip(torch_faces_list, torch_faces_updates)]
    torch_meshes = TorchMeshes(verts=torch_verts_list, faces=torch_faces_list)

    ## Compare the edge losses.
    tf_loss = tf_edge_loss(tf_verts, tf_verts_mask, tf_edges, tf_edges_mask)
    torch_loss = torch_edge_loss(torch_meshes)

    self.assertNear(tf_loss, torch_loss, err=1e-5)

if __name__ == "__main__":
  tf.random.set_seed(1)
  tf.test.main()
