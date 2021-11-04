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

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized
from pytorch3d.structures import Meshes as PyTorchMeshes

from official.vision.beta.projects.mesh_rcnn.ops.meshes import Meshes


class MeshesTest(parameterized.TestCase, tf.test.TestCase):
  """Meshes Tests"""

  def _create_meshes(self, num_verts_per_mesh: List[int],
                     faces: List[List[List[int]]])-> Tuple[Meshes,
                                                           PyTorchMeshes]:
    """Helper function that creates TF and PyTorch meshes.

    Args:
      num_verts_per_mesh: `List` of integers for the number of vertices in each
          mesh to be created
      faces: `List` of faces for each mesh to be created.
    Returns:
      `Tuple` where the first element is the TF Meshes object (ours) and the
          second element is the PyTorch Meshes object.
    """
    verts_list = []
    for n in num_verts_per_mesh:
      verts_list.append(np.random.uniform(low=-1, high=1, size=[n, 3]))

    faces_list = []
    for mesh_face_list in faces:
      faces_list.append(np.array(mesh_face_list, dtype=np.int32))

    tf_verts = list(
        map(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), verts_list))
    tf_faces = list(
        map(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), faces_list))

    torch_verts = list(map(torch.from_numpy, verts_list))
    torch_faces = list(map(torch.from_numpy, faces_list))

    tf_meshes = Meshes(verts_list=tf_verts, faces_list=tf_faces)

    torch_meshes = PyTorchMeshes(verts=torch_verts, faces=torch_faces)
    return tf_meshes, torch_meshes

  @parameterized.named_parameters(
      {'testcase_name': 'single-mesh-1-face',
       'num_verts_per_mesh': [3],
       'faces': [[[0, 1, 2]]]},
      {'testcase_name': 'multiple-meshes-same-number-verts',
       'num_verts_per_mesh': [3, 3, 3],
       'faces': [[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]]},
      {'testcase_name': 'multiple-meshes-different-number-verts',
       'num_verts_per_mesh': [3, 4, 5],
       'faces': [[[0, 1, 2]],
                 [[0, 1, 2], [1, 2, 3]],
                 [[0, 1, 2], [1, 2, 3], [3, 4, 0]]]}
  )
  def test_meshes(self, num_verts_per_mesh: List[int],
                  faces: List[List[List[int]]]):
    """Runs differential tests for meshes.

    Args:
      num_verts_per_mesh: `List` of integers for the number of vertices in each
          mesh to be created
      faces: `List` of faces for each mesh to be created.
    """
    tf_meshes, torch_meshes = self._create_meshes(num_verts_per_mesh, faces)
    self._test_meshes_initialization(tf_meshes, torch_meshes)
    self._test_meshes_packed_representation(tf_meshes, torch_meshes)

  def _test_meshes_initialization(self, tf_meshes: Meshes,
                                  torch_meshes: PyTorchMeshes):
    """Tests initialized state of meshes class.

    Args:
      tf_meshes: TF implementation of Meshes object.
      torch_meshes: PyTorch implementation of Meshes object.
    """
    self.assertAllEqual(
        tf_meshes.num_verts_per_mesh,
        torch_meshes.num_verts_per_mesh())
    self.assertAllEqual(
        tf_meshes.num_faces_per_mesh,
        torch_meshes.num_faces_per_mesh())

    self.assertEqual(tf_meshes.batch_size, len(torch_meshes))
    self.assertEqual(tf_meshes.max_verts_per_mesh, torch_meshes._V) # pylint: disable=protected-access
    self.assertEqual(tf_meshes.max_faces_per_mesh, torch_meshes._F) # pylint: disable=protected-access

  def _test_meshes_packed_representation(self, tf_meshes: Meshes,
                                         torch_meshes: PyTorchMeshes):
    """Tests packed representation of meshes class.

    Args:
      tf_meshes: TF implementation of Meshes object.
      torch_meshes: PyTorch implementation of Meshes object.
    """
    # Check that the packed representations have the same values
    self.assertAllClose(
        tf_meshes.verts_packed.numpy(),
        torch_meshes.verts_packed().numpy())
    self.assertAllClose(
        tf_meshes.faces_packed.numpy(),
        torch_meshes.faces_packed().numpy())
    self.assertAllClose(
        tf_meshes.edges_packed.numpy(),
        torch_meshes.edges_packed().numpy())

    # Check auxillary variables
    self.assertAllClose(
        tf_meshes.verts_packed_to_mesh_idx.numpy(),
        torch_meshes.verts_packed_to_mesh_idx().numpy())
    self.assertAllClose(
        tf_meshes.mesh_to_verts_packed_first_idx.numpy(),
        torch_meshes.mesh_to_verts_packed_first_idx().numpy())
    self.assertAllClose(
        tf_meshes.faces_packed_to_mesh_idx.numpy(),
        torch_meshes.faces_packed_to_mesh_idx().numpy())
    self.assertAllClose(
        tf_meshes.mesh_to_faces_packed_first_idx.numpy(),
        torch_meshes.mesh_to_faces_packed_first_idx().numpy())
    self.assertAllClose(
        tf_meshes.edges_packed_to_mesh_idx.numpy(),
        torch_meshes.edges_packed_to_mesh_idx().numpy())
    self.assertAllClose(
        tf_meshes.mesh_to_edges_packed_first_idx.numpy(),
        torch_meshes.mesh_to_edges_packed_first_idx().numpy())
    self.assertAllClose(
        tf_meshes.faces_packed_to_edges_packed.numpy(),
        torch_meshes.faces_packed_to_edges_packed().numpy())

    # TODO add tests for packed normals/areas

  def test_meshes_padded_presentation(self, tf_meshes: Meshes,
                                      torch_meshes: PyTorchMeshes):
    """Tests padded representation of meshes class.

    Args:
      tf_meshes: TF implementation of Meshes object.
      torch_meshes: PyTorch implementation of Meshes object.
    """
    # TODO add tests for padded representation

if __name__ == "__main__":
  tf.test.main()
