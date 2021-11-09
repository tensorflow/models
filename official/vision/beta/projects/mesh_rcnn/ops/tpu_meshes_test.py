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

from official.vision.beta.projects.mesh_rcnn.ops.tpu_meshes import Meshes

VERTS_MAX_INSTANCES=884736
FACES_MAX_INSTANCES=1327104
EDGES_MAX_INSTANCES=3981312


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
  def test_verts_packed(self, num_verts_per_mesh: List[int],
                                 faces: List[List[List[int]]]):
    """Tests packed verts representation.

    Args:
      num_verts_per_mesh: `List` of integers for the number of vertices in each
          mesh to be created
      faces: `List` of faces for each mesh to be created.
    """
    self.skipTest(reason="skipping")
    # Shape tests
    tf_meshes, torch_meshes = self._create_meshes(num_verts_per_mesh, faces)

    tf_verts_packed = tf_meshes._verts_packed.numpy()
    tf_num_verts_per_mesh = tf_meshes._num_verts_per_mesh.numpy()
    tf_mesh_to_verts_packed_first_idx = tf_meshes._mesh_to_verts_packed_first_idx.numpy()
    tf_verts_packed_to_mesh_idx = tf_meshes._verts_packed_to_mesh_idx.numpy()

    torch_verts_packed = torch_meshes.verts_packed().numpy()
    torch_num_verts_per_mesh = torch_meshes.num_verts_per_mesh().numpy()
    torch_mesh_to_verts_packed_first_idx = torch_meshes.mesh_to_verts_packed_first_idx().numpy()
    torch_verts_packed_to_mesh_idx = torch_meshes.verts_packed_to_mesh_idx().numpy()

    expected_verts_packed_shape = [VERTS_MAX_INSTANCES, 3]
    self.assertAllEqual(tf_verts_packed.shape, expected_verts_packed_shape)

    expected_num_verts_per_mesh_shape = [len(num_verts_per_mesh)]
    self.assertAllEqual(tf_num_verts_per_mesh.shape,
        expected_num_verts_per_mesh_shape)

    expected_mesh_to_verts_packed_first_idx_shape = [len(num_verts_per_mesh)]
    self.assertAllEqual(tf_mesh_to_verts_packed_first_idx.shape, 
        expected_mesh_to_verts_packed_first_idx_shape)

    expected_verts_packed_to_mesh_idx_shape = [VERTS_MAX_INSTANCES]
    self.assertAllEqual(tf_verts_packed_to_mesh_idx.shape, 
        expected_verts_packed_to_mesh_idx_shape)

    # Value tests
    self.assertAllClose(np.sum(tf_verts_packed), np.sum(torch_verts_packed))
    self.assertAllEqual(tf_num_verts_per_mesh, torch_num_verts_per_mesh)
    self.assertAllEqual(tf_mesh_to_verts_packed_first_idx, torch_mesh_to_verts_packed_first_idx)

    # Offsetting the sum to account for the fact that
    # tf_verts_packed_to_mesh_idx is padded with -1 up to VERTS_MAX_INSTANCES
    tf_verts_packed_to_mesh_idx_sum = np.sum(tf_verts_packed_to_mesh_idx)
    torch_verts_packed_to_mesh_idx_sum = np.sum(torch_verts_packed_to_mesh_idx)
    torch_verts_packed_to_mesh_idx_sum += -1 * (VERTS_MAX_INSTANCES - sum(num_verts_per_mesh))

    self.assertAllClose(tf_verts_packed_to_mesh_idx_sum, torch_verts_packed_to_mesh_idx_sum)

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
  def test_faces_packed(self, num_verts_per_mesh: List[int],
                                 faces: List[List[List[int]]]):
    """Tests packed faces representation.

    Args:
      num_verts_per_mesh: `List` of integers for the number of vertices in each
          mesh to be created
      faces: `List` of faces for each mesh to be created.
    """
    self.skipTest(reason="skipping")
    tf_meshes, torch_meshes = self._create_meshes(num_verts_per_mesh, faces)

    # Shape tests
    tf_faces_packed = tf_meshes._faces_packed.numpy()
    tf_num_faces_per_mesh = tf_meshes._num_faces_per_mesh.numpy()
    tf_mesh_to_faces_packed_first_idx = tf_meshes._mesh_to_faces_packed_first_idx.numpy()
    tf_faces_packed_to_mesh_idx = tf_meshes._faces_packed_to_mesh_idx.numpy()

    torch_faces_packed = torch_meshes.faces_packed().numpy()
    torch_num_faces_per_mesh = torch_meshes.num_faces_per_mesh().numpy()
    torch_mesh_to_faces_packed_first_idx = torch_meshes.mesh_to_faces_packed_first_idx().numpy()
    torch_faces_packed_to_mesh_idx = torch_meshes.faces_packed_to_mesh_idx().numpy()

    expected_faces_packed_shape = [FACES_MAX_INSTANCES, 3]
    self.assertAllEqual(tf_faces_packed.shape, expected_faces_packed_shape)

    expected_num_faces_per_mesh_shape = [len(faces)]
    self.assertAllEqual(tf_num_faces_per_mesh.shape,
        expected_num_faces_per_mesh_shape)

    expected_mesh_to_faces_packed_first_idx_shape = [len(faces)]
    self.assertAllEqual(tf_mesh_to_faces_packed_first_idx.shape, 
        expected_mesh_to_faces_packed_first_idx_shape)

    expected_faces_packed_to_mesh_idx_shape = [FACES_MAX_INSTANCES]
    self.assertAllEqual(tf_faces_packed_to_mesh_idx.shape, 
        expected_faces_packed_to_mesh_idx_shape)

    # Value tests
    num_faces = sum([len(x) for x in faces])

    tf_faces_packed_sum = np.sum(tf_faces_packed)
    torch_faces_packed_sum = np.sum(torch_faces_packed)
    torch_faces_packed_sum += -3 * (FACES_MAX_INSTANCES - num_faces)
    self.assertAllClose(tf_faces_packed_sum, torch_faces_packed_sum)

    self.assertAllEqual(tf_num_faces_per_mesh, torch_num_faces_per_mesh)
    self.assertAllEqual(tf_mesh_to_faces_packed_first_idx, 
        torch_mesh_to_faces_packed_first_idx)

    tf_verts_packed_to_mesh_idx_sum = np.sum(tf_faces_packed_to_mesh_idx)
    torch_verts_packed_to_mesh_idx_sum = np.sum(torch_faces_packed_to_mesh_idx)
    torch_verts_packed_to_mesh_idx_sum += -1 * (FACES_MAX_INSTANCES - num_faces)
    self.assertAllEqual(tf_verts_packed_to_mesh_idx_sum, 
        torch_verts_packed_to_mesh_idx_sum)

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
  def test_edges_packed(self, num_verts_per_mesh: List[int],
                                 faces: List[List[List[int]]]):
    """Tests packed faces representation.

    Args:
      num_verts_per_mesh: `List` of integers for the number of vertices in each
          mesh to be created
      faces: `List` of faces for each mesh to be created.
    """
    tf_meshes, torch_meshes = self._create_meshes(num_verts_per_mesh, faces)

    
if __name__ == "__main__":
  tf.test.main()
