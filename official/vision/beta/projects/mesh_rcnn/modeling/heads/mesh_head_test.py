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

"""Mesh R-CNN Mesh Head Tests"""

import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.modeling.heads.mesh_head import \
    MeshHead
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import create_voxels


class MeshHeadTest(parameterized.TestCase, tf.test.TestCase):
  """Mesh Refinement Branch Test"""

  @parameterized.named_parameters(
      {'testcase_name': 'batched_small_mesh',
       'grid_dims': 2,
       'batch_size': 2,
       'occupancy_locs': [
           [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
           [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
       ],
      }
  )
  def test_pass_through(self, grid_dims, batch_size, occupancy_locs):
    """Test forward-pass of the mesh head."""

    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)

    mesh = cubify(voxels, 0.5)
    verts = mesh['verts']
    faces = mesh['faces']
    verts_mask = mesh['verts_mask']
    faces_mask = mesh['faces_mask']

    feature_map = tf.random.uniform(shape=[batch_size, 12, 12, 256])

    inputs = {
        'feature_map': feature_map,
        'verts': verts,
        'verts_mask': verts_mask,
        'faces': faces,
        'faces_mask': faces_mask
    }

    input_shape = {
        'feature_map': tf.shape(feature_map),
        'verts': tf.shape(verts),
        'verts_mask': tf.shape(verts_mask),
        'faces': tf.shape(faces),
        'faces_mask': tf.shape(faces_mask)
    }

    model = MeshHead(input_shape)
    outputs = model(inputs)

    for v in outputs['verts'].values():
      self.assertAllEqual(tf.shape(v), tf.shape(verts))

    self.assertAllEqual(outputs['verts_mask'], verts_mask)
    self.assertAllEqual(outputs['faces'], faces)
    self.assertAllEqual(outputs['faces_mask'], faces_mask)

if __name__ == '__main__':
  tf.test.main()
