# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Test for Mesh R-CNN preprocessing operations."""

import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.ops import meshrcnn_preprocess_ops


class MeshPreprocessOpsTest(parameterized.TestCase, tf.test.TestCase):
  """Mesh R-CNN preprocessing ops test."""
  @parameterized.parameters(
      (2), (50), (100)
  )
  def test_horizontal_flip_coords(self, num_coords):
    "Test for flipping (x,y,z) coordinates."
    coords = tf.random.uniform(
        shape=[num_coords, 3], minval=-1, maxval=1, dtype=tf.float32, seed=1)

    flipped_coords = meshrcnn_preprocess_ops.horizontal_flip_coords(coords)

    self.assertAllEqual(tf.shape(coords), tf.shape(flipped_coords))
    self.assertAllEqual(coords[:, 0], -1 * flipped_coords[:, 0])
    self.assertAllEqual(coords[:, 1], flipped_coords[:, 1])
    self.assertAllEqual(coords[:, 2], flipped_coords[:, 2])

  @parameterized.parameters(
      (2, [0.5, 0.5]),
      (2, [1.5, 1.5]),
      (50, [0.7, 1.4]),
      (50, [2.1, 0.1])
  )
  def test_resize_coords(self, num_coords, scale_factor):
    "Test for resizing (x,y,z) coordinates."
    coords = tf.random.uniform(
        shape=[num_coords, 3], minval=-1, maxval=1, dtype=tf.float32, seed=1)

    resized_coords = meshrcnn_preprocess_ops.resize_coords(coords, scale_factor)

    self.assertAllEqual(tf.shape(coords), tf.shape(resized_coords))
    self.assertAllEqual(scale_factor[1] * coords[:, 0], resized_coords[:, 0])
    self.assertAllEqual(scale_factor[0] * coords[:, 1], resized_coords[:, 1])
    self.assertAllEqual(coords[:, 2], resized_coords[:, 2])

  @parameterized.parameters(
      (1, [[-1.6, -0.0, -0.7], [0.3, 0.8, -0.4], [0.6, -1.5, -0.5]],
       [0.1, 0.7, 1.0]),
      (10, [[-1.6, -0.0, -0.7], [0.3, 0.8, -0.4], [0.6, -1.5, -0.5]],
       [0.1, 0.7, 1.0])
  )
  def test_apply_3d_transforms(self, num_coords, rot_mat, trans_mat):
    "Test for applying transformation matrices to (x,y,z) coordinates."
    coords = tf.random.uniform(
        shape=[num_coords, 3], minval=-1, maxval=1, dtype=tf.float32, seed=1)

    transformed_coords = meshrcnn_preprocess_ops.apply_3d_transformations(
        coords, rot_mat, trans_mat
    )
    self.assertAllEqual(tf.shape(coords), tf.shape(transformed_coords))

if __name__ == '__main__':
  tf.test.main()
