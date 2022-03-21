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

"""Test for mesh ops."""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import (
    compute_edges, vert_align)


class MeshOpsTest(parameterized.TestCase, tf.test.TestCase):
  """Test for mesh ops."""
  def test_compute_edges(self):
    faces = tf.random.uniform(shape=[2, 1000, 3], maxval=1000, dtype=tf.int32)
    faces_mask = tf.random.uniform(shape=[2, 1000], maxval=1, dtype=tf.int32)

    edges, edges_mask = compute_edges(faces, faces_mask)

    self.assertAllEqual(tf.shape(edges)[1], tf.shape(faces)[1] * 3)
    self.assertAllEqual(tf.shape(edges)[1], tf.shape(edges_mask)[1])

    edges = edges.numpy()
    edges_mask = edges_mask.numpy()

    for edge, mask in zip(edges, edges_mask):
      valid_edges = edge[np.array(mask, dtype=bool), :]
      unique_edges = np.unique(valid_edges, axis=1)
      self.assertEqual(valid_edges.shape[0], unique_edges.shape[0])

  def test_vert_align(self):
    feature_map = tf.random.uniform(shape=[2, 4, 4, 3], minval=-1, maxval=1)
    verts = tf.convert_to_tensor([
        [
            [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]
        ],
        [
            [1.1, 1.2, 1.3], [-1.1, -1.2, -1.3]
        ]
    ])

    features = vert_align(feature_map, verts, True, 'border')
    self.assertAllEqual(tf.shape(features)[:-1], tf.shape(verts)[:-1])
    self.assertAllEqual(tf.shape(features)[-1], tf.shape(feature_map)[-1])

if __name__ == "__main__":
  tf.test.main()
