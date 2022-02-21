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

"Unit Tests for NN Blocks."
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.modeling.layers import nn_blocks
from official.vision.beta.projects.mesh_rcnn.ops.cubify import cubify
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import compute_edges
from official.vision.beta.projects.mesh_rcnn.ops.voxel_ops import create_voxels


class GraphConvTest(parameterized.TestCase, tf.test.TestCase):
  """Graph Convolution Layer Tests"""

  @parameterized.named_parameters(
      {'testcase_name': 'batched_small_mesh',
       'grid_dims': 2,
       'batch_size': 2,
       'occupancy_locs': [
           [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
           [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
       ],
       'output_dim': 128
      }
  )
  def test_pass_through(self,
                        grid_dims,
                        batch_size,
                        occupancy_locs,
                        output_dim):
    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)
    edges, edges_mask = compute_edges(faces, faces_mask)
    vert_feats = tf.random.uniform(shape=tf.shape(verts))

    layer = nn_blocks.GraphConv(output_dim)
    out = layer(vert_feats, edges, verts_mask, edges_mask)

    self.assertAllEqual(tf.shape(out)[0:2], tf.shape(vert_feats)[0:2])
    self.assertAllEqual(tf.shape(out)[2], output_dim)

  @parameterized.named_parameters(
      {'testcase_name': 'batched_small_mesh',
       'grid_dims': 2,
       'batch_size': 2,
       'occupancy_locs': [
           [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
           [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
       ],
       'output_dim': 128
      }
  )
  def test_gradient_pass_through(self,
                                 grid_dims,
                                 batch_size,
                                 occupancy_locs,
                                 output_dim):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()

    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)
    edges, edges_mask = compute_edges(faces, faces_mask)
    vert_feats = tf.random.uniform(shape=tf.shape(verts))

    layer = nn_blocks.GraphConv(output_dim)
    out = layer(vert_feats, edges, verts_mask, edges_mask)

    with tf.GradientTape() as tape:
      out = layer(vert_feats, edges, verts_mask, edges_mask)
      grad_loss = loss(out, tf.zeros(tf.shape(out)))

    grad = tape.gradient(grad_loss, layer.trainable_variables)
    self.assertNotIn(None, grad)
    optimizer.apply_gradients(zip(grad, layer.trainable_variables))


class MeshRefinementTest(parameterized.TestCase, tf.test.TestCase):
  """Mesh Refinement Block Tests"""

  @parameterized.named_parameters(
      {'testcase_name': 'batched_small_mesh',
       'grid_dims': 2,
       'batch_size': 2,
       'occupancy_locs': [
           [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
           [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
       ],
       'stage_depth': 3,
       'output_dim': 128
      }
  )
  def test_pass_through(self,
                        grid_dims,
                        batch_size,
                        occupancy_locs,
                        stage_depth,
                        output_dim):
    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)
    edges, edges_mask = compute_edges(faces, faces_mask)
    backbone_output = tf.random.uniform(shape=[batch_size, 12, 12, 256])
    original_verts_shape = tf.shape(verts)

    stage1 = nn_blocks.MeshRefinementStage(stage_depth, output_dim)
    verts, vert_feats = stage1(
        backbone_output, verts, verts_mask, None, edges, edges_mask)

    self.assertAllEqual(tf.shape(verts), original_verts_shape)
    self.assertAllEqual(tf.shape(vert_feats)[:-1], original_verts_shape[:-1])
    self.assertAllEqual(tf.shape(vert_feats)[-1], 128)

    stage2 = nn_blocks.MeshRefinementStage(stage_depth, output_dim)
    verts, vert_feats = stage2(
        backbone_output, verts, verts_mask, vert_feats, edges, edges_mask)

    self.assertAllEqual(tf.shape(verts), original_verts_shape)
    self.assertAllEqual(tf.shape(vert_feats)[:-1], original_verts_shape[:-1])
    self.assertAllEqual(tf.shape(vert_feats)[-1], 128)

  @parameterized.named_parameters(
      {'testcase_name': 'batched_small_mesh',
       'grid_dims': 2,
       'batch_size': 2,
       'occupancy_locs': [
           [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
           [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]
       ],
       'stage_depth': 3,
       'output_dim': 128
      }
  )
  def test_gradient_pass_though(self, grid_dims, batch_size, occupancy_locs,
                                stage_depth, output_dim):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()

    voxels = create_voxels(grid_dims, batch_size, occupancy_locs)
    verts, faces, verts_mask, faces_mask = cubify(voxels, 0.5)
    edges, edges_mask = compute_edges(faces, faces_mask)
    backbone_output = tf.random.uniform(shape=[batch_size, 12, 12, 256])

    stage1 = nn_blocks.MeshRefinementStage(stage_depth, output_dim)
    with tf.GradientTape(persistent=True) as tape:
      verts, vert_feats = stage1(
          backbone_output, verts, verts_mask, None, edges, edges_mask)
      grad_loss = (
          loss(verts, tf.zeros(tf.shape(verts))) +
          loss(vert_feats, tf.zeros(tf.shape(vert_feats)))
      )

    grad = tape.gradient(grad_loss, stage1.trainable_variables)
    self.assertNotIn(None, grad)
    optimizer.apply_gradients(zip(grad, stage1.trainable_variables))

    stage2 = nn_blocks.MeshRefinementStage(stage_depth, output_dim)
    with tf.GradientTape(persistent=True) as tape:
      verts, vert_feats = stage2(
          backbone_output, verts, verts_mask, vert_feats, edges, edges_mask)
      grad_loss = (
          loss(verts, tf.zeros(tf.shape(verts))) +
          loss(vert_feats, tf.zeros(tf.shape(vert_feats)))
      )

    grad = tape.gradient(grad_loss, stage2.trainable_variables)
    self.assertNotIn(None, grad)
    optimizer.apply_gradients(zip(grad, stage2.trainable_variables))

if __name__ == "__main__":
  tf.test.main()
