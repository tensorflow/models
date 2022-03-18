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

"Differential Tests for NN Blocks."

from unittest import SkipTest, skip
import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized
from pytorch3d.ops import GraphConv as torch_graph_conv

from official.vision.beta.projects.mesh_rcnn.modeling.layers.nn_blocks import \
    GraphConv as tf_graph_conv


class NNBLocksDifferentialTest(parameterized.TestCase, tf.test.TestCase):
  """Differential Tests for GraphConv."""
  def test_graph_conv(self):
    num_verts = 5
    num_edges = 10
    input_features = 20
    output_features = 30

    verts = np.random.uniform(
        low=-5.0, high=5.0, size=(num_verts, input_features))
    edges = np.random.randint(
        low=0, high=num_verts-1, size=(num_edges, 2), dtype=np.int64)

    w0_weights = np.random.uniform(
        low=-1.0, high=1.0, size=(output_features, input_features))
    w0_bias = np.random.uniform(low=-1.0, high=1.0, size=(output_features))
    w1_weights = np.random.uniform(
        low=-1.0, high=1.0, size=(output_features, input_features))
    w1_bias = np.random.uniform(low=-1.0, high=1.0, size=(output_features))

    tf_verts = tf.convert_to_tensor(verts, tf.float32)
    tf_verts = tf.expand_dims(tf_verts, axis=0)
    tf_verts_mask = tf.ones(shape=[1, num_verts], dtype=tf.int32)
    torch_verts = torch.as_tensor(verts, dtype=torch.float32)

    tf_edges = tf.convert_to_tensor(edges, tf.int64)
    tf_edges = tf.expand_dims(tf_edges, axis=0)
    torch_edges = torch.as_tensor(edges, dtype=torch.int64)
    tf_edges_mask = tf.ones(shape=[1, num_edges], dtype=tf.int32)

    torch_layer = torch_graph_conv(
        input_dim=input_features, output_dim=output_features)
    tf_layer = tf_graph_conv(output_dim=output_features)

    # Instantiate weights
    tf_layer(tf_verts, tf_edges, tf_verts_mask, tf_edges_mask)
    torch_layer(torch_verts, torch_edges)

    with torch.no_grad():
      torch_layer.w0.weight = torch.nn.Parameter(
          torch.as_tensor(w0_weights, dtype=torch.float32))
      torch_layer.w0.bias = torch.nn.Parameter(
          torch.as_tensor(w0_bias, dtype=torch.float32))
      torch_layer.w1.weight = torch.nn.Parameter(
          torch.as_tensor(w1_weights, dtype=torch.float32))
      torch_layer.w1.bias = torch.nn.Parameter(
          torch.as_tensor(w1_bias, dtype=torch.float32))

    tf_layer.set_weights(
        [np.transpose(w0_weights), w0_bias, np.transpose(w1_weights), w1_bias]
    )

    tf_outputs = tf_layer(tf_verts, tf_edges, tf_verts_mask, tf_edges_mask)
    tf_outputs = tf.squeeze(tf_outputs)
    torch_outputs = torch_layer(torch_verts, torch_edges)

    torch_outputs = torch_outputs.cpu().detach().numpy()
    tf_outputs = tf_outputs.numpy()

    self.assertAllClose(torch_outputs, tf_outputs, atol=1e-2, rtol=1e-2)

  def test_graph_conv_with_mask(self):
    num_verts = 5
    num_edges = 10
    input_features = 15
    output_features = 20
    num_extra_edges = 20

    verts = np.random.uniform(
        low=-5.0, high=5.0, size=(num_verts, input_features))
    edges = np.random.randint(
        low=0, high=num_verts-1, size=(num_edges, 2), dtype=np.int64)
    extra_edges = np.random.randint(
        low=0, high=num_verts-1, size=(num_extra_edges, 2), dtype=np.int64)

    w0_weights = np.random.uniform(
        low=-1.0, high=1.0, size=(output_features, input_features))
    w0_bias = np.random.uniform(low=-1.0, high=1.0, size=(output_features))
    w1_weights = np.random.uniform(
        low=-1.0, high=1.0, size=(output_features, input_features))
    w1_bias = np.random.uniform(
        low=-1.0, high=1.0, size=(output_features))

    tf_verts = tf.convert_to_tensor(verts, tf.float32)
    tf_verts = tf.expand_dims(tf_verts, axis=0)
    tf_verts_mask = tf.ones(shape=[1, num_verts], dtype=tf.int32)
    torch_verts = torch.as_tensor(verts, dtype=torch.float32)

    tf_edges = tf.convert_to_tensor(edges, tf.int64)
    tf_extra_edges = tf.convert_to_tensor(extra_edges, tf.int64)
    tf_edges = tf.concat([tf_edges, tf_extra_edges], axis=0)
    tf_edges = tf.expand_dims(tf_edges, axis=0)
    torch_edges = torch.as_tensor(edges, dtype=torch.int64)
    tf_edges_mask = tf.ones(shape=[1, num_edges], dtype=tf.int32)
    tf_edges_mask = tf.concat(
        [tf_edges_mask, tf.zeros(shape=[1, num_extra_edges], dtype=tf.int32)],
        axis=1)

    torch_layer = torch_graph_conv(
        input_dim=input_features, output_dim=output_features)
    tf_layer = tf_graph_conv(output_dim=output_features)

    # Instantiate weights
    tf_layer(tf_verts, tf_edges, tf_verts_mask, tf_edges_mask)
    torch_layer(torch_verts, torch_edges)

    with torch.no_grad():
      torch_layer.w0.weight = torch.nn.Parameter(
          torch.as_tensor(w0_weights, dtype=torch.float32))
      torch_layer.w0.bias = torch.nn.Parameter(
          torch.as_tensor(w0_bias, dtype=torch.float32))
      torch_layer.w1.weight = torch.nn.Parameter(
          torch.as_tensor(w1_weights, dtype=torch.float32))
      torch_layer.w1.bias = torch.nn.Parameter(
          torch.as_tensor(w1_bias, dtype=torch.float32))

    tf_layer.set_weights([
        np.transpose(w0_weights), w0_bias, np.transpose(w1_weights), w1_bias
    ])

    tf_outputs = tf_layer(tf_verts, tf_edges, tf_verts_mask, tf_edges_mask)
    tf_outputs = tf.squeeze(tf_outputs)
    torch_outputs = torch_layer(torch_verts, torch_edges)

    torch_outputs = torch_outputs.cpu().detach().numpy()
    tf_outputs = tf_outputs.numpy()

    self.assertAllClose(torch_outputs, tf_outputs, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
  tf.test.main()
