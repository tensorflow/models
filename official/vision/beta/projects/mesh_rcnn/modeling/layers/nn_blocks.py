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

"""Contains common building blocks for Mesh R-CNN."""
from typing import Union

import tensorflow as tf
import tensorflow.keras as ks
from torch import NoneType

from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import vert_align


# @tf.keras.utils.register_keras_serializable(package='mesh_rcnn')
class GraphConv(tf.keras.layers.Layer):
  """A single graph convolution layer."""
  def __init__(self,
               output_dim: int,
               init: str = 'normal',
               directed: bool = False,
               **kwargs) -> None:
    """
    Args:
      output_dim: `int`, number of output features per vertex.
      init: `string` to indicate initialization method. Can be one of 'zero' or
        'normal'.
      directed: `bool` indicating if edges in the graph are directed.
      **kwargs: Additional keyword arguments.
    Raises:
      ValueError: If an invalid initialization method is used.
    """

    self._output_dim = output_dim
    self._directed = directed
    self._initialization = init

    # Set initialization parameters
    if init == 'normal':
      self._w0_kernel_initializer = tf.keras.initializers.RandomNormal(
          mean=0, stddev=0.01)
      self._w0_bias_initializer = tf.keras.initializers.Zeros()
      self._w1_kernel_initializer = tf.keras.initializers.RandomNormal(
          mean=0, stddev=0.01)
      self._w1_bias_initializer = tf.keras.initializers.Zeros()
    elif init == 'zeros':
      self._w0_kernel_initializer = tf.keras.initializers.Zeros()
      self._w0_bias_initializer = tf.keras.initializers.GlorotUniform()
      self._w1_kernel_initializer = tf.keras.initializers.Zeros()
      self._w1_bias_initializer = tf.keras.initializers.GlorotUniform()
    else:
      raise ValueError(f'invalid GraphConv initialization "{init}"')

    super().__init__(**kwargs)

  def build(self, input_shape: list):
    """GraphConv build function.

    Args:
      input_shape: A `list` of `TensorShape`s that specifies the shapes of the
        inputs.
    """
    self._w0 = tf.keras.layers.Dense(
        units=self._output_dim,
        kernel_initializer=self._w0_kernel_initializer,
        bias_initializer=self._w0_bias_initializer
    )
    self._w1 = tf.keras.layers.Dense(
        units=self._output_dim,
        kernel_initializer=self._w1_kernel_initializer,
        bias_initializer=self._w1_bias_initializer
    )

    super().build(input_shape)

  def call(self,
           vert_feats: tf.Tensor,
           edges: tf.Tensor,
           verts_mask: tf.Tensor,
           edges_mask: tf.Tensor):
    """
    Args:
      vert_feats: A `Tensor` of shape [B, num_verts, channels] that gives the
        features for each vertex in a mesh.
      edges: A `Tensor` of shape [B, num_edges, 2], where the last dimension
        contain the vertex indices that make up the edge. This may include
        duplicate edges.
      verts_mask: A `Tensor` of shape [B, num_verts], a mask for valid vertices
        in the watertight mesh.
      edges_mask: A `Tensor` of shape [B, num_edges], a mask for valid edges
        in the watertight mesh.

    Returns:
      out: A `Tensor` with shape [B, num_verts, self._output_dims] that are the
        refined features for the mesh.
    """
    shape = tf.shape(vert_feats)
    batch_size, num_verts, _ = shape[0], shape[1], shape[2]
  
    # For empty meshes, return 0 for the features
    if tf.reduce_sum(verts_mask) == 0:
      return tf.zeros(shape=[batch_size, num_verts, self._output_dim])
    
    # Apply dense layers
    verts_w0 = self._w0(vert_feats)
    verts_w1 = self._w1(vert_feats)
    
    neighbor_sums = self._gather_scatter(
        verts_w1, edges, edges_mask, self._directed)
    
    out = verts_w0 + neighbor_sums

    return out

  def _gather_scatter(self,
                      vert_feats: tf.Tensor,
                      edges: tf.Tensor,
                      edges_mask: tf.Tensor,
                      directed: bool = False):
    """Sums features over neighboring verts using gather and scatter.

    Given the edges of a mesh and the features associated to each vertex, this
    function computes the sum of the features of each vertex's neighbors.

    Args:
      vert_feats: A `Tensor` of shape [B, num_verts, channels] that gives the
          features for each vertex in a mesh.
      edges: A `Tensor` of shape [B, num_edges, 2], where the last dimension
        contain the vertex indices that make up the edge. This may include
        duplicate edges.
      edges_mask: A `Tensor` of shape [B, num_edges], a mask for valid edges
        in the watertight mesh.
      directed: bool, whether to compute the sum in both edge directions

    Returns:
      output: A `Tensor` of shape [B, num_verts, channels] that gives the
        summed features for each vertex's neighbors in a mesh.
    """
    shape = tf.shape(vert_feats)
    batch_size = shape[0]

    # Initialize output
    output = tf.zeros_like(vert_feats)

    # Grab the features of adjacent verts
    edges_mask = tf.expand_dims(edges_mask, axis=-1)
    gather_0 = tf.gather(
        vert_feats, edges[..., 0], axis=1, batch_dims=1)
    gather_1 = tf.gather(
        vert_feats, edges[..., 1], axis=1, batch_dims=1)

    # Mask out unused features
    gather_0 = gather_0 * tf.cast(edges_mask, gather_0.dtype)
    gather_1 = gather_1 * tf.cast(edges_mask, gather_0.dtype)

    idx0 = tf.expand_dims(edges[..., 0], axis=-1)

    # Generate the update indices for scatter
    num_edges = tf.shape(edges)[1]
    batch_repeats = tf.repeat(num_edges, repeats=batch_size)
    batch_indices = tf.repeat(tf.range(batch_size), repeats=batch_repeats)
    batch_indices = tf.reshape(batch_indices, tf.shape(idx0))
    idx0 = tf.concat([tf.cast(batch_indices, idx0.dtype), idx0], axis=2)

    # Compute sum of adjacent features for each vertex
    output = tf.tensor_scatter_nd_add(output, idx0, gather_1)

    if not directed:
      idx1 = tf.expand_dims(edges[..., 1], axis=-1)
      idx1 = tf.concat([batch_indices, idx1], axis=2)
      output = tf.tensor_scatter_nd_add(output, idx1, gather_0)

    return output

  def get_config(self):
    """Get config"""
    layer_config = {
        'output_dim': self._output_dim,
        'init': self._initialization,
        'directed': self._directed
    }
    layer_config.update(super().get_config())
    return layer_config

class MeshRefinementStage(tf.keras.layers.Layer):
  """A single Mesh Refinment Stage."""
  def __init__(self,
               stage_depth: int,
               output_dim: int,
               graph_conv_init: int = 'normal',
               **kwargs):
    """
    Args:
      stage_depth: `int`, number of GraphConv layers to use in the stage.
      output_dim: `int` Number of output features to extract for each vertex.
      graph_conv_init: `string` to indicate initialization method. Can be one of
        'zero' or 'normal'.
      **kwargs: Additional keyword arguments.
    """
    self._stage_depth = stage_depth
    self._output_dim = output_dim
    self._graph_conv_init = graph_conv_init

    super().__init__(**kwargs)

  def build(self, input_shape: list):
    """Mesh Refinment Stage build function.

    Args:
      input_shape: A `list` of `TensorShape`s that specifies the shapes of the
        inputs.
    """
    self.bottleneck = ks.layers.Dense(
        self._output_dim,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0,
            stddev=0.01),
        bias_initializer='zeros')

    self.verts_offset = ks.layers.Dense(
        3,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0,
            stddev=0.01),
        bias_initializer='zeros')

    self.gconvs = []
    for _ in range(self._stage_depth):
      self.gconvs.append(GraphConv(self._output_dim,
                                   self._graph_conv_init))
    super().build(input_shape)

  def call(self,
           feature_map: tf.Tensor,
           verts: tf.Tensor,
           verts_mask: tf.Tensor,
           vert_feats: Union[tf.Tensor, NoneType],
           edges: tf.Tensor,
           edges_mask: tf.Tensor):
    """
    Args:
      feature_map: A `Tensor` of shape [B, H, W, C], from which to extract
        features for each vertex in the mesh.
      verts: A `Tensor` of shape [B, num_verts, 3], where the last dimension
        contains the (x,y,z) vertex coordinates.
      verts_mask: A `Tensor` of shape [B, num_verts], a mask for valid vertices
        in the watertight mesh.
      vert_feats: Either a `Tensor` of shape [B, num_verts, C] that
        gives the features for each vertex in a mesh, or 'None'.
      edges: A `Tensor` of shape [B, num_edges, 2], where the last dimension
        contain the vertex indices that make up the edge. This may include
        duplicate edges.
      edges_mask: A `Tensor` of shape [B, num_edges], a mask for valid edges
        in the watertight mesh.

    Returns:
      new_verts: A `Tensor` of shape [B, num_verts, 3], the updated vertex
        coordinates of the mesh.
      vert_feats_nopos: A `Tensor` of shape [B, num_verts, C], that contains
        the new vertex coordinates.
    """
    verts_mask = tf.expand_dims(verts_mask, axis=-1)
    img_feats = vert_align(feature_map, verts)

    img_feats = self.bottleneck(img_feats)
    img_feats = tf.nn.relu(img_feats)

    if vert_feats is None:
      vert_feats = tf.concat([img_feats, verts], axis=2)
    else:
      vert_feats = tf.concat([vert_feats, img_feats, verts], axis=2)

    # Mask features before applying graph convolution
    vert_feats *= tf.cast(verts_mask, vert_feats.dtype)
  
    for graph_conv in self.gconvs:
      vert_feats_nopos = tf.nn.relu(
          graph_conv(vert_feats, edges, verts_mask, edges_mask))
 
      vert_feats_nopos = vert_feats_nopos * tf.cast(verts_mask, vert_feats_nopos.dtype)
      vert_feats = tf.concat([vert_feats_nopos, verts], axis=2)

      vert_feats *= tf.cast(verts_mask, vert_feats.dtype)
    
    vert_feats = self.verts_offset(vert_feats)
    vert_feats *= tf.cast(verts_mask, vert_feats.dtype) 
    deform = tf.nn.tanh(vert_feats)

    new_verts = verts + deform

    return new_verts, vert_feats_nopos
