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

"""Mesh R-CNN Mesh Head"""

import tensorflow as tf

from official.vision.beta.projects.mesh_rcnn.modeling.layers.nn_blocks import \
    MeshRefinementStage
from official.vision.beta.projects.mesh_rcnn.ops.mesh_ops import compute_edges


class MeshHead(tf.keras.Model):
  """Mesh R-CNN Mesh Head."""
  def __init__(self,
               input_specs: dict,
               num_stages: int = 3,
               stage_depth: int = 3,
               output_dim: int = 128,
               graph_conv_init: str = 'normal',
               **kwargs):
    """Initializes Mesh R-CNN Mesh Head.

    Args:
      input_specs: Dictionary with the following key-value pairs:
        'feature_map': `Tensor` of shape [B, H, W, D] for the backbone features.
        'verts': `Tensor` of shape [B, num_verts, 3] for vertex coordinates.
        'verts_mask': `Tensor` of shape [B, num_verts] for vertex mask.
        'faces': `Tensor` of shape [B, num_verts, 3] for faces.
        'faces_mask': `Tensor` of shape [B, num_verts] for faces mask.
      num_stages: `int`, number of mesh refinement stages in the branch.
      stage_depth: `int`, number of graph convolution layers in each stage.
      output_dim:  `int`, number of output features per vertex.
      graph_conv_init: `string` to indicate graph convolution initialization
        method. Can be one of 'zero' or 'normal'.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._input_specs = input_specs
    self._num_stages = num_stages
    self._stage_depth = stage_depth
    self._output_dim = output_dim
    self._graph_conv_init = graph_conv_init

    inputs = {
        key: tf.keras.layers.Input(shape=value[1:])
        for key, value in input_specs.items()
    }

    feature_map = inputs['feature_map']
    verts = inputs['verts']
    verts_mask = tf.cast(inputs['verts_mask'], tf.int32)
    faces = tf.cast(inputs['faces'], tf.int32)
    faces_mask = tf.cast(inputs['faces_mask'], tf.int32)

    edges, edges_mask = compute_edges(faces, faces_mask)

    outputs = {}
    outputs['verts_mask'] = verts_mask
    outputs['faces'] = faces
    outputs['faces_mask'] = faces_mask
    outputs['verts'] = {}

    vert_feats = None
    for i in range(num_stages):
      verts, vert_feats = MeshRefinementStage(
          stage_depth, output_dim, graph_conv_init)(feature_map,
                                                    verts,
                                                    verts_mask,
                                                    vert_feats,
                                                    edges,
                                                    edges_mask)
      outputs['verts']['stage_' + str(i)] = verts

    super().__init__(inputs=inputs, outputs=outputs, name='MeshHead')

  def get_config(self):
    config = dict(
        input_specs=self._input_specs,
        use_fpn=self._num_stages,
        fpn_depth=self._stage_depth,
        output_dim=self._output_dim,
        graph_conv_init=self._graph_conv_init)
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
