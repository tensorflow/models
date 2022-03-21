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

"""Mesh R-CNN Head"""
import tensorflow as tf


class MeshRCNNHead(tf.keras.layers.Layer):
  """Creates a Mesh R-CNN head."""

  def __init__(
      self,
      cubify_threshold: float,
      cubify_alignment: str,
      vert_align_corners: bool,
      vert_align_padding_mode: str,
      num_refinement_stages: int,
      refinement_stage_depth: int,
      refinement_num_output_features: int,
      graph_conv_init: str,
      **kwargs):
    """Initializes a Mesh R-CNN head

    Args:
      cubify_threshold: A `float` specifies threshold for valid occupied voxels.
      cubify_alignment: A `str` one of 'topleft', 'corner', or 'center' that
        defines the alignment of the mesh vertices.
      vert_align_corners: A `bool` that indicates whether the vertex extrema
        coordinates (-1 and 1) will correspond to the corners or centers of the
        pixels. If set to True, the extrema will correspond to the corners.
        Otherwise, they will be set to the centers.
      vert_align_padding_mode: A `str` that defines the sampling behavor
        for vertices not within the range [-1, 1]. Can be one of 'zeros',
        'border', or 'reflection'. Only 'border' mode is currently supported.
      num_refinement_stages: A `int` the number of mesh refinement stages to
        use.
      refinement_stage_depth: A `int` the number of graph convolutions to use
        per mesh refinement stage.
      refinement_num_output_features: A `int` the number of features extracted
        per vertex.
      graph_conv_init: A `str` for the graph convolution initialization method,
        one of 'zero' or 'normal'.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(MeshRCNNHead, self).__init__(**kwargs)
    self._config_dict = {
        'cubify_threshold': cubify_threshold,
        'cubify_alignment': cubify_alignment,
        'vert_align_corners': vert_align_corners,
        'vert_align_padding_mode': vert_align_padding_mode,
        'num_refinement_stages': num_refinement_stages,
        'refinement_stage_depth': refinement_stage_depth,
        'refinement_num_output_features': refinement_num_output_features,
        'graph_conv_init': graph_conv_init
    }

  def build(self, input_shape):
    pass

  def call(self, inputs):
    pass
