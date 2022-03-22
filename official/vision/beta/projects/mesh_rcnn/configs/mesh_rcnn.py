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
"""Mesh R-CNN configuration definition."""

import dataclasses

from official.modeling import hyperparams  # type: ignore


@dataclasses.dataclass
class VoxelHead(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Voxel Branch Prediction Head."""
  voxel_depth: int = 28
  conv_dim: int = 256
  num_conv: int = 0
  use_group_norm: bool = False
  predict_classes: bool = False
  bilinearly_upscale_input: bool = True
  class_based_voxel: bool = False
  num_classes: int = 0

@dataclasses.dataclass
class MeshHead(hyperparams.Config):
  """Parameterization for the Mesh R-CNN Mesh Head."""
  num_stages: int = 3
  stage_depth: int = 3
  output_dim: int = 128
  graph_conv_init: str = 'normal'
