# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""S3D model configurations."""
import dataclasses
from typing import Text

from official.modeling import hyperparams
from official.vision.configs import backbones_3d
from official.vision.configs import video_classification


@dataclasses.dataclass
class S3D(hyperparams.Config):
  """S3D backbone config.

  Attributes:
    final_endpoint: Specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    first_temporal_kernel_size: Specifies the temporal kernel size for the first
      conv3d filter. A larger value slows down the model but provides little
      accuracy improvement. Must be set to one of 1, 3, 5 or 7.
    temporal_conv_start_at: Specifies the first conv block to use separable 3D
      convs rather than 2D convs (implemented as [1, k, k] 3D conv). This is
      used to construct the inverted pyramid models. 'Conv2d_2c_3x3' is the
      first valid block to use separable 3D convs. If provided block name is
      not present, all valid blocks will use separable 3D convs.
    gating_start_at: Specifies the first conv block to use self gating.
      'Conv2d_2c_3x3' is the first valid block to use self gating.
    swap_pool_and_1x1x1: If True, in Branch_3 1x1x1 convolution is performed
      first, then followed by max pooling. 1x1x1 convolution is used to reduce
      the number of filters. Thus, max pooling is performed on less filters.
    gating_style: Self gating can be applied after each branch and/or after each
      inception cell. It can be one of ['BRANCH', 'CELL', 'BRANCH_AND_CELL'].
    use_sync_bn: If True, use synchronized batch normalization.
    norm_momentum: A `float` of normalization momentum for the moving average.
    norm_epsilon: A `float` added to variance to avoid dividing by zero.
    temporal_conv_type: It can be one of ['3d', '2+1d', '1+2d', '1+1+1d'] where
      '3d' is SPATIOTEMPORAL 3d convolution, '2+1d' is SPATIAL_TEMPORAL_SEPARATE
      with 2D convolution on the spatial dimensions followed by 1D convolution
      on the temporal dimension, '1+2d' is TEMPORAL_SPATIAL_SEPARATE with 1D
      convolution on the temporal dimension followed by 2D convolution on the
      spatial dimensions, and '1+1+1d' is FULLY_SEPARATE with 1D convolutions on
      the horizontal, vertical, and temporal dimensions, respectively.
    depth_multiplier: Float multiplier for the depth (number of channels) for
      all convolution ops. The value must be greater than zero. Typical usage
      will be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
  """
  final_endpoint: Text = 'Mixed_5c'
  first_temporal_kernel_size: int = 3
  temporal_conv_start_at: Text = 'Conv2d_2c_3x3'
  gating_start_at: Text = 'Conv2d_2c_3x3'
  swap_pool_and_1x1x1: bool = True
  gating_style: Text = 'CELL'
  use_sync_bn: bool = False
  norm_momentum: float = 0.999
  norm_epsilon: float = 0.001
  temporal_conv_type: Text = '2+1d'
  depth_multiplier: float = 1.0


@dataclasses.dataclass
class Backbone3D(backbones_3d.Backbone3D):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, on the of fields below.
    s3d: s3d backbone config.
  """
  type: str = 's3d'
  s3d: S3D = dataclasses.field(default_factory=S3D)


@dataclasses.dataclass
class S3DModel(video_classification.VideoClassificationModel):
  """The S3D model config.

  Attributes:
    type: 'str', type of backbone be used, on the of fields below.
    backbone: backbone config.
  """
  model_type: str = 's3d'
  backbone: Backbone3D = dataclasses.field(default_factory=Backbone3D)
