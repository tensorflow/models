# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Configuration definitions for MobileNet."""

from typing import Text, Tuple, Union

from dataclasses import dataclass
from official.modeling.hyperparams import base_config


@dataclass
class Conv2DBlockConfig(base_config.Config):
  """Configuration for a block of MobileNetV1 model."""
  kernel: Tuple[int, int] = (3, 3)
  stride: int = 1
  filters: int = 32


@dataclass
class DepthwiseConv2DBlockConfig(base_config.Config):
  """Configuration for a block of MobileNetV1 model."""
  kernel: Tuple[int, int] = (3, 3)
  stride: int = 1
  filters: int = 32


@dataclass
class InvertedResConv2DBlockConfig(base_config.Config):
  """Configuration for a block of MobileNetV1 model."""
  kernel: Tuple[int, int] = (3, 3)
  stride: int = 1
  filters: int = 32
  expansion_size: int = 6


BlockConfig = Union[Conv2DBlockConfig,
                    DepthwiseConv2DBlockConfig,
                    InvertedResConv2DBlockConfig]


@dataclass
class MobileNetV1Config(base_config.Config):
  """Configuration for the MobileNetV1 model.

    Attributes:
      name: name of the target model.
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      dropout_keep_prob: the percentage of activation values that are retained.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when width_multiplier < 1, and not an active constraint when
        width_multiplier >= 1.
      width_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      output_stride: An integer that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of the activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.
      spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.

  """
  name: Text = 'MobileNetV1'
  num_classes: int = 1000
  dropout_keep_prob: float = 0.999
  min_depth: int = 8
  width_multiplier = 1.0
  weight_decay: float = 0.00004
  stddev: float = 0.09
  regularize_depthwise: bool = False
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  output_stride: int = None
  use_explicit_padding: bool = False
  global_pool: bool = True
  spatial_squeeze: bool = True

  blocks: Tuple[BlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    Conv2DBlockConfig.from_args((3, 3), 2, 32),
    # depthsep conv
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 64),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 2, 128),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 128),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 2, 256),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 256),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 2, 512),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 512),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 512),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 512),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 512),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 512),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 2, 1024),
    DepthwiseConv2DBlockConfig.from_args((3, 3), 1, 1024),
    # pylint: enable=bad-whitespace
  )


@dataclass
class MobileNetV2Config(base_config.Config):
  """Configuration for the MobileNetV2 model.

    Attributes:
      name: name of the target model.
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      dropout_keep_prob: the percentage of activation values that are retained.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when width_multiplier < 1, and not an active constraint when
        width_multiplier >= 1.
      width_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      output_stride: An integer that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of the activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.
      spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.

  """
  name: Text = 'MobileNetV2'
  num_classes: int = 1000
  dropout_keep_prob: float = 0.999
  min_depth: int = 8
  width_multiplier = 1.0
  weight_decay: float = 0.00004
  stddev: float = 0.09
  regularize_depthwise: bool = False
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  output_stride: int = None
  use_explicit_padding: bool = False
  global_pool: bool = True
  spatial_squeeze: bool = True

  blocks: Tuple[BlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    Conv2DBlockConfig.from_args((3, 3), 2, 32),
    # inverted res conv
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 16, 1),

    InvertedResConv2DBlockConfig.from_args((3, 3), 2, 24, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 24, 6),

    InvertedResConv2DBlockConfig.from_args((3, 3), 2, 32, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 32, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 32, 6),

    InvertedResConv2DBlockConfig.from_args((3, 3), 2, 64, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 64, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 64, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 64, 6),

    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 96, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 96, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 96, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 2, 24, 6),

    InvertedResConv2DBlockConfig.from_args((3, 3), 2, 160, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 160, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 160, 6),
    InvertedResConv2DBlockConfig.from_args((3, 3), 1, 320, 6),

    Conv2DBlockConfig.from_args((1, 1), 1, 1280),
    # pylint: enable=bad-whitespace
  )


@dataclass
class MobileNetV3Config(base_config.Config):
  """Configuration for the MobileNetV3 model."""
  pass
