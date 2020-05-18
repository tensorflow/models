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

from typing import Text, Tuple, Callable, Mapping
from dataclasses import dataclass

import tensorflow as tf

from official.modeling.hyperparams import base_config

Conv = 'Conv'
DepSepConv = 'DepSepConv'
InvertedResConv = 'InvertedResConv'


def get_activation_function() -> Mapping[Text, Callable]:
  return {
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'swish': tf.nn.swish,
    'elu': tf.nn.elu,
    'sigmoid': tf.nn.sigmoid,
    'softmax': tf.nn.softmax
  }


def get_normalization_layer() -> Mapping[Text, tf.keras.layers.Layer]:
  return {
    'batch_norm': tf.keras.layers.BatchNormalization,
    'layer_norm': tf.keras.layers.LayerNormalization
  }


@dataclass
class MobileNetBlockConfig(base_config.Config):
  """Configuration for a block of MobileNetV1 model."""
  kernel: Tuple[int, int] = (3, 3)
  stride: int = 1
  filters: int = 32
  expansion_size: int = 6  # used for block type InvertedResConv
  block_type: Text = Conv


@dataclass
class MobileNetV1Config(base_config.Config):
  """Configuration for the MobileNetV1 model.

    Attributes:
      name: name of the target model.
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when width_multiplier < 1, and not an active constraint when
        width_multiplier >= 1.
      width_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
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
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      activation_fn: Name of the activation function
      normalization_layer: Name of the normalization layer
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      dropout_keep_prob: the percentage of activation values that are retained.

  """
  name: Text = 'MobileNetV1'
  num_classes: int = 1000
  # model specific
  min_depth: int = 8
  width_multiplier = 1.0
  output_stride: int = None
  use_explicit_padding: bool = False
  global_pool: bool = True
  spatial_squeeze: bool = True
  # regularization
  weight_decay: float = 0.00004
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.999
  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32, block_type=Conv),
    # depthsep conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=12, filters=128, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=128, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=12, filters=256, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=256, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=12, filters=512, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=512, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=512, block_type=DepSepConv),
    MobileNetBlockConfig.from_args
    (kernel=(3, 3), stride=11, filters=512, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=512, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=512, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=12, filters=1024, block_type=DepSepConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=11, filters=1024, block_type=DepSepConv),
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
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when width_multiplier < 1, and not an active constraint when
        width_multiplier >= 1.
      width_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      output_stride: An integer that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of the activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      activation_fn: Name of the activation function
      normalization_layer: Name of the normalization layer
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      dropout_keep_prob: the percentage of activation values that are retained.

  """
  name: Text = 'MobileNetV2'
  num_classes: int = 1000
  # model specific
  min_depth: int = 8
  width_multiplier = 1.0
  output_stride: int = None
  use_explicit_padding: bool = False
  spatial_squeeze: bool = True
  # regularization
  weight_decay: float = 0.00004
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.999
  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32, block_type=Conv),
    # inverted res conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=16,
      expansion_size=1, block_type=InvertedResConv),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=24,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=24,
      expansion_size=6, block_type=InvertedResConv),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      expansion_size=6, block_type=InvertedResConv),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=64,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      expansion_size=6, block_type=InvertedResConv),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=24,
      expansion_size=6, block_type=InvertedResConv),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=160,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=160,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=160,
      expansion_size=6, block_type=InvertedResConv),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=320,
      expansion_size=6, block_type=InvertedResConv),

    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=1280, block_type=Conv),
    # pylint: enable=bad-whitespace
  )


@dataclass
class MobileNetV3Config(base_config.Config):
  """Configuration for the MobileNetV3 model."""
  pass
