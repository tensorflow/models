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

from typing import Text, Tuple, Callable, Mapping, Type
from dataclasses import dataclass
import enum

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from official.modeling.hyperparams import base_config

layers = tf.keras.layers


def hard_sigmoid(x):
  return tf.nn.relu6(x + 3) * 0.16667


def hard_swish(x):
  return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def get_activation_function() -> Mapping[Text, Callable]:
  return {
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'elu': tf.nn.elu,
    'swish': tf.nn.swish,
    'hard_swish': hard_swish,
    'sigmoid': tf.nn.sigmoid,
    'hard_sigmoid': hard_sigmoid,
    'softmax': tf.nn.softmax,
  }


def get_normalization_layer() -> Mapping[Text, Type[tf.keras.layers.Layer]]:
  return {
    'batch_norm': tf.keras.layers.BatchNormalization,
    'layer_norm': tf.keras.layers.LayerNormalization,
    'group_norm': tfa.layers.GroupNormalization
  }


class BlockType(enum.Enum):
  Conv = 'Conv'
  DepSepConv = 'DepSepConv'
  InvertedResConv = 'InvertedResConv'
  FusedInvertedResConv = 'FusedInvertedResConv'
  GobalPooling = 'GlobalPooling'


@dataclass
class MobileNetBlockConfig(base_config.Config):
  """Configuration for a block of MobileNet model."""
  kernel: Tuple[int, int] = (3, 3)
  stride: int = 1
  filters: int = 32
  use_biase: bool = False
  normalization: bool = True
  activation_name: Text = 'relu6'
  # used for block type InvertedResConv
  expansion_size: float = 6.
  # used for block type InvertedResConv with SE
  squeeze_factor: int = None
  depthwise: bool = True
  residual: bool = True
  block_type: Text = BlockType.Conv.value


@dataclass
class MobileNetConfig(base_config.Config):
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
      finegrain_classification_mode: When set to True, the model
        will keep the last layer large even for small multipliers. Following
        https://arxiv.org/abs/1801.04381
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
  name: Text = 'MobileNet'
  num_classes: int = 1001
  # model specific
  min_depth: int = 8
  width_multiplier: float = 1.0
  output_stride: int = None
  finegrain_classification_mode: bool = False
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
  dropout_keep_prob: float = 0.8


@dataclass
class MobileNetV1Config(MobileNetConfig):
  """Configuration for the MobileNetV1 model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV1'
  width_multiplier: float = 1.0

  # regularization
  weight_decay: float = 0.00002
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.8

  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32,
      block_type=BlockType.Conv.value),
    # depthsep conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=128,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=128,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=256,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=256,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=512,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=512,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=512,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=512,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=512,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=512,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=1024,
      block_type=BlockType.DepSepConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=1024,
      block_type=BlockType.DepSepConv.value),
    # pylint: enable=bad-whitespace
  )


@dataclass
class MobileNetV2Config(MobileNetConfig):
  """Configuration for the MobileNetV2 model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV2'
  width_multiplier: float = 1.0
  finegrain_classification_mode: bool = True

  # regularization
  weight_decay: float = 0.00002
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.8

  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32,
      block_type=BlockType.Conv.value),
    # inverted res conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=16,
      expansion_size=1,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=24,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=24,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=64,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=64,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=160,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=160,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=160,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=320,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=1280,
      block_type=BlockType.Conv.value),
    # pylint: enable=bad-whitespace
  )


@dataclass
class MobileNetV3LargeConfig(MobileNetConfig):
  """Configuration for the MobileNetV3 Large model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV3Large'
  width_multiplier: float = 1.0
  finegrain_classification_mode: bool = True

  # regularization
  weight_decay: float = 0.00002
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.8

  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=16,
      activation_name='hard_swish',
      block_type=BlockType.Conv.value),

    # inverted res conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=16,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=1,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=24,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=24,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=3,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=2, filters=40,
      activation_name='relu',
      squeeze_factor=4,
      expansion_size=3,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=40,
      activation_name='relu',
      squeeze_factor=4,
      expansion_size=3,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=40,
      activation_name='relu',
      squeeze_factor=4,
      expansion_size=3,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=80,
      activation_name='hard_swish',
      squeeze_factor=None,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=80,
      activation_name='hard_swish',
      squeeze_factor=None,
      expansion_size=2.5,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=80,
      activation_name='hard_swish',
      squeeze_factor=None,
      expansion_size=2.3,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=80,
      activation_name='hard_swish',
      squeeze_factor=None,
      expansion_size=2.3,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=112,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=112,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=2, filters=160,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=160,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=160,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    # Last conv
    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=960,
      activation_name='hard_swish',
      block_type=BlockType.Conv.value),
    MobileNetBlockConfig.from_args(
      block_type=BlockType.GobalPooling.value),
    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=1280,
      activation_name='hard_swish',
      normalization=False,
      use_biase=True,
      block_type=BlockType.Conv.value),
    # pylint: enable=bad-whitespace
  )


@dataclass
class MobileNetV3SmallConfig(MobileNetConfig):
  """Configuration for the MobileNetV3 Small model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV3Small'
  width_multiplier: float = 1.0
  finegrain_classification_mode: bool = True

  # regularization
  weight_decay: float = 0.00002
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.8

  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=16,
      activation_name='hard_swish',
      block_type=BlockType.Conv.value),

    # inverted res conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=16,
      activation_name='relu',
      squeeze_factor=4,
      expansion_size=1,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=24,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=72. / 16,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=24,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=88. / 24,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=2, filters=40,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=40,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=40,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=48,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=3,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=48,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=3,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=2, filters=96,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=96,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=96,
      activation_name='hard_swish',
      squeeze_factor=4,
      expansion_size=6,
      block_type=BlockType.InvertedResConv.value),

    # Last conv
    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=576,
      activation_name='hard_swish',
      block_type=BlockType.Conv.value),
    MobileNetBlockConfig.from_args(
      block_type=BlockType.GobalPooling.value),
    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=1024,
      activation_name='hard_swish',
      normalization=False,
      use_biase=True,
      block_type=BlockType.Conv.value),
    # pylint: enable=bad-whitespace
  )

@dataclass
class MobileNetV3EdgeTPUConfig(MobileNetConfig):
  """Configuration for the MobileNetV3 Edge TPU model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV3EdgeTPU'
  width_multiplier: float = 1.0
  finegrain_classification_mode: bool = True

  # regularization
  weight_decay: float = 0.00002
  stddev: float = 0.09
  regularize_depthwise: bool = False
  # activation
  activation_name: Text = 'relu6'
  # normalization
  normalization_name: Text = 'batch_norm'
  batch_norm_decay: float = 0.9997
  batch_norm_epsilon: float = 0.001
  # dropout
  dropout_keep_prob: float = 0.8

  # base architecture
  blocks: Tuple[MobileNetBlockConfig, ...] = (
    # (kernel, stride, depth)
    # pylint: disable=bad-whitespace
    # base normal conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32,
      activation_name='relu',
      block_type=BlockType.Conv.value),

    # inverted res conv
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=16,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=1,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=32,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=8,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=32,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=48,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=8,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=48,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=48,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=48,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      depthwise=False,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=2, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=8,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=8,
      residule=False,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=96,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=2, filters=160,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=8,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=160,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=160,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),
    MobileNetBlockConfig.from_args(
      kernel=(5, 5), stride=1, filters=160,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=4,
      block_type=BlockType.InvertedResConv.value),

    MobileNetBlockConfig.from_args(
      kernel=(3, 3), stride=1, filters=192,
      activation_name='relu',
      squeeze_factor=None,
      expansion_size=8,
      block_type=BlockType.InvertedResConv.value),

    # Last conv
    MobileNetBlockConfig.from_args(
      kernel=(1, 1), stride=1, filters=1280,
      activation_name='relu',
      block_type=BlockType.Conv.value),
    # pylint: enable=bad-whitespace
  )