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

"""Contains definitions for MobilenetEdgeTPUV2 model's building blocks."""
import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple, Union
# Import libraries
from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import oneof
from official.projects.edgetpu.vision.modeling import common_modules
from official.projects.edgetpu.vision.modeling import custom_layers

InitializerType = Optional[Union[str, tf_keras.initializers.Initializer]]


@dataclasses.dataclass
class BlockType(oneof.OneOfConfig):
  """Block OP types representing IBN version."""
  type: str = 'ibn_dw'
  skip: str = 'skip'
  ibn_dw: str = 'ibn_dw'
  ibn_fused: str = 'ibn_fused'
  ibn_grouped: str = 'ibn_grouped'
  ibn_fused_grouped: str = 'ibn_fused_grouped'


@dataclasses.dataclass
class BlockSearchConfig(base_config.Config):
  """Config for searchable BlockConfig parameters."""
  op_type: BlockType = dataclasses.field(default_factory=BlockType)
  kernel_size: Optional[int] = None
  expand_ratio: Optional[int] = None
  stride: Optional[int] = None
  group_size: Optional[int] = None


@dataclasses.dataclass
class BlockConfig(base_config.Config):
  """Full config for a single MB Conv Block."""
  input_filters: int = 0
  output_filters: int = 0
  kernel_size: int = 3
  num_repeat: int = 1
  expand_ratio: int = 1
  strides: Tuple[int, int] = (1, 1)
  se_ratio: Optional[float] = None
  id_skip: bool = True
  fused_expand: bool = False
  fused_project: bool = False
  conv_type: str = 'depthwise'
  group_size: Optional[int] = None

  @classmethod
  def from_search_config(cls,
                         input_filters: int,
                         output_filters: int,
                         block_search_config: BlockSearchConfig,
                         num_repeat: int = 1,
                         se_ratio: Optional[float] = None,
                         id_skip: bool = True) -> 'BlockConfig':
    """Creates BlockConfig from the given parameters."""
    block_op_type = block_search_config.op_type

    if block_op_type.type == BlockType.skip:
      raise ValueError('Received skip type within block creation.')
    elif block_op_type.type == BlockType.ibn_dw:
      fused_expand = False
      fused_project = False
      conv_type = 'depthwise'
    elif block_op_type.type == BlockType.ibn_fused:
      fused_expand = True
      fused_project = False
      conv_type = 'full'
    elif block_op_type.type == BlockType.ibn_fused_grouped:
      fused_expand = True
      fused_project = False
      conv_type = 'group'
    elif block_op_type.type == BlockType.ibn_grouped:
      fused_expand = False
      fused_project = False
      conv_type = 'group'
    else:
      raise NotImplementedError(f'Unsupported IBN type {block_op_type.type}.')

    return cls.from_args(
        input_filters=input_filters,
        output_filters=output_filters,
        kernel_size=block_search_config.kernel_size,
        num_repeat=num_repeat,
        expand_ratio=block_search_config.expand_ratio,
        strides=(block_search_config.stride, block_search_config.stride),
        se_ratio=se_ratio,
        id_skip=id_skip,
        fused_expand=fused_expand,
        fused_project=fused_project,
        conv_type=conv_type,
        group_size=block_search_config.group_size)


@dataclasses.dataclass
class BlockGroupConfig(base_config.Config):
  """Config for group of blocks that share the same filter size."""
  blocks: List[BlockSearchConfig] = dataclasses.field(default_factory=list)
  filters: int = 64


def _default_mobilenet_edgetpu_v2_topology():
  return [
      # Block Group 0
      BlockGroupConfig(
          blocks=[
              # BlockSearchConfig: op_type, kernel_size, expand_ratio, stride
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused'), 3, 1, 1),
          ],
          filters=24),
      # Block Group 1
      BlockGroupConfig(
          blocks=[
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused'), 3, 8, 2),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused_grouped'), 3, 4, 1),
          ],
          filters=48),
      # Block Group 2
      BlockGroupConfig(
          blocks=[
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused'), 3, 8, 2),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused_grouped'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused_grouped'), 3, 4, 1),
          ],
          filters=64),
      # Block Group 3
      BlockGroupConfig(
          blocks=[
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_fused'), 3, 8, 2),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
          ],
          filters=128),
      # Block Group 4
      BlockGroupConfig(
          blocks=[
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 8, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
          ],
          filters=160),
      # Block Group 5
      BlockGroupConfig(
          blocks=[
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 8, 2),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 4, 1),
          ],
          filters=192),
      # Block Group 6
      BlockGroupConfig(
          blocks=[
              BlockSearchConfig.from_args(
                  BlockType.from_args('ibn_dw'), 3, 8, 1),
          ],
          filters=256),
  ]


@dataclasses.dataclass
class TopologyConfig(base_config.Config):
  """Config for model topology as a collection of BlockGroupConfigs."""
  block_groups: List[BlockGroupConfig] = dataclasses.field(
      default_factory=_default_mobilenet_edgetpu_v2_topology)


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """Default Config for MobilenetEdgeTPUV2."""
  width_coefficient: float = 1.0
  depth_coefficient: float = 1.0
  resolution: Union[int, Tuple[int, int]] = 224
  dropout_rate: float = 0.1
  stem_base_filters: int = 64
  stem_kernel_size: int = 5
  top_base_filters: int = 1280
  conv_kernel_initializer: InitializerType = None
  dense_kernel_initializer: InitializerType = None
  blocks: Tuple[BlockConfig, ...] = (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio, id_skip, fused_conv, conv_type)
      # pylint: disable=bad-whitespace
      BlockConfig.from_args(
          stem_base_filters, 24, 3, 1, 1, (1, 1), conv_type='full'),
      BlockConfig.from_args(
          24, 48, 3, 1, 8, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(
          48, 48, 3, 1, 4, (1, 1), fused_expand=True, conv_type='group'),
      BlockConfig.from_args(
          48, 64, 3, 1, 8, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(
          64, 64, 3, 1, 4, (1, 1), fused_expand=True, conv_type='group'),
      BlockConfig.from_args(
          64, 64, 3, 1, 4, (1, 1), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(
          64, 64, 3, 1, 4, (1, 1), fused_expand=True, conv_type='group'),
      BlockConfig.from_args(
          64, 128, 3, 1, 8, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(128, 128, 3, 3, 4, (1, 1)),
      BlockConfig.from_args(128, 160, 3, 1, 8, (1, 1)),
      BlockConfig.from_args(160, 160, 3, 3, 4, (1, 1)),
      BlockConfig.from_args(160, 192, 5, 1, 8, (2, 2)),
      BlockConfig.from_args(192, 192, 5, 3, 4, (1, 1)),
      BlockConfig.from_args(192, 256, 5, 1, 8, (1, 1)),
      # pylint: enable=bad-whitespace
  )
  activation: str = 'relu'
  batch_norm: str = 'default'
  bn_momentum: float = 0.99
  bn_epsilon: float = 1e-3
  # While the original implementation used a weight decay of 1e-5,
  # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
  weight_decay: float = 5e-6
  drop_connect_rate: float = 0.1
  depth_divisor: int = 8
  min_depth: Optional[int] = None
  # No Squeeze/Excite for MobilenetEdgeTPUV2
  use_se: bool = False
  input_channels: int = 3
  num_classes: int = 1001
  model_name: str = 'mobilenet_edgetpu_v2'
  rescale_input: bool = False
  data_format: str = 'channels_last'
  dtype: str = 'float32'
  # The number of filters in each group. HW arch dependent.
  group_base_size: int = 64
  backbone_only: bool = False
  features_as_dict: bool = False


def mobilenet_edgetpu_v2_base(
    width_coefficient: float = 1.0,
    depth_coefficient: float = 1.0,
    stem_base_filters: int = 64,
    stem_kernel_size: int = 5,
    top_base_filters: int = 1280,
    group_base_size: int = 64,
    dropout_rate: float = 0.2,
    drop_connect_rate: float = 0.1,
    filter_size_overrides: Optional[Dict[int, int]] = None,
    block_op_overrides: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    block_group_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
    topology: Optional[TopologyConfig] = None):
  """Creates MobilenetEdgeTPUV2 ModelConfig based on tuning parameters."""

  config = ModelConfig()
  param_overrides = {
      'width_coefficient': width_coefficient,
      'depth_coefficient': depth_coefficient,
      'stem_base_filters': stem_base_filters,
      'stem_kernel_size': stem_kernel_size,
      'top_base_filters': top_base_filters,
      'group_base_size': group_base_size,
      'dropout_rate': dropout_rate,
      'drop_connect_rate': drop_connect_rate
  }
  config = config.replace(**param_overrides)

  topology_config = TopologyConfig() if topology is None else topology
  if filter_size_overrides:
    for group_id in filter_size_overrides:
      topology_config.block_groups[group_id].filters = filter_size_overrides[
          group_id]

  if block_op_overrides:
    for group_id in block_op_overrides:
      for block_id in block_op_overrides[group_id]:
        replaced_block = topology_config.block_groups[group_id].blocks[
            block_id].replace(**block_op_overrides[group_id][block_id])
        topology_config.block_groups[group_id].blocks[block_id] = replaced_block

  if block_group_overrides:
    for group_id in block_group_overrides:
      replaced_group = topology_config.block_groups[group_id].replace(
          **block_group_overrides[group_id])
      topology_config.block_groups[group_id] = replaced_group

  blocks = ()
  input_filters = stem_base_filters

  for group in topology_config.block_groups:
    for block_search in group.blocks:
      if block_search.op_type != BlockType.skip:
        block = BlockConfig.from_search_config(
            input_filters=input_filters,
            output_filters=group.filters,
            block_search_config=block_search)
        blocks += (block,)
        # Set input filters for the next block
        input_filters = group.filters

  config = config.replace(blocks=blocks)

  return config


def autoseg_edgetpu_backbone_base(
    width_coefficient: float = 1.0,
    depth_coefficient: float = 1.0,
    stem_base_filters: int = 64,
    stem_kernel_size: int = 5,
    top_base_filters: int = 1280,
    group_base_size: int = 64,
    dropout_rate: float = 0.2,
    drop_connect_rate: float = 0.1,
    blocks_overrides: Optional[Tuple[BlockConfig, ...]] = None):
  """Creates a edgetpu ModelConfig based on search on segmentation."""

  config = ModelConfig()
  config.depth_divisor = 4
  param_overrides = {
      'width_coefficient': width_coefficient,
      'depth_coefficient': depth_coefficient,
      'stem_base_filters': stem_base_filters,
      'stem_kernel_size': stem_kernel_size,
      'top_base_filters': top_base_filters,
      'group_base_size': group_base_size,
      'dropout_rate': dropout_rate,
      'drop_connect_rate': drop_connect_rate,
  }
  if blocks_overrides:
    param_overrides['blocks'] = blocks_overrides
  config = config.replace(**param_overrides)
  return config


def autoseg_edgetpu_backbone_s() -> ModelConfig:
  """AutoML searched model with 2.5ms target simulated latency."""
  stem_base_filters = 32
  stem_kernel_size = 3
  top_base_filters = 1280
  blocks = (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio, id_skip, fused_conv, conv_type)
      # pylint: disable=bad-whitespace
      BlockConfig.from_args(
          stem_base_filters,
          12,
          3,
          1,
          1, (1, 1),
          fused_expand=True,
          conv_type='full'),
      BlockConfig.from_args(
          12, 36, 3, 1, 6, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(36, 18, 5, 1, 3, (1, 1)),
      BlockConfig.from_args(
          18, 60, 5, 1, 6, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(60, 60, 3, 1, 3, (1, 1)),
      BlockConfig.from_args(60, 120, 5, 1, 6, (2, 2)),
      BlockConfig.from_args(120, 120, 3, 1, 3, (1, 1)),
      BlockConfig.from_args(120, 120, 5, 1, 6, (1, 1)),
      BlockConfig.from_args(120, 112, 3, 1, 6, (1, 1)),
      BlockConfig.from_args(112, 112, 5, 2, 6, (1, 1)),
      BlockConfig.from_args(112, 112, 5, 1, 1, (2, 2), id_skip=False),
      BlockConfig.from_args(
          112, 192, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(192, 192, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          192, 96, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(96, 96, 5, 1, 3, (1, 1)),
      BlockConfig.from_args(96, 96, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          96, 192, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(192, 192, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          192, 160, 1, 1, 3, (1, 1), fused_expand=True, id_skip=False),
      # pylint: enable=bad-whitespace
  )
  return autoseg_edgetpu_backbone_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      blocks_overrides=blocks,
      dropout_rate=0.2,
      drop_connect_rate=0.2)


def autoseg_edgetpu_backbone_xs() -> ModelConfig:
  """AutoML searched model with 2ms target simulated latency."""
  stem_base_filters = 32
  stem_kernel_size = 3
  top_base_filters = 1280
  blocks = (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio, id_skip, fused_conv, conv_type)
      # pylint: disable=bad-whitespace
      BlockConfig.from_args(
          stem_base_filters,
          12,
          3,
          1,
          1, (1, 1),
          fused_expand=True,
          conv_type='full'),
      BlockConfig.from_args(
          12, 24, 3, 1, 6, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(24, 24, 3, 1, 3, (1, 1)),
      BlockConfig.from_args(
          24, 60, 3, 1, 3, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(60, 40, 3, 1, 6, (1, 1)),
      BlockConfig.from_args(40, 40, 5, 1, 3, (2, 2)),
      BlockConfig.from_args(40, 40, 3, 1, 6, (1, 1)),
      BlockConfig.from_args(
          40, 120, 3, 1, 6, (1, 1), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(120, 168, 3, 1, 6, (1, 1)),
      BlockConfig.from_args(168, 84, 5, 1, 6, (1, 1)),
      BlockConfig.from_args(84, 84, 5, 1, 3, (1, 1)),

      BlockConfig.from_args(84, 84, 5, 1, 1, (2, 2), id_skip=False),
      BlockConfig.from_args(
          84, 288, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),

      BlockConfig.from_args(288, 288, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          288, 96, 1, 1, 3, (1, 1), fused_expand=True, id_skip=False),

      BlockConfig.from_args(96, 96, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          96, 96, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(96, 96, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          96, 96, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(96, 480, 5, 1, 3, (1, 1)),
      # pylint: enable=bad-whitespace
  )
  return autoseg_edgetpu_backbone_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      blocks_overrides=blocks,
      dropout_rate=0.2,
      drop_connect_rate=0.2)


def autoseg_edgetpu_backbone_m() -> ModelConfig:
  """AutoML searched model with 3ms target simulated latency."""
  stem_base_filters = 32
  stem_kernel_size = 3
  top_base_filters = 1280
  blocks = (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio, id_skip, fused_conv, conv_type)
      # pylint: disable=bad-whitespace
      BlockConfig.from_args(stem_base_filters, 16, 5, 1, 1, (1, 1)),
      BlockConfig.from_args(
          16, 36, 3, 1, 6, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(36, 36, 3, 1, 3, (1, 1)),
      BlockConfig.from_args(
          36, 60, 3, 1, 6, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(60, 60, 3, 1, 6, (1, 1)),
      BlockConfig.from_args(
          60, 120, 5, 1, 6, (2, 2), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(120, 120, 5, 1, 6, (1, 1)),
      BlockConfig.from_args(
          120, 80, 3, 1, 6, (1, 1), fused_expand=True, conv_type='full'),
      BlockConfig.from_args(80, 168, 3, 1, 6, (1, 1)),
      BlockConfig.from_args(168, 168, 5, 1, 6, (1, 1)),
      BlockConfig.from_args(168, 168, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          168, 168, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(168, 168, 3, 1, 1, (2, 2), id_skip=False),
      BlockConfig.from_args(
          168, 192, 1, 1, 3, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(192, 192, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          192, 288, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(288, 288, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          288, 96, 1, 1, 6, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(96, 96, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          96, 192, 1, 1, 3, (1, 1), fused_expand=True, id_skip=False),
      BlockConfig.from_args(192, 192, 5, 1, 1, (1, 1), id_skip=False),
      BlockConfig.from_args(
          192, 320, 1, 1, 3, (1, 1), fused_expand=True, id_skip=False),
      # pylint: enable=bad-whitespace
  )
  return autoseg_edgetpu_backbone_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      blocks_overrides=blocks,
      dropout_rate=0.3,
      drop_connect_rate=0.3)


def mobilenet_edgetpu_v2_tiny() -> ModelConfig:
  """MobilenetEdgeTPUV2 tiny model config."""
  stem_base_filters = 32
  stem_kernel_size = 5
  top_base_filters = 1280
  filter_sizes = [16, 32, 48, 80, 112, 160, 192]
  filter_size_overrides = {
      k: v for (k, v) in zip(range(len(filter_sizes)), filter_sizes)
  }
  block_op_overrides = {
      2: {
          0: {'op_type': BlockType.from_args('ibn_fused_grouped')},
          2: {'op_type': BlockType.from_args('ibn_fused_grouped')},
      },
      3: {
          0: {'op_type': BlockType.from_args('ibn_fused_grouped')},
      }
  }

  return mobilenet_edgetpu_v2_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      filter_size_overrides=filter_size_overrides,
      block_op_overrides=block_op_overrides,
      dropout_rate=0.05,
      drop_connect_rate=0.05)


def mobilenet_edgetpu_v2_xs() -> ModelConfig:
  """MobilenetEdgeTPUV2 extra small model config."""
  stem_base_filters = 32
  stem_kernel_size = 5
  top_base_filters = 1280
  filter_sizes = [16, 32, 48, 96, 144, 160, 192]
  filter_size_overrides = {
      k: v for (k, v) in zip(range(len(filter_sizes)), filter_sizes)
  }

  return mobilenet_edgetpu_v2_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      filter_size_overrides=filter_size_overrides,
      dropout_rate=0.05,
      drop_connect_rate=0.05)


def mobilenet_edgetpu_v2_s():
  """MobilenetEdgeTPUV2 small model config."""
  stem_base_filters = 64
  stem_kernel_size = 5
  top_base_filters = 1280
  filter_sizes = [24, 48, 64, 128, 160, 192, 256]
  filter_size_overrides = {
      k: v for (k, v) in zip(range(len(filter_sizes)), filter_sizes)
  }

  return mobilenet_edgetpu_v2_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      filter_size_overrides=filter_size_overrides)


def mobilenet_edgetpu_v2_m():
  """MobilenetEdgeTPUV2 medium model config."""
  stem_base_filters = 64
  stem_kernel_size = 5
  top_base_filters = 1344
  filter_sizes = [32, 64, 80, 160, 192, 240, 320]
  filter_size_overrides = {
      k: v for (k, v) in zip(range(len(filter_sizes)), filter_sizes)
  }

  return mobilenet_edgetpu_v2_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      filter_size_overrides=filter_size_overrides)


def mobilenet_edgetpu_v2_l():
  """MobilenetEdgeTPUV2 large model config."""
  stem_base_filters = 64
  stem_kernel_size = 7
  top_base_filters = 1408
  filter_sizes = [32, 64, 96, 192, 240, 256, 384]
  filter_size_overrides = {
      k: v for (k, v) in zip(range(len(filter_sizes)), filter_sizes)
  }
  group_base_size = 128

  return mobilenet_edgetpu_v2_base(
      stem_base_filters=stem_base_filters,
      stem_kernel_size=stem_kernel_size,
      top_base_filters=top_base_filters,
      group_base_size=group_base_size,
      filter_size_overrides=filter_size_overrides)


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # Note: this is a truncated normal distribution
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1 / 3.0,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def round_filters(filters: int,
                  config: ModelConfig) -> int:
  """Round number of filters based on width coefficient."""
  width_coefficient = config.width_coefficient
  min_depth = config.min_depth
  divisor = config.depth_divisor
  orig_filters = filters

  if not width_coefficient:
    return filters

  filters *= width_coefficient
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  logging.info('round_filter input=%s output=%s', orig_filters, new_filters)
  return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
  """Round number of repeats based on depth coefficient."""
  return int(math.ceil(depth_coefficient * repeats))


def groupconv2d_block(conv_filters: Optional[int],
                      config: ModelConfig,
                      kernel_size: Any = (1, 1),
                      strides: Any = (1, 1),
                      group_size: Optional[int] = None,
                      use_batch_norm: bool = True,
                      use_bias: bool = False,
                      activation: Any = None,
                      name: Optional[str] = None) -> tf_keras.layers.Layer:
  """2D group convolution with batchnorm and activation."""
  batch_norm = common_modules.get_batch_norm(config.batch_norm)
  bn_momentum = config.bn_momentum
  bn_epsilon = config.bn_epsilon
  data_format = tf_keras.backend.image_data_format()
  weight_decay = config.weight_decay
  if group_size is None:
    group_size = config.group_base_size

  name = name or ''
  # Compute the # of groups
  if conv_filters % group_size != 0:
    raise ValueError(f'Number of filters: {conv_filters} is not divisible by '
                     f'size of the groups: {group_size}')
  groups = int(conv_filters / group_size)
  # Collect args based on what kind of groupconv2d block is desired
  init_kwargs = {
      'kernel_size': kernel_size,
      'strides': strides,
      'use_bias': use_bias,
      'padding': 'same',
      'name': name + '_groupconv2d',
      'kernel_regularizer': tf_keras.regularizers.l2(weight_decay),
      'bias_regularizer': tf_keras.regularizers.l2(weight_decay),
      'filters': conv_filters,
      'groups': groups,
      'batch_norm_layer': batch_norm if use_batch_norm else None,
      'bn_epsilon': bn_epsilon,
      'bn_momentum': bn_momentum,
      'activation': activation,
      'data_format': data_format,
  }
  return custom_layers.GroupConv2D(**init_kwargs)


def conv2d_block_as_layers(
    conv_filters: Optional[int],
    config: ModelConfig,
    kernel_size: Any = (1, 1),
    strides: Any = (1, 1),
    use_batch_norm: bool = True,
    use_bias: bool = False,
    activation: Any = None,
    depthwise: bool = False,
    kernel_initializer: InitializerType = None,
    name: Optional[str] = None) -> List[tf_keras.layers.Layer]:
  """A conv2d followed by batch norm and an activation."""
  batch_norm = common_modules.get_batch_norm(config.batch_norm)
  bn_momentum = config.bn_momentum
  bn_epsilon = config.bn_epsilon
  data_format = tf_keras.backend.image_data_format()
  weight_decay = config.weight_decay

  name = name or ''

  # Collect args based on what kind of conv2d block is desired
  init_kwargs = {
      'kernel_size': kernel_size,
      'strides': strides,
      'use_bias': use_bias,
      'padding': 'same',
      'name': name + '_conv2d',
      'kernel_regularizer': tf_keras.regularizers.l2(weight_decay),
      'bias_regularizer': tf_keras.regularizers.l2(weight_decay),
  }

  sequential_layers: List[tf_keras.layers.Layer] = []
  if depthwise:
    conv2d = tf_keras.layers.DepthwiseConv2D
    init_kwargs.update({'depthwise_initializer': kernel_initializer})
  else:
    conv2d = tf_keras.layers.Conv2D
    init_kwargs.update({
        'filters': conv_filters,
        'kernel_initializer': kernel_initializer
    })

  sequential_layers.append(conv2d(**init_kwargs))

  if use_batch_norm:
    bn_axis = 1 if data_format == 'channels_first' else -1
    sequential_layers.append(
        batch_norm(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + '_bn'))

  if activation is not None:
    sequential_layers.append(
        tf_keras.layers.Activation(activation, name=name + '_activation'))
  return sequential_layers


def conv2d_block(inputs: tf.Tensor,
                 conv_filters: Optional[int],
                 config: ModelConfig,
                 kernel_size: Any = (1, 1),
                 strides: Any = (1, 1),
                 use_batch_norm: bool = True,
                 use_bias: bool = False,
                 activation: Any = None,
                 depthwise: bool = False,
                 kernel_initializer: Optional[InitializerType] = None,
                 name: Optional[str] = None) -> tf.Tensor:
  """Compatibility with third_party/car/deep_nets."""
  x = inputs
  for layer in conv2d_block_as_layers(
      conv_filters=conv_filters,
      config=config,
      kernel_size=kernel_size,
      strides=strides,
      use_batch_norm=use_batch_norm,
      use_bias=use_bias,
      activation=activation,
      depthwise=depthwise,
      kernel_initializer=kernel_initializer,
      name=name):
    x = layer(x)
  return x


# Do not inherit from (tf_keras.layers.Layer), will break weights loading.
class _MbConvBlock:
  """Mobile Inverted Residual Bottleneck composite layer."""

  def __call__(self, inputs: tf.Tensor, training=False):
    x = inputs
    for layer in self.expand_block:
      x = layer(x)
    if self.squeeze_excitation:
      se = x
      for layer in self.squeeze_excitation:
        se = layer(se)
      x = tf_keras.layers.multiply([x, se], name=self.name + 'se_excite')
    for layer in self.project_block:
      x = layer(x)
    if self.has_skip_add:
      x = tf_keras.layers.add([x, inputs], name=self.name + 'add')
    return x

  def __init__(self,
               block: BlockConfig,
               config: ModelConfig,
               prefix: Optional[str] = None):
    """Mobile Inverted Residual Bottleneck.

    Args:
      block: BlockConfig, arguments to create a Block
      config: ModelConfig, a set of model parameters
      prefix: prefix for naming all layers
    """
    use_se = config.use_se
    activation = tf_utils.get_activation(config.activation)
    drop_connect_rate = config.drop_connect_rate
    data_format = tf_keras.backend.image_data_format()
    use_depthwise = block.conv_type == 'depthwise'
    use_groupconv = block.conv_type == 'group'
    prefix = prefix or ''
    self.name = prefix
    conv_kernel_initializer = (
        config.conv_kernel_initializer if config.conv_kernel_initializer
        is not None else CONV_KERNEL_INITIALIZER)

    filters = block.input_filters * block.expand_ratio

    self.expand_block: List[tf_keras.layers.Layer] = []
    self.squeeze_excitation: List[tf_keras.layers.Layer] = []
    self.project_block: List[tf_keras.layers.Layer] = []

    if block.fused_project:
      raise NotImplementedError('Fused projection is not supported.')

    if block.fused_expand and block.expand_ratio != 1:
      # If we use fused mbconv, fuse expansion with the main kernel.
      # If conv_type is depthwise we still fuse it to a full conv.
      if use_groupconv:
        self.expand_block.append(groupconv2d_block(
            filters,
            config,
            kernel_size=block.kernel_size,
            strides=block.strides,
            group_size=block.group_size,
            activation=activation,
            name=prefix + 'fused'))
      else:
        self.expand_block.extend(
            conv2d_block_as_layers(
                conv_filters=filters,
                config=config,
                kernel_size=block.kernel_size,
                strides=block.strides,
                activation=activation,
                kernel_initializer=conv_kernel_initializer,
                name=prefix + 'fused'))
    else:
      if block.expand_ratio != 1:
        # Expansion phase with a pointwise conv
        self.expand_block.extend(
            conv2d_block_as_layers(
                conv_filters=filters,
                config=config,
                kernel_size=(1, 1),
                activation=activation,
                kernel_initializer=conv_kernel_initializer,
                name=prefix + 'expand'))

      # Main kernel, after the expansion (if applicable, i.e. not fused).
      if use_depthwise:
        self.expand_block.extend(conv2d_block_as_layers(
            conv_filters=filters,
            config=config,
            kernel_size=block.kernel_size,
            strides=block.strides,
            activation=activation,
            kernel_initializer=conv_kernel_initializer,
            depthwise=True,
            name=prefix + 'depthwise'))
      elif use_groupconv:
        self.expand_block.append(groupconv2d_block(
            conv_filters=filters,
            config=config,
            kernel_size=block.kernel_size,
            strides=block.strides,
            group_size=block.group_size,
            activation=activation,
            name=prefix + 'group'))

    # Squeeze and Excitation phase
    if use_se:
      assert block.se_ratio is not None
      assert 0 < block.se_ratio <= 1
      num_reduced_filters = max(1, int(
          block.input_filters * block.se_ratio
      ))

      if data_format == 'channels_first':
        se_shape = (filters, 1, 1)
      else:
        se_shape = (1, 1, filters)

      self.squeeze_excitation.append(
          tf_keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze'))
      self.squeeze_excitation.append(
          tf_keras.layers.Reshape(se_shape, name=prefix + 'se_reshape'))
      self.squeeze_excitation.extend(
          conv2d_block_as_layers(
              conv_filters=num_reduced_filters,
              config=config,
              use_bias=True,
              use_batch_norm=False,
              activation=activation,
              kernel_initializer=conv_kernel_initializer,
              name=prefix + 'se_reduce'))
      self.squeeze_excitation.extend(
          conv2d_block_as_layers(
              conv_filters=filters,
              config=config,
              use_bias=True,
              use_batch_norm=False,
              activation='sigmoid',
              kernel_initializer=conv_kernel_initializer,
              name=prefix + 'se_expand'))

    # Output phase
    self.project_block.extend(
        conv2d_block_as_layers(
            conv_filters=block.output_filters,
            config=config,
            activation=None,
            kernel_initializer=conv_kernel_initializer,
            name=prefix + 'project'))

    # Add identity so that quantization-aware training can insert quantization
    # ops correctly.
    self.project_block.append(
        tf_keras.layers.Activation('linear', name=prefix + 'id'))

    self.has_skip_add = False
    if (block.id_skip
        and all(s == 1 for s in block.strides)
        and block.input_filters == block.output_filters):
      self.has_skip_add = True
      if drop_connect_rate and drop_connect_rate > 0:
        # Apply dropconnect
        # The only difference between dropout and dropconnect in TF is scaling
        # by drop_connect_rate during training. See:
        # https://github.com/keras-team/keras/pull/9898#issuecomment-380577612
        self.project_block.append(
            tf_keras.layers.Dropout(
                drop_connect_rate,
                noise_shape=(None, 1, 1, 1),
                name=prefix + 'drop'))


def mb_conv_block(inputs: tf.Tensor,
                  block: BlockConfig,
                  config: ModelConfig,
                  prefix: Optional[str] = None) -> tf.Tensor:
  """Mobile Inverted Residual Bottleneck.

  Args:
    inputs: the Keras input to the block
    block: BlockConfig, arguments to create a Block
    config: ModelConfig, a set of model parameters
    prefix: prefix for naming all layers

  Returns:
    the output of the block
  """
  return _MbConvBlock(block, config, prefix)(inputs)


def mobilenet_edgetpu_v2(image_input: tf_keras.layers.Input,
                         config: ModelConfig):  # pytype: disable=invalid-annotation  # typed-keras
  """Creates a MobilenetEdgeTPUV2 graph given the model parameters.

  This function is wrapped by the `MobilenetEdgeTPUV2` class to make a
  tf_keras.Model.

  Args:
    image_input: the input batch of images
    config: the model config

  Returns:
    The output of classification model or if backbone is needed, dictionary with
    backbone feature levels.
  """
  depth_coefficient = config.depth_coefficient
  blocks = config.blocks
  stem_base_filters = config.stem_base_filters
  stem_kernel_size = config.stem_kernel_size
  top_base_filters = config.top_base_filters
  activation = tf_utils.get_activation(config.activation)
  dropout_rate = config.dropout_rate
  drop_connect_rate = config.drop_connect_rate
  conv_kernel_initializer = (
      config.conv_kernel_initializer if config.conv_kernel_initializer
      is not None else CONV_KERNEL_INITIALIZER)
  dense_kernel_initializer = (
      config.dense_kernel_initializer if config.dense_kernel_initializer
      is not None else DENSE_KERNEL_INITIALIZER)
  num_classes = config.num_classes
  input_channels = config.input_channels
  rescale_input = config.rescale_input
  data_format = tf_keras.backend.image_data_format()
  dtype = config.dtype
  weight_decay = config.weight_decay

  x = image_input
  if data_format == 'channels_first':
    # Happens on GPU/TPU if available.
    x = tf_keras.layers.Permute((3, 1, 2))(x)
  if rescale_input:
    x = common_modules.normalize_images(
        x, num_channels=input_channels, dtype=dtype, data_format=data_format)

  # Build stem
  x = conv2d_block(
      inputs=x,
      conv_filters=round_filters(stem_base_filters, config),
      config=config,
      kernel_size=[stem_kernel_size, stem_kernel_size],
      strides=[2, 2],
      activation=activation,
      kernel_initializer=conv_kernel_initializer,
      name='stem')

  # Build blocks
  num_blocks_total = sum(block.num_repeat for block in blocks)
  block_num = 0

  backbone_levels = []
  for stack_idx, block in enumerate(blocks):
    is_reduction = False
    assert block.num_repeat > 0
    # Update block input and output filters based on depth multiplier
    block = block.replace(
        input_filters=round_filters(block.input_filters, config),
        output_filters=round_filters(block.output_filters, config),
        num_repeat=round_repeats(block.num_repeat, depth_coefficient))

    if stack_idx == 0:
      backbone_levels.append(x)
    elif (stack_idx == len(blocks) - 1) or (blocks[stack_idx + 1].strides
                                            == (2, 2)):
      is_reduction = True
    # The first block needs to take care of stride and filter size increase
    drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
    config = config.replace(drop_connect_rate=drop_rate)
    block_prefix = 'stack_{}/block_0/'.format(stack_idx)
    x = _MbConvBlock(block, config, block_prefix)(x)
    block_num += 1
    if block.num_repeat > 1:
      block = block.replace(
          input_filters=block.output_filters,
          strides=[1, 1]
      )

      for block_idx in range(block.num_repeat - 1):
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        config = config.replace(drop_connect_rate=drop_rate)
        block_prefix = 'stack_{}/block_{}/'.format(stack_idx, block_idx + 1)
        x = _MbConvBlock(block, config, prefix=block_prefix)(x)
        block_num += 1
    if is_reduction:
      backbone_levels.append(x)

  if config.backbone_only:
    return backbone_levels
  # Build top
  x = conv2d_block(
      inputs=x,
      conv_filters=round_filters(top_base_filters, config),
      config=config,
      activation=activation,
      kernel_initializer=conv_kernel_initializer,
      name='top')

  # Build classifier
  pool_size = (x.shape.as_list()[1], x.shape.as_list()[2])
  x = tf_keras.layers.AveragePooling2D(pool_size, name='top_pool')(x)
  if dropout_rate and dropout_rate > 0:
    x = tf_keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
  x = tf_keras.layers.Conv2D(
      num_classes,
      1,
      kernel_initializer=dense_kernel_initializer,
      kernel_regularizer=tf_keras.regularizers.l2(weight_decay),
      bias_regularizer=tf_keras.regularizers.l2(weight_decay),
      name='logits')(
          x)
  x = tf_keras.layers.Activation('softmax', name='probs')(x)
  x = tf.squeeze(x, axis=[1, 2])

  return x
