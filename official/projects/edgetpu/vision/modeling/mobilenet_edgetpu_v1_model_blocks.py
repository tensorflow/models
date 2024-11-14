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

"""Contains definitions for MobilenetEdgeTPU image classification models."""
import dataclasses
import math
from typing import Any, Optional, Tuple, Union

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.projects.edgetpu.vision.modeling import common_modules


@dataclasses.dataclass
class BlockConfig(base_config.Config):
  """Config for a single MB Conv Block."""
  input_filters: int = 0
  output_filters: int = 0
  kernel_size: int = 3
  num_repeat: int = 1
  expand_ratio: int = 1
  strides: Tuple[int, int] = (1, 1)
  se_ratio: Optional[float] = None
  id_skip: bool = True
  fused_conv: bool = False
  conv_type: str = 'depthwise'


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """Default Config for MobilenetEdgeTPU."""
  width_coefficient: float = 1.0
  depth_coefficient: float = 1.0
  resolution: Union[int, Tuple[int, int]] = 224
  dropout_rate: float = 0.1
  blocks: Tuple[BlockConfig, ...] = (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides, se_ratio, id_skip, fused_conv, conv_type)
      # pylint: disable=bad-whitespace
      BlockConfig.from_args(32, 16, 3, 1, 1, (1, 1), conv_type='no_depthwise'),
      BlockConfig.from_args(16, 32, 3, 1, 8, (2, 2), fused_conv=True),
      BlockConfig.from_args(32, 32, 3, 3, 4, (1, 1), conv_type='no_depthwise'),
      BlockConfig.from_args(32, 48, 3, 1, 8, (2, 2), fused_conv=True),
      BlockConfig.from_args(48, 48, 3, 3, 4, (1, 1), conv_type='no_depthwise'),
      BlockConfig.from_args(48, 96, 3, 1, 8, (2, 2)),
      BlockConfig.from_args(96, 96, 3, 3, 4, (1, 1)),
      BlockConfig.from_args(96, 96, 3, 1, 8, (1, 1), id_skip=False),
      BlockConfig.from_args(96, 96, 3, 3, 4, (1, 1)),
      BlockConfig.from_args(96, 160, 5, 1, 8, (2, 2)),
      BlockConfig.from_args(160, 160, 5, 3, 4, (1, 1)),
      BlockConfig.from_args(160, 192, 3, 1, 8, (1, 1)),
      # pylint: enable=bad-whitespace
  )
  stem_base_filters: int = 32
  top_base_filters: int = 1280
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
  # No Squeeze/Excite for MobilenetEdgeTPU
  use_se: bool = False
  input_channels: int = 3
  num_classes: int = 1001
  model_name: str = 'mobilenet_edgetpu'
  rescale_input: bool = False
  data_format: str = 'channels_last'
  dtype: str = 'float32'
  backbone_only: bool = False


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


# TODO(longy): Reuse the utility functions for V1/V2 models.
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


def conv2d_block(inputs: tf.Tensor,
                 conv_filters: Optional[int],
                 config: ModelConfig,
                 kernel_size: Any = (1, 1),
                 strides: Any = (1, 1),
                 use_batch_norm: bool = True,
                 use_bias: bool = False,
                 activation: Any = None,
                 depthwise: bool = False,
                 name: Optional[str] = None):
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

  if depthwise:
    conv2d = tf_keras.layers.DepthwiseConv2D
    init_kwargs.update({'depthwise_initializer': CONV_KERNEL_INITIALIZER})
  else:
    conv2d = tf_keras.layers.Conv2D
    init_kwargs.update({'filters': conv_filters,
                        'kernel_initializer': CONV_KERNEL_INITIALIZER})

  x = conv2d(**init_kwargs)(inputs)

  if use_batch_norm:
    bn_axis = 1 if data_format == 'channels_first' else -1
    x = batch_norm(axis=bn_axis,
                   momentum=bn_momentum,
                   epsilon=bn_epsilon,
                   name=name + '_bn')(x)

  if activation is not None:
    x = tf_keras.layers.Activation(activation,
                                   name=name + '_activation')(x)
  return x


def mb_conv_block(inputs: tf.Tensor,
                  block: BlockConfig,
                  config: ModelConfig,
                  prefix: Optional[str] = None):
  """Mobile Inverted Residual Bottleneck.

  Args:
    inputs: the Keras input to the block
    block: BlockConfig, arguments to create a Block
    config: ModelConfig, a set of model parameters
    prefix: prefix for naming all layers

  Returns:
    the output of the block
  """
  use_se = config.use_se
  activation = tf_utils.get_activation(config.activation)
  drop_connect_rate = config.drop_connect_rate
  data_format = tf_keras.backend.image_data_format()
  use_depthwise = block.conv_type == 'depthwise'
  prefix = prefix or ''

  filters = block.input_filters * block.expand_ratio

  x = inputs

  if block.fused_conv:
    # If we use fused mbconv, skip expansion and use regular conv.
    x = conv2d_block(x,
                     filters,
                     config,
                     kernel_size=block.kernel_size,
                     strides=block.strides,
                     activation=activation,
                     name=prefix + 'fused')
  else:
    if block.expand_ratio != 1:
      # Expansion phase
      kernel_size = (1, 1) if use_depthwise else (3, 3)
      x = conv2d_block(x,
                       filters,
                       config,
                       kernel_size=kernel_size,
                       activation=activation,
                       name=prefix + 'expand')

    # Depthwise Convolution
    if use_depthwise:
      x = conv2d_block(x,
                       conv_filters=None,
                       config=config,
                       kernel_size=block.kernel_size,
                       strides=block.strides,
                       activation=activation,
                       depthwise=True,
                       name=prefix + 'depthwise')

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

    se = tf_keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
    se = tf_keras.layers.Reshape(se_shape, name=prefix + 'se_reshape')(se)

    se = conv2d_block(se,
                      num_reduced_filters,
                      config,
                      use_bias=True,
                      use_batch_norm=False,
                      activation=activation,
                      name=prefix + 'se_reduce')
    se = conv2d_block(se,
                      filters,
                      config,
                      use_bias=True,
                      use_batch_norm=False,
                      activation='sigmoid',
                      name=prefix + 'se_expand')
    x = tf_keras.layers.multiply([x, se], name=prefix + 'se_excite')

  # Output phase
  x = conv2d_block(x,
                   block.output_filters,
                   config,
                   activation=None,
                   name=prefix + 'project')

  # Add identity so that quantization-aware training can insert quantization
  # ops correctly.
  x = tf_keras.layers.Activation('linear', name=prefix + 'id')(x)

  if (block.id_skip
      and all(s == 1 for s in block.strides)
      and block.input_filters == block.output_filters):
    if drop_connect_rate and drop_connect_rate > 0:
      # Apply dropconnect
      # The only difference between dropout and dropconnect in TF is scaling by
      # drop_connect_rate during training. See:
      # https://github.com/keras-team/keras/pull/9898#issuecomment-380577612
      x = tf_keras.layers.Dropout(drop_connect_rate,
                                  noise_shape=(None, 1, 1, 1),
                                  name=prefix + 'drop')(x)

    x = tf_keras.layers.add([x, inputs], name=prefix + 'add')

  return x


def mobilenet_edgetpu(image_input: tf_keras.layers.Input, config: ModelConfig):  # pytype: disable=invalid-annotation  # typed-keras
  """Creates a MobilenetEdgeTPU graph given the model parameters.

  This function is wrapped by the `MobilenetEdgeTPU` class to make a
  tf_keras.Model.

  Args:
    image_input: the input batch of images
    config: the model config

  Returns:
    The output of clossification model or if backbone is needed, dictionary with
    backbone feature levels.
  """
  depth_coefficient = config.depth_coefficient
  blocks = config.blocks
  stem_base_filters = config.stem_base_filters
  top_base_filters = config.top_base_filters
  activation = tf_utils.get_activation(config.activation)
  dropout_rate = config.dropout_rate
  drop_connect_rate = config.drop_connect_rate
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
  x = conv2d_block(x,
                   round_filters(stem_base_filters, config),
                   config,
                   kernel_size=[3, 3],
                   strides=[2, 2],
                   activation=activation,
                   name='stem')

  # Build blocks
  num_blocks_total = sum(block.num_repeat for block in blocks)
  block_num = 0

  backbone_levels = {}
  for stack_idx, block in enumerate(blocks):
    assert block.num_repeat > 0
    # Update block input and output filters based on depth multiplier
    block = block.replace(
        input_filters=round_filters(block.input_filters, config),
        output_filters=round_filters(block.output_filters, config),
        num_repeat=round_repeats(block.num_repeat, depth_coefficient))

    # The first block needs to take care of stride and filter size increase
    drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
    config = config.replace(drop_connect_rate=drop_rate)
    block_prefix = 'stack_{}/block_0/'.format(stack_idx)
    x = mb_conv_block(x, block, config, block_prefix)
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
        x = mb_conv_block(x, block, config, prefix=block_prefix)
        block_num += 1
    backbone_levels[str(stack_idx)] = x

  if config.backbone_only:
    return backbone_levels

  # Build top
  x = conv2d_block(x,
                   round_filters(top_base_filters, config),
                   config,
                   activation=activation,
                   name='top')

  # Build classifier
  pool_size = (x.shape.as_list()[1], x.shape.as_list()[2])
  x = tf_keras.layers.AveragePooling2D(pool_size, name='top_pool')(x)
  if dropout_rate and dropout_rate > 0:
    x = tf_keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
  x = tf_keras.layers.Conv2D(
      num_classes,
      1,
      kernel_initializer=DENSE_KERNEL_INITIALIZER,
      kernel_regularizer=tf_keras.regularizers.l2(weight_decay),
      bias_regularizer=tf_keras.regularizers.l2(weight_decay),
      name='logits')(
          x)
  x = tf_keras.layers.Activation('softmax', name='probs')(x)
  x = tf.squeeze(x, axis=[1, 2])

  return x
