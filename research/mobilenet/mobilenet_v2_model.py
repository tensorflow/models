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
"""MobileNet v2.

Adapted from tf.keras.applications.mobilenet_v2.MobileNetV2().

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""

import logging
from typing import Tuple, Union

import tensorflow as tf

from research.mobilenet import common_modules

layers = tf.keras.layers


def _inverted_res_block(inputs: tf.Tensor,
                        filters: int,
                        width_multiplier: float,
                        min_depth: int,
                        weight_decay: float,
                        stddev: float,
                        batch_norm_decay: float,
                        batch_norm_epsilon: float,
                        dilation_rate: int = 1,
                        expansion_size: int = 6,
                        regularize_depthwise: bool = False,
                        use_explicit_padding: bool = False,
                        residual=True,
                        kernel: Union[int, Tuple[int, int]] = (3, 3),
                        strides: Union[int, Tuple[int, int]] = 1,
                        block_id: int = 1
                        ) -> tf.Tensor:
  """Depthwise Convolution Block with expansion.

  Builds a composite convolution that has the following structure
  expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels]
    filters: the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    width_multiplier: controls the width of the network.
      - If `width_multiplier` < 1.0, proportionally decreases the number
            of filters in each layer.
      - If `width_multiplier` > 1.0, proportionally increases the number
            of filters in each layer.
      - If `width_multiplier` = 1, default number of filters from the paper
            are used at each layer.
      This is called `width multiplier (\alpha)` in the original paper.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when width_multiplier < 1, and not an active constraint when
      width_multiplier >= 1.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    expansion_size: the size of expansion, could be a constant or a callable.
      If latter it will be provided 'num_inputs' as an input. For forward
      compatibility it should accept arbitrary keyword arguments.
      Default will expand the input by factor of 6.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    residual: whether to include residual connection between input
      and output.
    kernel: An integer or tuple/list of 2 integers, specifying the
      width and height of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution
        along the width and height.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    block_id: a unique identification designating the block number.

  Returns:
    Tensor of depth num_outputs
  """

  prefix = 'block_{}_'.format(block_id)
  filters = common_modules.width_multiplier_op_divisible(
    filters=filters,
    width_multiplier=width_multiplier,
    min_depth=min_depth)

  weights_init = tf.keras.initializers.TruncatedNormal(stddev=stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  depth_regularizer = regularizer if regularize_depthwise else None

  # Expand
  in_channels = inputs.shape.as_list()[-1]
  expended_size = common_modules.expand_input_by_factor(
    num_inputs=in_channels,
    expansion_size=expansion_size)
  x = layers.Conv2D(filters=expended_size,
                    kernel_size=kernel,
                    strides=strides,
                    padding='SAME',
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name=prefix + 'expand')(inputs)

  x = layers.BatchNormalization(epsilon=batch_norm_epsilon,
                                momentum=batch_norm_decay,
                                axis=-1,
                                name=prefix + 'expand_BN')(x)

  x = layers.ReLU(max_value=6.,
                  name=prefix + 'expand_ReLU')(x)

  # Depthwise
  padding = 'SAME'
  if use_explicit_padding:
    padding = 'VALID'
    x = common_modules.FixedPadding(
      kernel_size=kernel,
      name=prefix + 'pad')(x)

  x = layers.DepthwiseConv2D(kernel_size=kernel,
                             padding=padding,
                             depth_multiplier=1,
                             strides=strides,
                             kernel_initializer=weights_init,
                             kernel_regularizer=depth_regularizer,
                             dilation_rate=dilation_rate,
                             use_bias=False,
                             name=prefix + 'depthwise')(x)
  x = layers.BatchNormalization(epsilon=batch_norm_epsilon,
                                momentum=batch_norm_decay,
                                axis=-1,
                                name=prefix + 'depthwise_BN')(x)
  x = layers.ReLU(max_value=6.,
                  name=prefix + 'depthwise_ReLU')(x)

  # Project
  x = layers.Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    padding='SAME',
                    strides=(1, 1),
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name=prefix + 'project')(x)
  x = layers.BatchNormalization(epsilon=batch_norm_epsilon,
                                momentum=batch_norm_decay,
                                axis=-1,
                                name=prefix + 'project_BN')(x)

  if (residual and
      # stride check enforces that we don't add residuals when spatial
      # dimensions are None
      strides == 1 and
      # Depth matches
      in_channels == filters):
    x = layers.Add(name=prefix + 'add')([inputs, x])
  return x
