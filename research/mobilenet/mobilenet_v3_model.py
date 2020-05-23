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
"""MobileNet v3.

Adapted from tf.keras.applications.mobilenet_v3.MobileNetV3().

Architecture: https://arxiv.org/abs/1905.02244
"""

import logging
from typing import Tuple, Union, Text

import tensorflow as tf

from research.mobilenet import common_modules
from research.mobilenet.configs import archs

layers = tf.keras.layers

MobileNetV3SmallConfig = archs.MobileNetV3SmallConfig
MobileNetV3LargeConfig = archs.MobileNetV3LargeConfig

MobileNetV3Config = Union[MobileNetV3SmallConfig, MobileNetV3LargeConfig]


def mobilenet_v3_base(inputs: tf.Tensor,
                      config: MobileNetV3Config
                      ) -> tf.Tensor:
  """Build the base MobileNet architecture."""

  min_depth = config.min_depth
  width_multiplier = config.width_multiplier
  finegrain_classification_mode = config.finegrain_classification_mode
  weight_decay = config.weight_decay
  stddev = config.stddev
  regularize_depthwise = config.regularize_depthwise
  batch_norm_decay = config.batch_norm_decay
  batch_norm_epsilon = config.batch_norm_epsilon
  output_stride = config.output_stride
  use_explicit_padding = config.use_explicit_padding
  normalization_name = config.normalization_name
  normalization_params = {
    'momentum': batch_norm_decay,
    'epsilon': batch_norm_epsilon
  }
  blocks = config.blocks

  if width_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  if finegrain_classification_mode and width_multiplier < 1.0:
    blocks[-1].filters /= width_multiplier

  # The current_stride variable keeps track of the output stride of the
  # activations, i.e., the running product of convolution strides up to the
  # current network layer. This allows us to invoke atrous convolution
  # whenever applying the next convolution would result in the activations
  # having output stride larger than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  net = inputs
  for i, block_def in enumerate(blocks):
    if output_stride is not None and current_stride == output_stride:
      # If we have reached the target output_stride, then we need to employ
      # atrous convolution with stride=1 and multiply the atrous rate by the
      # current unit's stride for use in subsequent layers.
      layer_stride = 1
      layer_rate = rate
      rate *= block_def.stride
    else:
      layer_stride = block_def.stride
      layer_rate = 1
      current_stride *= block_def.stride

    if block_def.block_type == archs.BlockType.Conv.value:
      if i == 0 or width_multiplier > 1.0:
        filters = common_modules.width_multiplier_op_divisible(
          filters=block_def.filters,
          width_multiplier=width_multiplier,
          min_depth=min_depth)
      else:
        filters = block_def.filters
      net = common_modules.conv2d_block(
        inputs=net,
        filters=filters,
        kernel=block_def.kernel,
        strides=block_def.stride,
        activation_name=block_def.activation_name,
        width_multiplier=1,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        use_explicit_padding=use_explicit_padding,
        normalization_name=normalization_name,
        normalization_params=normalization_params,
        block_id=i
      )
    elif block_def.block_type == archs.BlockType.InvertedResConv.value:
      use_rate = rate
      if layer_rate > 1 and block_def.kernel != (1, 1):
        # We will apply atrous rate in the following cases:
        # 1) When kernel_size is not in params, the operation then uses
        #   default kernel size 3x3.
        # 2) When kernel_size is in params, and if the kernel_size is not
        #   equal to (1, 1) (there is no need to apply atrous convolution to
        #   any 1x1 convolution).
        use_rate = layer_rate
      net = common_modules.inverted_res_block(
        inputs=net,
        filters=block_def.filters,
        kernel=block_def.kernel,
        strides=layer_stride,
        expansion_size=block_def.expansion_size,
        squeeze_factor=block_def.squeeze_factor,
        activation_name=block_def.activation_name,
        dilation_rate=use_rate,
        width_multiplier=width_multiplier,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        regularize_depthwise=regularize_depthwise,
        use_explicit_padding=use_explicit_padding,
        normalization_name=normalization_name,
        normalization_params=normalization_params,
        block_id=i
      )
    else:
      raise ValueError('Unknown block type {} for layer {}'.format(
        block_def.block_type, i))
  return net


def mobilenet_v3(config: MobileNetV3Config,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet V3 Small Model."""

  dropout_keep_prob = config.dropout_keep_prob
  num_classes = config.num_classes
  spatial_squeeze = config.spatial_squeeze
  model_name = config.name
  activation_name = config.activation_name

  img_input = layers.Input(shape=input_shape, name='Input')
  x = mobilenet_v3_base(img_input, config)

  # Build top
  # Global average pooling.
  x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                    name='top_GlobalPool')(x)
  x = layers.Reshape((1, 1, x.shape[1]))(x)

  x = layers.Conv2D(filters=1280,
                    kernel_size=(1, 1),
                    padding='SAME',
                    name='top_Conv2d_1x1')(x)
  x = layers.Activation(
    activation=archs.get_activation_function()[activation_name],
    name='top_Conv2d_1x1_{}'.format(activation_name))(x)

  x = layers.Dropout(rate=1 - dropout_keep_prob,
                     name='top_Dropout')(x)

  # 1 x 1 x 1024
  x = layers.Conv2D(filters=num_classes,
                    kernel_size=(1, 1),
                    padding='SAME',
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name='top_Conv2d_1x1_output')(x)
  if spatial_squeeze:
    x = layers.Reshape(target_shape=(num_classes,),
                       name='top_SpatialSqueeze')(x)

  x = layers.Activation(activation='softmax',
                        name='top_Predictions')(x)

  return tf.keras.models.Model(inputs=img_input,
                               outputs=x,
                               name=model_name)


def mobilenet_v3_small(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    config: MobileNetV3SmallConfig = MobileNetV3SmallConfig()
) -> tf.keras.models.Model:
  assert isinstance(config, MobileNetV3SmallConfig)
  return mobilenet_v3(input_shape=input_shape, config=config)


def mobilenet_v3_large(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    config: MobileNetV3LargeConfig = MobileNetV3LargeConfig(),
) -> tf.keras.models.Model:
  assert isinstance(config, MobileNetV3LargeConfig)
  return mobilenet_v3(input_shape=input_shape, config=config)


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)
  model_small = mobilenet_v3_small()
  model_small.compile(
    optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy])
  logging.info(model_small.summary())

  model_large = mobilenet_v3_large()
  model_large.compile(
    optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy])
  logging.info(model_large.summary())
