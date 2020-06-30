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
from typing import Tuple, Union

import tensorflow as tf

from research.mobilenet import common_modules
from research.mobilenet.configs import archs

layers = tf.keras.layers

MobileNetV3Config = Union[archs.MobileNetV3SmallConfig,
                          archs.MobileNetV3LargeConfig]


def mobilenet_v3(config: MobileNetV3Config,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet V3 Model."""

  width_multiplier = config.width_multiplier
  min_depth = config.min_depth
  finegrain_classification_mode = config.finegrain_classification_mode
  model_name = config.name
  activation_name = config.activation_name

  img_input = layers.Input(shape=input_shape, name='Input')

  # build network base
  x = common_modules.mobilenet_base(img_input, config)

  # build top
  # global average pooling.
  x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                    name='top/GlobalPool')(x)
  x = layers.Reshape((1, 1, x.shape[1]), name='top/Reshape')(x)

  # last layer of conv
  if isinstance(config, archs.MobileNetV3SmallConfig):
    last_conv_channels = 1024
  elif isinstance(config, archs.MobileNetV3LargeConfig):
    last_conv_channels = 1280
  else:
    raise ValueError('Only support MobileNetV3S and MobileNetV3L')

  if (not finegrain_classification_mode
      or (finegrain_classification_mode and width_multiplier > 1.0)):
    last_conv_channels = common_modules.width_multiplier_op_divisible(
      filters=last_conv_channels,
      width_multiplier=width_multiplier,
      min_depth=min_depth)

  x = layers.Conv2D(filters=last_conv_channels,
                    kernel_size=(1, 1),
                    padding='SAME',
                    name='top/Conv2d_1x1')(x)
  x = layers.Activation(
    activation=archs.get_activation_function()[activation_name],
    name='top/Conv2d_1x1_{}'.format(activation_name))(x)

  # build classification head
  x = common_modules.mobilenet_head(x, config)

  return tf.keras.models.Model(inputs=img_input,
                               outputs=x,
                               name=model_name)


def mobilenet_v3_small(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    config: archs.MobileNetV3SmallConfig = archs.MobileNetV3SmallConfig()
) -> tf.keras.models.Model:
  """Instantiates the MobileNet V3 Small Model."""
  assert isinstance(config, archs.MobileNetV3SmallConfig)
  return mobilenet_v3(input_shape=input_shape, config=config)


def mobilenet_v3_large(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    config: archs.MobileNetV3LargeConfig = archs.MobileNetV3LargeConfig(),
) -> tf.keras.models.Model:
  """Instantiates the MobileNet V3 Large Model."""
  assert isinstance(config, archs.MobileNetV3LargeConfig)
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
