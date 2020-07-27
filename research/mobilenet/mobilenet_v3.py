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
                          archs.MobileNetV3LargeConfig,
                          archs.MobileNetV3EdgeTPUConfig]


def mobilenet_v3(config: MobileNetV3Config,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet V3 Model."""

  model_name = config.name

  img_input = layers.Input(shape=input_shape, name='Input')

  # build network base
  x = common_modules.mobilenet_base(img_input, config)

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


def mobilenet_v3_edge_tpu(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    config: archs.MobileNetV3EdgeTPUConfig = archs.MobileNetV3EdgeTPUConfig(),
) -> tf.keras.models.Model:
  """Instantiates the MobileNet V3 Large Model."""
  assert isinstance(config, archs.MobileNetV3EdgeTPUConfig)
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

  model_edge_tpu = mobilenet_v3_edge_tpu()
  model_edge_tpu.compile(
    optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy])
  logging.info(model_edge_tpu.summary())
