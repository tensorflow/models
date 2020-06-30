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
from typing import Tuple

import tensorflow as tf

from research.mobilenet import common_modules
from research.mobilenet.configs import archs

layers = tf.keras.layers

MobileNetV2Config = archs.MobileNetV2Config


def mobilenet_v2(input_shape: Tuple[int, int, int] = (224, 224, 3),
                 config: MobileNetV2Config = MobileNetV2Config()
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet Model."""

  model_name = config.name

  img_input = layers.Input(shape=input_shape, name='Input')

  # build network base
  x = common_modules.mobilenet_base(img_input, config)

  # build top
  # global average pooling.
  x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                    name='top/GlobalPool')(x)
  x = layers.Reshape((1, 1, x.shape[1]), name='top/Reshape')(x)

  # build classification head
  x = common_modules.mobilenet_head(x, config)

  return tf.keras.models.Model(inputs=img_input,
                               outputs=x,
                               name=model_name)


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)
  model = mobilenet_v2()
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy])
  logging.info(model.summary())
