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
"""MobileNet v1.

Adapted from tf.keras.applications.mobilenet.MobileNet().

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1704.04861.

  MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications
  Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    Tobias Weyand, Marco Andreetto, Hartwig Adam

"""

import logging

import tensorflow as tf

from research.mobilenet import common_modules
from research.mobilenet.configs import archs

layers = tf.keras.layers

MobileNetV1Config = archs.MobileNetV1Config


def mobilenet_v1(config: MobileNetV1Config = MobileNetV1Config()
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet Model."""

  model_name = config.name
  input_shape = config.input_shape

  img_input = layers.Input(shape=input_shape, name='Input')

  # build network base
  x = common_modules.mobilenet_base(img_input, config)

  # build classification head
  x = common_modules.mobilenet_head(x, config)

  return tf.keras.models.Model(inputs=img_input,
                               outputs=x,
                               name=model_name)


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)
  model = mobilenet_v1()
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy])
  logging.info(model.summary())
