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

  dropout_keep_prob = config.dropout_keep_prob
  num_classes = config.num_classes
  spatial_squeeze = config.spatial_squeeze
  model_name = config.name

  img_input = layers.Input(shape=input_shape, name='Input')
  x = common_modules.mobilenet_base(img_input, config)

  # Build top
  # Global average pooling.
  x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                    name='top_GlobalPool')(x)
  x = layers.Reshape((1, 1, x.shape[1]))(x)

  # 1 x 1 x num_classes
  x = layers.Dropout(rate=1 - dropout_keep_prob,
                     name='top_Dropout')(x)

  x = layers.Conv2D(filters=num_classes,
                    kernel_size=(1, 1),
                    padding='SAME',
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name='top_Conv2d_1x1')(x)
  if spatial_squeeze:
    x = layers.Reshape(target_shape=(num_classes,),
                       name='top_SpatialSqueeze')(x)

  x = layers.Activation(activation='softmax',
                        name='top_Predictions')(x)

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
