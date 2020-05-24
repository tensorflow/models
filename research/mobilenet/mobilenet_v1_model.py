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
from typing import Tuple

import tensorflow as tf

from research.mobilenet import common_modules
from research.mobilenet.configs import archs

layers = tf.keras.layers

MobileNetV1Config = archs.MobileNetV1Config


def _reduced_kernel_size_for_small_input(input_tensor: tf.Tensor,
                                         kernel_size: Tuple[int, int]
                                         ) -> Tuple[int, int]:
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = (min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1]))
  return kernel_size_out


def mobilenet_v1(input_shape: Tuple[int, int, int] = (224, 224, 3),
                 config: MobileNetV1Config = MobileNetV1Config()
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet Model."""

  global_pool = config.global_pool
  model_name = config.name

  img_input = layers.Input(shape=input_shape, name='Input')

  # build network base
  x = common_modules.mobilenet_base(img_input, config)

  # build top
  if global_pool:
    # global average pooling.
    x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                      name='top_GlobalPool')(x)
    x = layers.Reshape((1, 1, x.shape[1]))(x)
  else:
    # pooling with a fixed kernel size
    kernel_size = _reduced_kernel_size_for_small_input(x, (7, 7))
    x = layers.AvgPool2D(pool_size=kernel_size,
                         padding='VALID',
                         data_format='channels_last',
                         name='top_AvgPool')(x)

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
