# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Yolo heads."""

import tensorflow as tf
from official.projects.yolo.modeling.layers import nn_blocks


class YoloHead(tf.keras.layers.Layer):
  """YOLO Prediction Head."""

  def __init__(self,
               min_level,
               max_level,
               classes=80,
               boxes_per_level=3,
               output_extras=0,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation=None,
               smart_bias=False,
               use_separable_conv=False,
               **kwargs):
    """Yolo Prediction Head initialization function.

    Args:
      min_level: `int`, the minimum backbone output level.
      max_level: `int`, the maximum backbone output level.
      classes: `int`, number of classes per category.
      boxes_per_level: `int`, number of boxes to predict per level.
      output_extras: `int`, number of additional output channels that the head.
        should predict for non-object detection and non-image classification
        tasks.
      norm_momentum: `float`, normalization momentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      activation: `str`, the activation function to use typically leaky or mish.
      smart_bias: `bool`, whether to use smart bias.
      use_separable_conv: `bool` wether to use separable convs.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level

    self._key_list = [
        str(key) for key in range(self._min_level, self._max_level + 1)
    ]

    self._classes = classes
    self._boxes_per_level = boxes_per_level
    self._output_extras = output_extras

    self._output_conv = (classes + output_extras + 5) * boxes_per_level
    self._smart_bias = smart_bias
    self._use_separable_conv = use_separable_conv

    self._base_config = dict(
        activation=activation,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

    self._conv_config = dict(
        filters=self._output_conv,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        use_bn=False,
        use_separable_conv=self._use_separable_conv,
        **self._base_config)

  def bias_init(self, scale, inshape, isize=640, no_per_conf=8):

    def bias(shape, dtype):
      init = tf.keras.initializers.Zeros()
      base = init(shape, dtype=dtype)
      if self._smart_bias:
        base = tf.reshape(base, [self._boxes_per_level, -1])
        box, conf, classes = tf.split(base, [4, 1, -1], axis=-1)
        conf += tf.math.log(no_per_conf / ((isize / scale)**2))
        classes += tf.math.log(0.6 / (self._classes - 0.99))
        base = tf.concat([box, conf, classes], axis=-1)
        base = tf.reshape(base, [-1])
      return base

    return bias

  def build(self, input_shape):
    self._head = dict()
    for key in self._key_list:
      scale = 2**int(key)
      self._head[key] = nn_blocks.ConvBN(
          bias_initializer=self.bias_init(scale, input_shape[key][-1]),
          **self._conv_config)

  def call(self, inputs):
    outputs = dict()
    for key in self._key_list:
      outputs[key] = self._head[key](inputs[key])
    return outputs

  @property
  def output_depth(self):
    return (self._classes + self._output_extras + 5) * self._boxes_per_level

  @property
  def num_boxes(self):
    if self._min_level is None or self._max_level is None:
      raise Exception(
          'Model has to be built before number of boxes can be determined.')
    return (self._max_level - self._min_level + 1) * self._boxes_per_level

  @property
  def num_heads(self):
    return self._max_level - self._min_level + 1

  def get_config(self):
    config = dict(
        min_level=self._min_level,
        max_level=self._max_level,
        classes=self._classes,
        boxes_per_level=self._boxes_per_level,
        output_extras=self._output_extras,
        **self._base_config)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
