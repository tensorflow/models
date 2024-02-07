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

"""YOLOv7 heads."""

import tensorflow as tf, tf_keras
from official.projects.yolo.ops import initializer_ops


class YoloV7DetectionHead(tf_keras.layers.Layer):
  """YOLOv7 Detection Head."""

  def __init__(
      self,
      num_classes=80,
      min_level=3,
      max_level=5,
      num_anchors=3,
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_initializer='zeros',
      bias_regularizer=None,
      use_separable_conv=False,
      **kwargs,
  ):
    """Initializes YOLOv7 head.

    Args:
      num_classes: integer.
      min_level: minimum feature level.
      max_level: maximum feature level.
      num_anchors: integer for number of anchors at each location.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      bias_initializer: bias initializer for convolutional layers.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2d.
      use_separable_conv: `bool` wether to use separable convs.
      **kwargs: other keyword arguments.
    """
    super().__init__(**kwargs)
    self._num_classes = num_classes
    self._min_level = min_level
    self._max_level = max_level
    self._num_anchors = num_anchors

    self._kernel_initializer = initializer_ops.pytorch_kernel_initializer(
        kernel_initializer
    )
    self._kernel_regularizer = kernel_regularizer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_separable_conv = use_separable_conv

  def _bias_init(self, scale, in_channels, isize=640, no_per_conf=8):

    def bias(shape, dtype):
      init = tf_keras.initializers.VarianceScaling(
          scale=1 / 3, mode='fan_in', distribution='uniform')
      base = init([in_channels, *shape], dtype=dtype)[0]

      base = tf.reshape(base, [self._num_anchors, -1])
      box, conf, classes = tf.split(base, [4, 1, -1], axis=-1)
      conf += tf.math.log(no_per_conf / ((isize / scale)**2))
      classes += tf.math.log(0.6 / (self._num_classes - 0.99))
      base = tf.concat([box, conf, classes], axis=-1)
      base = tf.reshape(base, [-1])
      return base

    return bias

  def build(self, input_shape):
    self._convs = []
    self._implicit_adds = []
    self._implicit_muls = []
    conv_op = (
        tf_keras.layers.SeparableConv2D
        if self._use_separable_conv
        else tf_keras.layers.Conv2D
    )
    for level in range(self._min_level, self._max_level + 1):
      # Note that we assume height == width.
      h = input_shape[str(level)][2]
      scale = 2 ** int(level)
      in_channels = input_shape[str(level)][-1]
      # Outputs are num_classes + 5 (box coordinates + objectness score)
      self._convs.append(
          conv_op(
              (self._num_classes + 5) * self._num_anchors,
              kernel_size=1,
              padding='same',
              kernel_initializer=self._kernel_initializer,
              kernel_regularizer=self._kernel_regularizer,
              bias_initializer=self._bias_init(scale, in_channels, h * scale),
          )
      )
      self._implicit_adds.append(
          self.add_weight(
              name=f'implicit_adds_l{level}',
              shape=[1, 1, 1, in_channels],
              initializer=tf_keras.initializers.random_normal(
                  mean=0.0, stddev=0.02
              ),
              trainable=True,
          )
      )
      self._implicit_muls.append(
          self.add_weight(
              name=f'implicit_muls_l{level}',
              shape=[1, 1, 1, (self._num_classes + 5) * self._num_anchors],
              initializer=tf_keras.initializers.random_normal(
                  mean=1.0, stddev=0.02
              ),
              trainable=True,
          )
      )
    super().build(input_shape)

  def call(self, inputs, training=False):
    outputs = {}
    for i, level in enumerate(range(self._min_level, self._max_level + 1)):
      x = inputs[str(level)]
      x = self._implicit_adds[i] + x
      x = self._convs[i](x)
      x = self._implicit_muls[i] * x
      _, h, w, _ = x.get_shape().as_list()
      x = tf.reshape(x, [-1, h, w, self._num_anchors, self._num_classes + 5])
      outputs[str(level)] = x
    return outputs

  def get_config(self):
    config = dict(
        num_classes=self._num_classes,
        min_level=self._min_level,
        max_level=self._max_level,
        num_anchors=self._num_anchors,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_initializer=self._bias_initializer,
        bias_regularizer=self._bias_regularizer,
        use_separable_conv=self._use_separable_conv,
    )
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
