# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Layers for DeepLabV3."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='keras_cv')
class ASPP(tf.keras.layers.Layer):
  """Implements the Atrous Spatial Pyramid Pooling.

  Reference:
    [Rethinking Atrous Convolution for Semantic Image Segmentation](
      https://arxiv.org/pdf/1706.05587.pdf)
  """

  def __init__(
      self,
      output_channels,
      dilation_rates,
      batchnorm_momentum=0.99,
      dropout=0.5,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      interpolation='bilinear',
      **kwargs):
    """Initializes `ASPP`.

    Arguments:
      output_channels: Number of channels produced by ASPP.
      dilation_rates: A list of integers for parallel dilated conv.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      dropout: A float for the dropout rate before output. Defaults to 0.5.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      **kwargs: Other keyword arguments for the layer.
    """
    super(ASPP, self).__init__(**kwargs)

    self.output_channels = output_channels
    self.dilation_rates = dilation_rates
    self.batchnorm_momentum = batchnorm_momentum
    self.dropout = dropout
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.interpolation = interpolation
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)

  def build(self, input_shape):
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    self.aspp_layers = []

    conv_sequential = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=self.output_channels, kernel_size=(1, 1),
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=self.batchnorm_momentum),
        tf.keras.layers.Activation('relu')])
    self.aspp_layers.append(conv_sequential)

    for dilation_rate in self.dilation_rates:
      conv_sequential = tf.keras.Sequential([
          tf.keras.layers.Conv2D(
              filters=self.output_channels, kernel_size=(3, 3),
              padding='same', kernel_regularizer=self.kernel_regularizer,
              kernel_initializer=self.kernel_initializer,
              dilation_rate=dilation_rate, use_bias=False),
          tf.keras.layers.BatchNormalization(momentum=self.batchnorm_momentum),
          tf.keras.layers.Activation('relu')])
      self.aspp_layers.append(conv_sequential)

    pool_sequential = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, channels)),
        tf.keras.layers.Conv2D(
            filters=self.output_channels, kernel_size=(1, 1),
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=self.batchnorm_momentum),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.experimental.preprocessing.Resizing(
            height, width, interpolation=self.interpolation)])
    self.aspp_layers.append(pool_sequential)

    self.projection = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=self.output_channels, kernel_size=(1, 1),
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=self.batchnorm_momentum),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(rate=self.dropout)])

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    result = []
    for layer in self.aspp_layers:
      result.append(layer(inputs, training=training))
    result = tf.concat(result, axis=-1)
    result = self.projection(result, training=training)
    return result

  def get_config(self):
    config = {
        'output_channels': self.output_channels,
        'dilation_rates': self.dilation_rates,
        'batchnorm_momentum': self.batchnorm_momentum,
        'dropout': self.dropout,
        'kernel_initializer': tf.keras.initializers.serialize(
            self.kernel_initializer),
        'kernel_regularizer': tf.keras.regularizers.serialize(
            self.kernel_regularizer),
        'interpolation': self.interpolation,
    }
    base_config = super(ASPP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
