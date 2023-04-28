# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Layers for DeepLabV3."""

import tensorflow as tf

from official.modeling import tf_utils


class SpatialPyramidPooling(tf.keras.layers.Layer):
  """Implements the Atrous Spatial Pyramid Pooling.

  References:
    [Rethinking Atrous Convolution for Semantic Image Segmentation](
      https://arxiv.org/pdf/1706.05587.pdf)
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
  """

  def __init__(
      self,
      output_channels,
      dilation_rates,
      pool_kernel_size=None,
      use_sync_bn=False,
      batchnorm_momentum=0.99,
      batchnorm_epsilon=0.001,
      activation='relu',
      dropout=0.5,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      interpolation='bilinear',
      use_depthwise_convolution=False,
      **kwargs):
    """Initializes `SpatialPyramidPooling`.

    Args:
      output_channels: Number of channels produced by SpatialPyramidPooling.
      dilation_rates: A list of integers for parallel dilated conv.
      pool_kernel_size: A list of integers or None. If None, global average
        pooling is applied, otherwise an average pooling of pool_kernel_size
        is applied.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for type of activation to be used. Defaults to 'relu'.
      dropout: A float for the dropout rate before output. Defaults to 0.5.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      use_depthwise_convolution: Allows spatial pooling to be separable
         depthwise convolusions. [Encoder-Decoder with Atrous Separable
         Convolution for Semantic Image Segmentation](
         https://arxiv.org/pdf/1802.02611.pdf)
      **kwargs: Other keyword arguments for the layer.
    """
    super(SpatialPyramidPooling, self).__init__(**kwargs)

    self.output_channels = output_channels
    self.dilation_rates = dilation_rates
    self.use_sync_bn = use_sync_bn
    self.batchnorm_momentum = batchnorm_momentum
    self.batchnorm_epsilon = batchnorm_epsilon
    self.activation = activation
    self.dropout = dropout
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.interpolation = interpolation
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    self.pool_kernel_size = pool_kernel_size
    self.use_depthwise_convolution = use_depthwise_convolution

  def build(self, input_shape):
    channels = input_shape[3]

    self.aspp_layers = []
    bn_op = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    conv_sequential = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=self.output_channels,
            kernel_size=(1, 1),
            kernel_initializer=tf_utils.clone_initializer(
                self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False),
        bn_op(
            axis=bn_axis,
            momentum=self.batchnorm_momentum,
            epsilon=self.batchnorm_epsilon,
            synchronized=self.use_sync_bn),
        tf.keras.layers.Activation(self.activation)
    ])
    self.aspp_layers.append(conv_sequential)

    for dilation_rate in self.dilation_rates:
      leading_layers = []
      kernel_size = (3, 3)
      if self.use_depthwise_convolution:
        leading_layers += [
            tf.keras.layers.DepthwiseConv2D(
                depth_multiplier=1,
                kernel_size=kernel_size,
                padding='same',
                dilation_rate=dilation_rate,
                use_bias=False)
        ]
        kernel_size = (1, 1)
      conv_sequential = tf.keras.Sequential(leading_layers + [
          tf.keras.layers.Conv2D(
              filters=self.output_channels,
              kernel_size=kernel_size,
              padding='same',
              kernel_regularizer=self.kernel_regularizer,
              kernel_initializer=tf_utils.clone_initializer(
                  self.kernel_initializer),
              dilation_rate=dilation_rate,
              use_bias=False),
          bn_op(
              axis=bn_axis,
              momentum=self.batchnorm_momentum,
              epsilon=self.batchnorm_epsilon,
              synchronized=self.use_sync_bn),
          tf.keras.layers.Activation(self.activation)
      ])
      self.aspp_layers.append(conv_sequential)

    if self.pool_kernel_size is None:
      pool_sequential = tf.keras.Sequential([
          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Reshape((1, 1, channels))])
    else:
      pool_sequential = tf.keras.Sequential([
          tf.keras.layers.AveragePooling2D(self.pool_kernel_size)])

    pool_sequential.add(
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.output_channels,
                kernel_size=(1, 1),
                kernel_initializer=tf_utils.clone_initializer(
                    self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False),
            bn_op(
                axis=bn_axis,
                momentum=self.batchnorm_momentum,
                epsilon=self.batchnorm_epsilon,
                synchronized=self.use_sync_bn),
            tf.keras.layers.Activation(self.activation)
        ]))

    self.aspp_layers.append(pool_sequential)

    self.projection = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=self.output_channels,
            kernel_size=(1, 1),
            kernel_initializer=tf_utils.clone_initializer(
                self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False),
        bn_op(
            axis=bn_axis,
            momentum=self.batchnorm_momentum,
            epsilon=self.batchnorm_epsilon,
            synchronized=self.use_sync_bn),
        tf.keras.layers.Activation(self.activation),
        tf.keras.layers.Dropout(rate=self.dropout)
    ])

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    result = []
    for i, layer in enumerate(self.aspp_layers):
      x = layer(inputs, training=training)
      # Apply resize layer to the end of the last set of layers.
      if i == len(self.aspp_layers) - 1:
        x = tf.image.resize(tf.cast(x, tf.float32), tf.shape(inputs)[1:3])
      result.append(tf.cast(x, inputs.dtype))
    result = tf.concat(result, axis=-1)
    result = self.projection(result, training=training)
    return result

  def get_config(self):
    config = {
        'output_channels': self.output_channels,
        'dilation_rates': self.dilation_rates,
        'pool_kernel_size': self.pool_kernel_size,
        'use_sync_bn': self.use_sync_bn,
        'batchnorm_momentum': self.batchnorm_momentum,
        'batchnorm_epsilon': self.batchnorm_epsilon,
        'activation': self.activation,
        'dropout': self.dropout,
        'kernel_initializer': tf.keras.initializers.serialize(
            self.kernel_initializer),
        'kernel_regularizer': tf.keras.regularizers.serialize(
            self.kernel_regularizer),
        'interpolation': self.interpolation,
    }
    base_config = super(SpatialPyramidPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
