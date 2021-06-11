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
"""Residual Refinement Module of BASNet.

Boundary-Awar network (BASNet) were proposed in:
[1] Qin, Xuebin, et al. 
    Basnet: Boundary-aware salient object detection.
"""


# Import libraries
import tensorflow as tf
from official.vision.beta.projects.basnet.modeling.layers import nn_blocks


@tf.keras.utils.register_keras_serializable(package='Vision')
class RefUnet(tf.keras.Model):

  def __init__(self,
               input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 1]),
               activation='relu',
               use_sync_bn=False,
               use_bias=True,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Residual Refinement Module of BASNet.

    Args:
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      use_bias: if True, use bias in conv2d.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      **kwargs: keyword arguments to be passed.
    """
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._use_bias = use_bias
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build ResNet.
    inputs = tf.keras.Input(shape=self._input_specs.shape[1:])

    endpoints = {}  
    residual = inputs

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=1,
        use_bias=self._use_bias, padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)


    # Top-down
    for i in range(4):
      x = nn_blocks.ConvBlock(
          filters=64,
          kernel_size=3,
          strides=1,
          dilation_rate=1,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation='relu',
          use_sync_bn=self._use_sync_bn,
          use_bias=self._use_bias,
          norm_momentum=0.99,
          norm_epsilon=0.001
          )(x)
        
      endpoints[str(i)] = x

      x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)

    # Bridge
    x = nn_blocks.ConvBlock(
        filters=64,
        kernel_size=3,
        strides=1,
        dilation_rate=1,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation='relu',
        use_sync_bn=self._use_sync_bn,
        use_bias=self._use_bias,
        norm_momentum=0.99,
        norm_epsilon=0.001
        )(x)

    x = tf.keras.layers.UpSampling2D(
        size=2,
        interpolation='bilinear'
        )(x)

    # Bottom-up

    for i in range(4):
      x = tf.keras.layers.Concatenate(axis=-1)([endpoints[str(3-i)], x])
      x = nn_blocks.ConvBlock(
          filters=64,
          kernel_size=3,
          strides=1,
          dilation_rate=1,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation='relu',
          use_sync_bn=self._use_sync_bn,
          use_bias=self._use_bias,
          norm_momentum=0.99,
          norm_epsilon=0.001
          )(x)

      if i == 3:
        x = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3, strides=1,
            use_bias=self._use_bias, padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer
            )(x)
      else:
        x = tf.keras.layers.UpSampling2D(
            size=2,
            interpolation='bilinear'
            )(x)

    residual = tf.cast(residual, dtype=x.dtype)

    output = x + residual

    output = tf.keras.layers.Activation(
        activation='sigmoid'
        )(output)

    self._output_specs = output.get_shape()

    super(RefUnet, self).__init__(inputs=inputs, outputs=output, **kwargs)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    return self._output_specs
