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

"""RefUNet model."""
import tensorflow as tf
from official.projects.basnet.modeling import nn_blocks


@tf.keras.utils.register_keras_serializable(package='Vision')
class RefUnet(tf.keras.layers.Layer):
  """Residual Refinement Module of BASNet.

  Boundary-Aware network (BASNet) were proposed in:
  [1] Qin, Xuebin, et al.
      Basnet: Boundary-aware salient object detection.
  """

  def __init__(self,
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
    super(RefUnet, self).__init__(**kwargs)
    self._config_dict = {
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'use_bias': use_bias,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    self._concat = tf.keras.layers.Concatenate(axis=-1)
    self._sigmoid = tf.keras.layers.Activation(activation='sigmoid')
    self._maxpool = tf.keras.layers.MaxPool2D(
        pool_size=2,
        strides=2,
        padding='valid')
    self._upsample = tf.keras.layers.UpSampling2D(
        size=2,
        interpolation='bilinear')

  def build(self, input_shape):
    """Creates the variables of the BASNet decoder."""
    conv_op = tf.keras.layers.Conv2D
    conv_kwargs = {
        'kernel_size': 3,
        'strides': 1,
        'use_bias': self._config_dict['use_bias'],
        'kernel_initializer': self._config_dict['kernel_initializer'],
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
        'bias_regularizer': self._config_dict['bias_regularizer'],
    }

    self._in_conv = conv_op(
        filters=64,
        padding='same',
        **conv_kwargs)

    self._en_convs = []
    for _ in range(4):
      self._en_convs.append(nn_blocks.ConvBlock(
          filters=64,
          use_sync_bn=self._config_dict['use_sync_bn'],
          norm_momentum=self._config_dict['norm_momentum'],
          norm_epsilon=self._config_dict['norm_epsilon'],
          **conv_kwargs))

    self._bridge_convs = []
    for _ in range(1):
      self._bridge_convs.append(nn_blocks.ConvBlock(
          filters=64,
          use_sync_bn=self._config_dict['use_sync_bn'],
          norm_momentum=self._config_dict['norm_momentum'],
          norm_epsilon=self._config_dict['norm_epsilon'],
          **conv_kwargs))

    self._de_convs = []
    for _ in range(4):
      self._de_convs.append(nn_blocks.ConvBlock(
          filters=64,
          use_sync_bn=self._config_dict['use_sync_bn'],
          norm_momentum=self._config_dict['norm_momentum'],
          norm_epsilon=self._config_dict['norm_epsilon'],
          **conv_kwargs))

    self._out_conv = conv_op(
        filters=1,
        padding='same',
        **conv_kwargs)

  def call(self, inputs):
    endpoints = {}
    residual = inputs
    x = self._in_conv(inputs)

    # Top-down
    for i, block in enumerate(self._en_convs):
      x = block(x)
      endpoints[str(i)] = x
      x = self._maxpool(x)

    # Bridge
    for i, block in enumerate(self._bridge_convs):
      x = block(x)

    # Bottom-up
    for i, block in enumerate(self._de_convs):
      dtype = x.dtype
      x = tf.cast(x, tf.float32)
      x = self._upsample(x)
      x = tf.cast(x, dtype)
      x = self._concat([endpoints[str(3-i)], x])
      x = block(x)

    x = self._out_conv(x)
    residual = tf.cast(residual, dtype=x.dtype)
    output = self._sigmoid(x + residual)

    self._output_specs = output.get_shape()
    return output

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    return self._output_specs
