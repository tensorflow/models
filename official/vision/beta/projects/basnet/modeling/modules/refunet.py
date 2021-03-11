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
from official.modeling import tf_utils
from official.vision.beta.projects.basnet.modeling.layers import nn_blocks
from official.vision.beta.projects.basnet.modeling.layers import nn_layers

layers = tf.keras.layers

@tf.keras.utils.register_keras_serializable(package='Vision')
class RefUnet(tf.keras.Model):

  def __init__(self,
               input_specs=layers.InputSpec(shape=[None, None, None, 1]),
               activation='relu',
               use_sync_bn=False,
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
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
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

    x = layers.Conv2D(
        filters=64, kernel_size=3, strides=1, use_bias=True, padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)


    # Top-down
    for i in range(4):
      x = nn_layers.ConvBNReLU(
          filters=64,
          kernel_size=3,
          strides=1,
          dilation=1,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation='relu',
          use_sync_bn=self._use_sync_bn,
          norm_momentum=0.99,
          norm_epsilon=0.001
          )(x)
        
      endpoints[str(i)] = x

      #x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
      x = layers.MaxPool2D(pool_size=2, strides=2, padding='valid')(x)

    # Bridge
    x = nn_layers.ConvBNReLU(
        filters=64,
        kernel_size=3,
        strides=1,
        dilation=1,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation='relu',
        use_sync_bn=self._use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001
        )(x)

    x = layers.UpSampling2D(
        size=2,
        interpolation='bilinear'
        )(x)

    # Bottom-up

    for i in range(4):
      x = layers.Concatenate(axis=-1)([endpoints[str(3-i)], x])
      x = nn_layers.ConvBNReLU(
          filters=64,
          kernel_size=3,
          strides=1,
          dilation=1,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation='relu',
          use_sync_bn=self._use_sync_bn,
          norm_momentum=0.99,
          norm_epsilon=0.001
          )(x)

      if i == 3:
        x = layers.Conv2D(
            filters=1, kernel_size=3, strides=1, use_bias=True, padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer
            )(x)
      else:
        x = layers.UpSampling2D(
            size=2,
            interpolation='bilinear'
            )(x)

    output = x + residual

    output = layers.Activation(
        activation='sigmoid'
        )(output)

    #self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    self._output_specs = output.get_shape()

    super(RefUnet, self).__init__(inputs=inputs, outputs=output, **kwargs)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    return self._output_specs
