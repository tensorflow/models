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
"""Decoder of BASNet.

Boundary-Awar network (BASNet) were proposed in:
[1] Qin, Xuebin, et al. 
    Basnet: Boundary-aware salient object detection.
"""

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils
#from official.vision.beta.projects.basnet.modeling.layers import nn_layers
from official.vision.beta.projects.basnet.modeling.layers import nn_blocks

layers = tf.keras.layers

# nf : num_filters, dr : dilation_rate
# (conv1_nf, conv1_dr, convm_nf, convm_dr, conv2_nf, conv2_dr, scale_factor)
BASNET_DE_SPECS = [
            (512, 2, 512, 2, 512, 2, 32),    #Bridge(Sup0)
            (512, 1, 512, 2, 512, 2, 32), #Sup1, stage6d
            (512, 1, 512, 1, 512, 1, 16), #Sup2, stage5d
            (512, 1, 512, 1, 256, 1, 8),  #Sup3, stage4d
            (256, 1, 256, 1, 128, 1, 4),  #Sup4, stage3d
            (128, 1, 128, 1, 64,  1, 2),  #Sup5, stage2d
            (64,  1, 64,  1, 64,  1, 1)   #Sup6, stage1d
        ]

@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNet_De(tf.keras.Model):
  """BASNet Decoder."""

  def __init__(self,
               input_specs,
               use_separable_conv=False,
               activation='relu',
               use_sync_bn=False,
               use_bias=True,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """BASNet Decoder initialization function.

    Args:
      input_specs: `dict` input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      use_separable_conv: `bool`, if True use separable convolution for
        convolution in BASNet layers.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'use_bias': use_bias,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if use_separable_conv:
      conv2d = tf.keras.layers.SeparableConv2D
    else:
      conv2d = tf.keras.layers.Conv2D
    if use_sync_bn:
      norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      norm = tf.keras.layers.BatchNormalization
    activation_fn = tf.keras.layers.Activation(
        tf_utils.get_activation(activation))

    # Build input feature pyramid.
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Get input feature pyramid from backbone.
    inputs = self._build_input_pyramid(input_specs)


    sup = {}

    for i, spec in enumerate(BASNET_DE_SPECS):
      if i == 0:
        x = inputs['5'] # Bridge input
      else:
        x = layers.Concatenate(axis=-1)([x, inputs[str(6-i)]])

      for j in range(3):
        x = nn_blocks.ConvBlock(
            filters=spec[2*j],
            kernel_size=3,
            strides=1,
            dilation_rate=spec[2*j+1],
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activation='relu',
            use_sync_bn=use_sync_bn,
            use_bias=use_bias,
            norm_momentum=0.99,
            norm_epsilon=0.001
            )(x)

      output = layers.Conv2D(
          filters=1, kernel_size=3, strides=1, use_bias=use_bias, padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer
          )(x)
      output = layers.UpSampling2D(
          size=spec[6],
          interpolation='bilinear'
          )(output)
      output = tf.keras.layers.Activation(
          activation='sigmoid'
          )(output)
      sup[str(i+1)] = output
      if i != 0:
        x = layers.UpSampling2D(
            size=2,
            interpolation='bilinear'
            )(x)

    self._output_specs = {
        str(order): sup[str(order)].get_shape()
        for order in range(1, 7)
    }

    super(BASNet_De, self).__init__(inputs=inputs, outputs=sup, **kwargs)

  def _build_input_pyramid(self, input_specs):
    assert isinstance(input_specs, dict)

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf.keras.Input(shape=spec[1:])
    return inputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {order: TensorShape} pairs for the model output."""
    return self._output_specs
