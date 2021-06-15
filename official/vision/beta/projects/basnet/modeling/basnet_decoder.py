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

# Import libraries
from typing import Mapping
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.projects.basnet.modeling.layers import nn_blocks

# nf : num_filters, dr : dilation_rate
# (conv1_nf, conv1_dr, convm_nf, convm_dr, conv2_nf, conv2_dr, scale_factor)
BASNET_BRIDGE_SPECS = [
            (512, 2, 512, 2, 512, 2, 32), #Sup0, Bridge
        ]

BASNET_DECODER_SPECS = [
            (512, 1, 512, 2, 512, 2, 32), #Sup1, stage6d
            (512, 1, 512, 1, 512, 1, 16), #Sup2, stage5d
            (512, 1, 512, 1, 256, 1, 8),  #Sup3, stage4d
            (256, 1, 256, 1, 128, 1, 4),  #Sup4, stage3d
            (128, 1, 128, 1, 64,  1, 2),  #Sup5, stage2d
            (64,  1, 64,  1, 64,  1, 1)   #Sup6, stage1d
        ]


@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNet_Decoder(tf.keras.layers.Layer):
  """Decoder of BASNet.

  Boundary-Awar network (BASNet) were proposed in:
  [1] Qin, Xuebin, et al. 
      Basnet: Boundary-aware salient object detection.
  """

  def __init__(self,
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
      use_separable_conv: `bool`, if True use separable convolution for
        convolution in BASNet layers.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      use_bias: if True, use bias in convolution.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    super(BASNet_Decoder, self).__init__(**kwargs)
    self._config_dict = {
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
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1
    self._activation = tf_utils.get_activation(activation)
    self._concat = tf.keras.layers.Concatenate(axis=-1)
    self._sigmoid = tf.keras.layers.Activation(activation='sigmoid')

  def build(self, input_shape):
    """Creates the variables of the BASNet decoder."""
    if self._config_dict['use_separable_conv']:
      conv_op = tf.keras.layers.SeparableConv2D
    else:
      conv_op = tf.keras.layers.Conv2D
    conv_kwargs = {
      'kernel_size': 3,
      'strides': 1,
      'use_bias': self._config_dict['use_bias'],
      'kernel_initializer': self._config_dict['kernel_initializer'],
      'kernel_regularizer': self._config_dict['kernel_regularizer'],
      'bias_regularizer': self._config_dict['bias_regularizer'],
    }

    self._out_convs = []
    self._out_usmps = []

    # Bridge layers.
    self._bdg_convs = []
    for i, spec in enumerate(BASNET_BRIDGE_SPECS):
      blocks = []
      for j in range(3):
        blocks.append(nn_blocks.ConvBlock(
            filters=spec[2*j],
            dilation_rate=spec[2*j+1],
            activation='relu',
            norm_momentum=0.99,
            norm_epsilon=0.001,
            **conv_kwargs))
      self._bdg_convs.append(blocks)
      self._out_convs.append(conv_op(
          filters=1,
          padding='same',
          **conv_kwargs))
      self._out_usmps.append(tf.keras.layers.UpSampling2D(
          size=spec[6],
          interpolation='bilinear'
          ))

    # Decoder layers.
    self._dec_convs = []
    for i, spec in enumerate(BASNET_DECODER_SPECS):
      blocks = []
      for j in range(3):
        blocks.append(nn_blocks.ConvBlock(
            filters=spec[2*j],
            dilation_rate=spec[2*j+1],
            activation='relu',
            norm_momentum=0.99,
            norm_epsilon=0.001,
            **conv_kwargs))
      self._dec_convs.append(blocks)
      self._out_convs.append(conv_op(
          filters=1,
          padding='same',
          **conv_kwargs))
      self._out_usmps.append(tf.keras.layers.UpSampling2D(
          size=spec[6],
          interpolation='bilinear'
          ))

  def call(self, backbone_output: Mapping[str, tf.Tensor]):
    levels = sorted(backbone_output.keys(), reverse=True)
    sup = {}
    x = backbone_output[levels[0]]

    for blocks in self._bdg_convs:
      for block in blocks:
        x = block(x)
    sup['0'] = x
    
    for i, blocks in enumerate(self._dec_convs):
      x = self._concat([x, backbone_output[levels[i]]])
      for block in blocks:
        x = block(x)
      sup[str(i+1)] = x
      x = tf.keras.layers.UpSampling2D(
          size=2,
          interpolation='bilinear'
          )(x)
    for i, (conv, usmp) in enumerate(zip(self._out_convs, self._out_usmps)):
      sup[str(i)] = self._sigmoid(usmp(conv(sup[str(i)])))

    self._output_specs = {
        str(order): sup[str(order)].get_shape()
        for order in range(0, len(BASNET_DECODER_SPECS))
    }

    return sup

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {order: TensorShape} pairs for the model output."""
    return self._output_specs
