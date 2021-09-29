# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Build BASNet models."""

from typing import Mapping

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.basnet.modeling import nn_blocks
from official.vision.beta.modeling.backbones import factory

# Specifications for BASNet encoder.
# Each element in the block configuration is in the following format:
# (num_filters, stride, block_repeats, maxpool)
BASNET_ENCODER_SPECS = [
    (64, 1, 3, 0),  # ResNet-34,
    (128, 2, 4, 0),  # ResNet-34,
    (256, 2, 6, 0),  # ResNet-34,
    (512, 2, 3, 1),  # ResNet-34,
    (512, 1, 3, 1),  # BASNet,
    (512, 1, 3, 0),  # BASNet,
]

# Specifications for BASNet decoder.
# Each element in the block configuration is in the following format:
# (conv1_nf, conv1_dr, convm_nf, convm_dr, conv2_nf, conv2_dr, scale_factor)
# nf : num_filters, dr : dilation_rate
BASNET_BRIDGE_SPECS = [
    (512, 2, 512, 2, 512, 2, 32),  # Sup0, Bridge
]

BASNET_DECODER_SPECS = [
    (512, 1, 512, 2, 512, 2, 32),  # Sup1, stage6d
    (512, 1, 512, 1, 512, 1, 16),  # Sup2, stage5d
    (512, 1, 512, 1, 256, 1, 8),  # Sup3, stage4d
    (256, 1, 256, 1, 128, 1, 4),  # Sup4, stage3d
    (128, 1, 128, 1, 64, 1, 2),  # Sup5, stage2d
    (64, 1, 64, 1, 64, 1, 1)  # Sup6, stage1d
]


@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNetModel(tf.keras.Model):
  """A BASNet model.

  Boundary-Awar network (BASNet) were proposed in:
  [1] Qin, Xuebin, et al.
      Basnet: Boundary-aware salient object detection.

  Input images are passed through backbone first. Decoder network is then
  applied, and finally, refinement module is applied on the output of the
  decoder network.
  """

  def __init__(self,
               backbone,
               decoder,
               refinement=None,
               **kwargs):
    """BASNet initialization function.

    Args:
      backbone: a backbone network. basnet_encoder.
      decoder: a decoder network. basnet_decoder.
      refinement: a module for salient map refinement.
      **kwargs: keyword arguments to be passed.
    """
    super(BASNetModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'refinement': refinement,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.refinement = refinement

  def call(self, inputs, training=None):
    features = self.backbone(inputs)

    if self.decoder:
      features = self.decoder(features)

    levels = sorted(features.keys())
    new_key = str(len(levels))
    if self.refinement:
      features[new_key] = self.refinement(features[levels[-1]])

    return features

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    if self.refinement is not None:
      items.update(refinement=self.refinement)
    return items

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNetEncoder(tf.keras.Model):
  """BASNet encoder."""

  def __init__(
      self,
      input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      activation='relu',
      use_sync_bn=False,
      use_bias=True,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_regularizer=None,
      **kwargs):
    """BASNet encoder initialization function.

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

    # Build BASNet Encoder.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=1,
        use_bias=self._use_bias, padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    x = tf_utils.get_activation(activation)(x)

    endpoints = {}
    for i, spec in enumerate(BASNET_ENCODER_SPECS):
      x = self._block_group(
          inputs=x,
          filters=spec[0],
          strides=spec[1],
          block_repeats=spec[2],
          name='block_group_l{}'.format(i + 2))
      endpoints[str(i)] = x
      if spec[3]:
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    super(BASNetEncoder, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs,
                   filters,
                   strides,
                   block_repeats=1,
                   name='block_group'):
    """Creates one group of residual blocks for the BASNet encoder model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
      block_repeats: `int` number of blocks contained in the layer.
      name: `str`name for the block.

    Returns:
      The output `Tensor` of the block layer.
    """
    x = nn_blocks.ResBlock(
        filters=filters,
        strides=strides,
        use_projection=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        use_bias=self._use_bias,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)

    for _ in range(1, block_repeats):
      x = nn_blocks.ResBlock(
          filters=filters,
          strides=1,
          use_projection=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          use_bias=self._use_bias,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              x)

    return tf.identity(x, name=name)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('basnet_encoder')
def build_basnet_encoder(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds BASNet Encoder backbone from a config."""
  backbone_type = model_config.backbone.type
  norm_activation_config = model_config.norm_activation
  assert backbone_type == 'basnet_encoder', (f'Inconsistent backbone type '
                                             f'{backbone_type}')
  return BASNetEncoder(
      input_specs=input_specs,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_bias=norm_activation_config.use_bias,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)


@tf.keras.utils.register_keras_serializable(package='Vision')
class BASNetDecoder(tf.keras.layers.Layer):
  """BASNet decoder."""

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
    """BASNet decoder initialization function.

    Args:
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
    super(BASNetDecoder, self).__init__(**kwargs)
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

    self._activation = tf_utils.get_activation(activation)
    self._concat = tf.keras.layers.Concatenate(axis=-1)
    self._sigmoid = tf.keras.layers.Activation(activation='sigmoid')

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

    self._out_convs = []
    self._out_usmps = []

    # Bridge layers.
    self._bdg_convs = []
    for spec in BASNET_BRIDGE_SPECS:
      blocks = []
      for j in range(3):
        blocks.append(nn_blocks.ConvBlock(
            filters=spec[2*j],
            dilation_rate=spec[2*j+1],
            activation='relu',
            use_sync_bn=self._config_dict['use_sync_bn'],
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
    for spec in BASNET_DECODER_SPECS:
      blocks = []
      for j in range(3):
        blocks.append(nn_blocks.ConvBlock(
            filters=spec[2*j],
            dilation_rate=spec[2*j+1],
            activation='relu',
            use_sync_bn=self._config_dict['use_sync_bn'],
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
    """Forward pass of the BASNet decoder.

    Args:
      backbone_output: A `dict` of tensors
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].

    Returns:
      sup: A `dict` of tensors
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].
    """
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
