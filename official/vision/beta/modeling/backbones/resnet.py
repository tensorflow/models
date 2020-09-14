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
"""Contains definitions of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

# Import libraries
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks

layers = tf.keras.layers

# Specifications for different ResNet variants.
# Each entry specifies block configurations of the particular ResNet variant.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, block_repeats)
RESNET_SPECS = {
    18: [
        ('residual', 64, 2),
        ('residual', 128, 2),
        ('residual', 256, 2),
        ('residual', 512, 2),
    ],
    34: [
        ('residual', 64, 3),
        ('residual', 128, 4),
        ('residual', 256, 6),
        ('residual', 512, 3),
    ],
    50: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 6),
        ('bottleneck', 512, 3),
    ],
    101: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 23),
        ('bottleneck', 512, 3),
    ],
    152: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 8),
        ('bottleneck', 256, 36),
        ('bottleneck', 512, 3),
    ],
    200: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 24),
        ('bottleneck', 256, 36),
        ('bottleneck', 512, 3),
    ],
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResNet(tf.keras.Model):
  """Class to build ResNet family model."""

  def __init__(self,
               model_id,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """ResNet initialization function.

    Args:
      model_id: `int` depth of ResNet backbone model.
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
    self._model_id = model_id
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
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    x = layers.Conv2D(
        filters=64, kernel_size=7, strides=2, use_bias=False, padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    x = tf_utils.get_activation(activation)(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # TODO(xianzhi): keep a list of blocks to make blocks accessible.
    endpoints = {}
    for i, spec in enumerate(RESNET_SPECS[model_id]):
      if spec[0] == 'residual':
        block_fn = nn_blocks.ResidualBlock
      elif spec[0] == 'bottleneck':
        block_fn = nn_blocks.BottleneckBlock
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))
      x = self._block_group(
          inputs=x,
          filters=spec[1],
          strides=(1 if i == 0 else 2),
          block_fn=block_fn,
          block_repeats=spec[2],
          name='block_group_l{}'.format(i + 2))
      endpoints[str(i + 2)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(ResNet, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs,
                   filters,
                   strides,
                   block_fn,
                   block_repeats=1,
                   name='block_group'):
    """Creates one group of blocks for the ResNet model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
      block_fn: Either `nn_blocks.ResidualBlock` or `nn_blocks.BottleneckBlock`.
      block_repeats: `int` number of blocks contained in the layer.
      name: `str`name for the block.

    Returns:
      The output `Tensor` of the block layer.
    """
    x = block_fn(
        filters=filters,
        strides=strides,
        use_projection=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)

    for _ in range(1, block_repeats):
      x = block_fn(
          filters=filters,
          strides=1,
          use_projection=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              x)

    return tf.identity(x, name=name)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
