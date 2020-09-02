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
"""Contains definitions of 3D Residual Networks."""
from typing import List, Tuple

# Import libraries
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks_3d

layers = tf.keras.layers

RESNET_SPECS = {
    50: [
        ('bottleneck3d', 64, 3),
        ('bottleneck3d', 128, 4),
        ('bottleneck3d', 256, 6),
        ('bottleneck3d', 512, 3),
    ],
    101: [
        ('bottleneck3d', 64, 3),
        ('bottleneck3d', 128, 4),
        ('bottleneck3d', 256, 23),
        ('bottleneck3d', 512, 3),
    ],
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResNet3D(tf.keras.Model):
  """Class to build 3D ResNet family model."""

  def __init__(self,
               model_id: int,
               temporal_strides: List[int],
               temporal_kernel_sizes: List[Tuple[int]],
               use_self_gating: List[int] = None,
               input_specs=layers.InputSpec(shape=[None, None, None, None, 3]),
               stem_conv_temporal_stride=2,
               stem_pool_temporal_stride=2,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """ResNet3D initialization function.

    Args:
      model_id: `int` depth of ResNet backbone model.
      temporal_strides: a list of integers that specifies the temporal strides
        for all 3d blocks.
      temporal_kernel_sizes: a list of tuples that specifies the temporal kernel
        sizes for all 3d blocks in different block groups.
      use_self_gating: a list of booleans to specify applying self-gating module
        or not in each block group. If None, self-gating is not applied.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      stem_conv_temporal_stride: `int` temporal stride for the first conv layer.
      stem_pool_temporal_stride: `int` temporal stride for the first pool layer.
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
    self._temporal_strides = temporal_strides
    self._temporal_kernel_sizes = temporal_kernel_sizes
    self._input_specs = input_specs
    self._stem_conv_temporal_stride = stem_conv_temporal_stride
    self._stem_pool_temporal_stride = stem_pool_temporal_stride
    self._use_self_gating = use_self_gating
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

    # Build ResNet3D backbone.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    # Build stem.
    x = layers.Conv3D(
        filters=64,
        kernel_size=[5, 7, 7],
        strides=[stem_conv_temporal_stride, 2, 2],
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    x = tf_utils.get_activation(activation)(x)

    temporal_kernel_size = 1 if stem_pool_temporal_stride == 1 else 3
    x = layers.MaxPool3D(
        pool_size=[temporal_kernel_size, 3, 3],
        strides=[stem_pool_temporal_stride, 2, 2],
        padding='same')(
            x)

    # Build intermediate blocks and endpoints.
    resnet_specs = RESNET_SPECS[model_id]
    if len(temporal_strides) != len(resnet_specs) or len(
        temporal_kernel_sizes) != len(resnet_specs):
      raise ValueError(
          'Number of blocks in temporal specs should equal to resnet_specs.')

    endpoints = {}
    for i, resnet_spec in enumerate(resnet_specs):
      if resnet_spec[0] == 'bottleneck3d':
        block_fn = nn_blocks_3d.BottleneckBlock3D
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(
            resnet_spec[0]))

      x = self._block_group(
          inputs=x,
          filters=resnet_spec[1],
          temporal_kernel_sizes=temporal_kernel_sizes[i],
          temporal_strides=temporal_strides[i],
          spatial_strides=(1 if i == 0 else 2),
          block_fn=block_fn,
          block_repeats=resnet_spec[2],
          use_self_gating=use_self_gating[i] if use_self_gating else False,
          name='block_group_l{}'.format(i + 2))
      endpoints[i + 2] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(ResNet3D, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs,
                   filters,
                   temporal_kernel_sizes,
                   temporal_strides,
                   spatial_strides,
                   block_fn=nn_blocks_3d.BottleneckBlock3D,
                   block_repeats=1,
                   use_self_gating=False,
                   name='block_group'):
    """Creates one group of blocks for the ResNet3D model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      temporal_kernel_sizes: a tuple that specifies the temporal kernel sizes
        for each block in the current group.
      temporal_strides: `int` temporal strides for the first convolution in this
        group.
      spatial_strides: `int` stride to use for the first convolution of the
        layer. If greater than 1, this layer will downsample the input.
      block_fn: Either `nn_blocks.ResidualBlock` or `nn_blocks.BottleneckBlock`.
      block_repeats: `int` number of blocks contained in the layer.
      use_self_gating: `bool` apply self-gating module or not.
      name: `str`name for the block.

    Returns:
      The output `Tensor` of the block layer.
    """
    if len(temporal_kernel_sizes) != block_repeats:
      raise ValueError(
          'Number of elements in `temporal_kernel_sizes` must equal to `block_repeats`.'
      )

    # Only apply self-gating module in the last block.
    use_self_gating_list = [False] * (block_repeats - 1) + [use_self_gating]

    x = block_fn(
        filters=filters,
        temporal_kernel_size=temporal_kernel_sizes[0],
        temporal_strides=temporal_strides,
        spatial_strides=spatial_strides,
        use_self_gating=use_self_gating_list[0],
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)

    for i in range(1, block_repeats):
      x = block_fn(
          filters=filters,
          temporal_kernel_size=temporal_kernel_sizes[i],
          temporal_strides=1,
          spatial_strides=1,
          use_self_gating=use_self_gating_list[i],
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
        'temporal_strides': self._temporal_strides,
        'temporal_kernel_sizes': self._temporal_kernel_sizes,
        'stem_conv_temporal_stride': self._stem_conv_temporal_stride,
        'stem_pool_temporal_stride': self._stem_pool_temporal_stride,
        'use_self_gating': self._use_self_gating,
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
