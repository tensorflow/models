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

from typing import Optional, List

import tensorflow as tf

from official.vision.beta.modeling.layers import nn_blocks


def _apply_blocks(inputs, blocks):
  net = inputs
  
  for block in blocks:
    net = block(net)
  
  return net


def _make_repeated_residual_blocks(reps: int,
                                   out_channels: int,
                                   use_sync_bn: bool = True,
                                   norm_momentum: float = 0.1,
                                   norm_epsilon: float = 1e-5,
                                   residual_channels: Optional[int] = None,
                                   initial_stride: int = 1,
                                   initial_skip_conv: bool = False):
  """Stack Residual blocks one after the other..
  
  Args:
    reps: `int` for desired number of residual blocks
    out_channels: `int`, filter depth of the final residual block
    use_sync_bn: A `bool`, if True, use synchronized batch normalization.
    norm_momentum: `float`, momentum for the batch normalization layers
    norm_epsilon: `float`, epsilon for the batch normalization layers
    residual_channels: `int`, filter depth for the first reps - 1 residual
      blocks. If None, defaults to the same value as out_channels. If not
      equal to out_channels, then uses a projection shortcut in the final
      residual block
    initial_stride: `int`, stride for the first residual block
    initial_skip_conv: `bool`, if set, the first residual block uses a skip
      convolution. This is useful when the number of channels in the input
      are not the same as residual_channels.
  """
  blocks = []
  
  if residual_channels is None:
    residual_channels = out_channels
  
  for i in range(reps - 1):
    # Only use the stride at the first block so we don't repeatedly downsample
    # the input
    stride = initial_stride if i == 0 else 1
    
    # If the stride is more than 1, we cannot use an identity layer for the
    # skip connection and are forced to use a conv for the skip connection.
    skip_conv = stride > 1
    
    if i == 0 and initial_skip_conv:
      skip_conv = True
    
    blocks.append(nn_blocks.ResidualBlock(
        filters=residual_channels,
        strides=stride,
        use_projection=skip_conv,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon))
  
  if reps == 1:
    # If there is only 1 block, the `for` loop above is not run,
    # therefore we honor the requested stride in the last residual block
    stride = initial_stride
    # We are forced to use a conv in the skip connection if stride > 1
    skip_conv = stride > 1
  else:
    stride = 1
    skip_conv = residual_channels != out_channels
  
  blocks.append(nn_blocks.ResidualBlock(
      filters=out_channels,
      strides=stride,
      use_projection=skip_conv,
      use_sync_bn=use_sync_bn,
      norm_momentum=norm_momentum,
      norm_epsilon=norm_epsilon))
  
  return blocks


@tf.keras.utils.register_keras_serializable(package='centernet')
class HourglassBlock(tf.keras.layers.Layer):
  """
  Hourglass module: an encoder-decoder block which recursively
  defines the hourglass network.
  """
  
  def __init__(self,
               channel_dims_per_stage: List[int],
               blocks_per_stage: List[int],
               strides: int = 1,
               use_sync_bn: bool = True,
               norm_momentum: float = 0.1,
               norm_epsilon: float = 1e-5,
               **kwargs):
    """
    Args:
      channel_dims_per_stage: List[int], list of filter sizes for Residual
        blocks. the output channels dimensions of stages in
        the network. `channel_dims[0]` is used to define the number of
        channels in the first encoder block and `channel_dims[1]` is used to
        define the number of channels in the second encoder block. The channels
        in the recursive inner layers are defined using `channel_dims[1:]`.
        For example, [nc * 2, nc * 2, nc * 3, nc * 3, nc * 3, nc*4]
        where nc is the input_channel_dimension.
      blocks_per_stage: List[int], list of residual block repetitions per
        down/upsample. `blocks_per_stage[0]` defines the number of blocks at the
        current stage and `blocks_per_stage[1:]` is used at further stages.
        For example, [2, 2, 2, 2, 2, 4].
      strides: `int`, stride parameter to the Residual block
      use_sync_bn: A `bool`, if True, use synchronized batch normalization.
      norm_momentum: `float`, momentum for the batch normalization layers
      norm_epsilon: `float`, epsilon for the batch normalization layers
    """
    super(HourglassBlock, self).__init__(**kwargs)
    
    if len(channel_dims_per_stage) != len(blocks_per_stage):
      raise ValueError('filter size and residual block repetition '
                       'lists must have the same length')
    
    self._num_stages = len(channel_dims_per_stage) - 1
    self._channel_dims_per_stage = channel_dims_per_stage
    self._blocks_per_stage = blocks_per_stage
    self._strides = strides
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    
    self._filters = channel_dims_per_stage[0]
    if self._num_stages > 0:
      self._filters_downsampled = channel_dims_per_stage[1]
    
    self._reps = blocks_per_stage[0]
  
  def build(self, input_shape):
    if self._num_stages == 0:
      # base case, residual block repetitions in most inner part of hourglass
      self.blocks = _make_repeated_residual_blocks(
          reps=self._reps,
          out_channels=self._filters,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)
    
    else:
      # outer hourglass structures
      self.encoder_block1 = _make_repeated_residual_blocks(
          reps=self._reps,
          out_channels=self._filters,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)
      
      self.encoder_block2 = _make_repeated_residual_blocks(
          reps=self._reps,
          out_channels=self._filters_downsampled,
          initial_stride=2,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon,
          initial_skip_conv=self._filters != self._filters_downsampled)
      
      # recursively define inner hourglasses
      self.inner_hg = type(self)(
          channel_dims_per_stage=self._channel_dims_per_stage[1:],
          blocks_per_stage=self._blocks_per_stage[1:],
          strides=self._strides)
      
      # outer hourglass structures
      self.decoder_block = _make_repeated_residual_blocks(
          reps=self._reps,
          residual_channels=self._filters_downsampled,
          out_channels=self._filters,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)
      
      self.upsample_layer = tf.keras.layers.UpSampling2D(
          size=2,
          interpolation='nearest')
    
    super(HourglassBlock).build(input_shape)
  
  def call(self, x, training=None):
    if self._num_stages == 0:
      return self._apply_blocks(self.blocks, x)
    else:
      encoded_outputs = _apply_blocks(x, self.encoder_block1)
      encoded_downsampled_outputs = _apply_blocks(x, self.encoder_block2)
      inner_outputs = self.inner_hg(encoded_downsampled_outputs)
      hg_output = _apply_blocks(inner_outputs, self.decoder_block)
      return self.upsample_layer(hg_output) + encoded_outputs
  
  def get_config(self):
    config = {
        'channel_dims_per_stage': self._channel_dims_per_stage,
        'blocks_per_stage': self._blocks_per_stage,
        'strides': self._strides,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    config.update(super(HourglassBlock).get_config())
    return config


@tf.keras.utils.register_keras_serializable(package='centernet')
class CenterNetDecoderConv(tf.keras.layers.Layer):
  """
  Convolution block for the CenterNet head. This is used to generate
  both the confidence heatmaps and other regressed predictions such as 
  center offsets, object size, etc.
  """
  
  def __init__(self,
               output_filters: int,
               bias_init: float,
               name: str,
               **kwargs):
    """
    Args:
      output_filters: `int`, channel depth of layer output
      bias_init: `float`, value to initialize the bias vector for the final
        convolution layer
      name: `string`, layer name
    """
    super(CenterNetDecoderConv).__init__(name=name, **kwargs)
    self._output_filters = output_filters
    self._bias_init = bias_init
  
  def build(self, input_shape):
    n_channels = input_shape[-1]
    
    self.conv1 = tf.keras.layers.Conv2D(
        filters=n_channels,
        kernel_size=(3, 3),
        padding='same')
    
    self.relu = tf.keras.layers.ReLU()
    
    # Initialize bias to the last Conv2D Layer
    self.conv2 = tf.keras.layers.Conv2D(
        filters=self._output_filters,
        kernel_size=(1, 1),
        padding='valid',
        bias_initializer=tf.constant_initializer(self._bias_init))
    super(CenterNetDecoderConv, self).build(input_shape)
  
  def call(self, x, training=None):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    return x
  
  def get_config(self):
    config = {
        'output_filters': self._output_filters,
        'bias_init': self._bias_init,
    }
    config.update(super(CenterNetDecoderConv).get_config())
    return config
