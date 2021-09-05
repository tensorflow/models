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

"""Contains common building blocks for centernet neural networks."""

from typing import Optional, List

import tensorflow as tf

from official.vision.beta.modeling.layers import nn_layers
from official.modeling import tf_utils


def _get_padding_for_kernel_size(kernel_size):
  """Compute padding size given kernel size."""
  if kernel_size == 7:
    return (3, 3)
  elif kernel_size == 3:
    return (1, 1)
  else:
    raise ValueError('Padding for kernel size {} not known.'.format(
        kernel_size))


def _apply_blocks(inputs, blocks):
  """Apply blocks to inputs."""
  net = inputs
  
  for block in blocks:
    net = block(net)
  
  return net


def _make_repeated_residual_blocks(
    reps: int,
    out_channels: int,
    use_sync_bn: bool = True,
    norm_momentum: float = 0.1,
    norm_epsilon: float = 1e-5,
    residual_channels: Optional[int] = None,
    initial_stride: int = 1,
    initial_skip_conv: bool = False,
    kernel_initializer: str = 'VarianceScaling',
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
):
  """Stack Residual blocks one after the other.
  
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
    kernel_initializer: A `str` for kernel initializer of convolutional layers.
    kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
      Conv2D. Default to None.
    bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      Default to None.
      
  Returns:
    blocks: A list of residual blocks to be applied in sequence.
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
    
    blocks.append(ResidualEPBlock(
        filters=residual_channels,
        strides=stride,
        use_explicit_padding=True,
        use_projection=skip_conv,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer))
  
  if reps == 1:
    # If there is only 1 block, the `for` loop above is not run,
    # therefore we honor the requested stride in the last residual block
    stride = initial_stride
    # We are forced to use a conv in the skip connection if stride > 1
    skip_conv = stride > 1
  else:
    stride = 1
    skip_conv = residual_channels != out_channels
  
  blocks.append(ResidualEPBlock(
      filters=out_channels,
      strides=stride,
      use_explicit_padding=True,
      use_projection=skip_conv,
      use_sync_bn=use_sync_bn,
      norm_momentum=norm_momentum,
      norm_epsilon=norm_epsilon,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer))
  
  return tf.keras.Sequential(blocks)


@tf.keras.utils.register_keras_serializable(package='centernet')
class IdentityLayer(tf.keras.layers.Layer):
  """A layer which passes through the input as it is."""
  
  def call(self, inputs):
    return inputs


@tf.keras.utils.register_keras_serializable(package='centernet')
class Conv2DBNEPBlock(tf.keras.layers.Layer):
  """A convolution block with batch normalization."""
  
  def __init__(
      self,
      filters: int,
      kernel_size: int = 3,
      strides: int = 1,
      use_bias: bool = False,
      use_explicit_padding: bool = False,
      activation: str = 'relu6',
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_normalization: bool = True,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs):
    """A convolution block with batch normalization.
    
    This is a modification of Conv2DBNBlock under
    office/vision/beta/modeling/backbones/mobilenet:Conv2DBNBlock
    with additional explicit padding option.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      kernel_size: An `int` specifying the height and width of the 2D
        convolution window.
      strides: An `int` of block stride. If greater than 1, this block will
        ultimately downsample the input.
      use_bias: If True, use bias in the convolution layer.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      activation: A `str` name of the activation function.
      kernel_initializer: A `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      use_normalization: If True, use batch normalization.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(Conv2DBNEPBlock, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._use_bias = use_bias
    self._use_explicit_padding = use_explicit_padding
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_normalization = use_normalization
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if use_explicit_padding and kernel_size > 1:
      self._padding = 'VALID'
    else:
      self._padding = 'SAME'
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
  
  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'use_bias': self._use_bias,
        'use_explicit_padding': self._use_explicit_padding,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'use_normalization': self._use_normalization,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(Conv2DBNEPBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
  def build(self, input_shape):
    if self._use_explicit_padding and self._kernel_size > 1:
      padding_size = _get_padding_for_kernel_size(self._kernel_size)
      self._pad = tf.keras.layers.ZeroPadding2D(padding_size)
    else:
      self._pad = IdentityLayer()
    self._conv0 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding=self._padding,
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    if self._use_normalization:
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
    self._activation_layer = tf_utils.get_activation(
        self._activation, use_keras_layer=True)
    
    super(Conv2DBNEPBlock, self).build(input_shape)
  
  def call(self, inputs, training=None):
    inputs = self._pad(inputs)
    x = self._conv0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    return self._activation_layer(x)


@tf.keras.utils.register_keras_serializable(package='centernet')
class ResidualEPBlock(tf.keras.layers.Layer):
  """A residual block."""
  
  def __init__(self,
               filters,
               strides,
               use_projection=False,
               se_ratio=None,
               resnetd_shortcut=False,
               use_explicit_padding: bool = False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Initializes a residual block with BN after convolutions.

    This is a modification of ResidualBlock under
    office/vision/beta/modeling/layers/nn_blocks:ResidualBlock
    with additional explicit padding option.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      strides: An `int` block stride. If greater than 1, this block will
        ultimately downsample the input.
      use_projection: A `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      resnetd_shortcut: A `bool` if True, apply the resnetd style modification
        to the shortcut connection. Not implemented in residual blocks.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ResidualEPBlock, self).__init__(**kwargs)
    
    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._resnetd_shortcut = resnetd_shortcut
    self._use_explicit_padding = use_explicit_padding
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)
  
  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf.keras.layers.Conv2D(
          filters=self._filters,
          kernel_size=1,
          strides=self._strides,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
    
    conv1_padding = 'same'
    # explicit padding here is added for centernet
    if self._use_explicit_padding:
      self._pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
      conv1_padding = 'valid'
    else:
      self._pad = IdentityLayer()
    self._conv1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding=conv1_padding,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)
    
    self._conv2 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)
    
    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters,
          out_filters=self._filters,
          se_ratio=self._se_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None
    
    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    
    super(ResidualEPBlock, self).build(input_shape)
  
  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'resnetd_shortcut': self._resnetd_shortcut,
        'use_explicit_padding': self._use_explicit_padding,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(ResidualEPBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)
    
    inputs = self._pad(inputs)
    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)
    
    x = self._conv2(x)
    x = self._norm2(x)
    
    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)
    
    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)
    
    return self._activation_fn(x + shortcut)


@tf.keras.utils.register_keras_serializable(package='centernet')
class HourglassBlock(tf.keras.layers.Layer):
  """Hourglass module: an encoder-decoder block."""
  
  def __init__(
      self,
      channel_dims_per_stage: List[int],
      blocks_per_stage: List[int],
      strides: int = 1,
      use_sync_bn: bool = True,
      norm_momentum: float = 0.1,
      norm_epsilon: float = 1e-5,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialize Hourglass module.
    
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
      strides: `int`, stride parameter to the Residual block.
      use_sync_bn: A `bool`, if True, use synchronized batch normalization.
      norm_momentum: `float`, momentum for the batch normalization layers.
      norm_epsilon: `float`, epsilon for the batch normalization layers.
      kernel_initializer: A `str` for kernel initializer of conv layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      **kwargs: Additional keyword arguments to be passed.
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
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    
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
          norm_epsilon=self._norm_epsilon,
          bias_regularizer=self._bias_regularizer,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer)
    
    else:
      # outer hourglass structures
      self.encoder_block1 = _make_repeated_residual_blocks(
          reps=self._reps,
          out_channels=self._filters,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon,
          bias_regularizer=self._bias_regularizer,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer)
      
      self.encoder_block2 = _make_repeated_residual_blocks(
          reps=self._reps,
          out_channels=self._filters_downsampled,
          initial_stride=2,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon,
          bias_regularizer=self._bias_regularizer,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
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
          norm_epsilon=self._norm_epsilon,
          bias_regularizer=self._bias_regularizer,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer)
      
      self.upsample_layer = tf.keras.layers.UpSampling2D(
          size=2,
          interpolation='nearest')
    
    super(HourglassBlock, self).build(input_shape)
  
  def call(self, x, training=None):
    if self._num_stages == 0:
      return self.blocks(x)
    else:
      encoded_outputs = self.encoder_block1(x)
      encoded_downsampled_outputs = self.encoder_block2(x)
      inner_outputs = self.inner_hg(encoded_downsampled_outputs)
      hg_output = self.decoder_block(inner_outputs)
      return self.upsample_layer(hg_output) + encoded_outputs
  
  def get_config(self):
    config = {
        'channel_dims_per_stage': self._channel_dims_per_stage,
        'blocks_per_stage': self._blocks_per_stage,
        'strides': self._strides,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    config.update(super(HourglassBlock, self).get_config())
    return config


@tf.keras.utils.register_keras_serializable(package='centernet')
class CenterNetHeadConv(tf.keras.layers.Layer):
  """Convolution block for the CenterNet head."""
  
  def __init__(self,
               output_filters: int,
               bias_init: float,
               name: str,
               **kwargs):
    """Initialize CenterNet head.

    Args:
      output_filters: `int`, channel depth of layer output
      bias_init: `float`, value to initialize the bias vector for the final
        convolution layer
      name: `string`, layer name
      **kwargs: Additional keyword arguments to be passed.
    """
    super(CenterNetHeadConv, self).__init__(name=name, **kwargs)
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
    super(CenterNetHeadConv, self).build(input_shape)
  
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
    config.update(super(CenterNetHeadConv, self).get_config())
    return config
