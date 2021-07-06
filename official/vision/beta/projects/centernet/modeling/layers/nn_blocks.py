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

import tensorflow as tf

from official.vision.beta.modeling.layers import nn_layers

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='centernet')
class Identity(tf.keras.layers.Layer):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def call(self, input):
    return input


@tf.keras.utils.register_keras_serializable(package='centernet')
class CenterNetResidualBlock(tf.keras.layers.Layer):
  """A residual block."""
  
  def __init__(self,
               filters,
               strides,
               use_projection=False,
               se_ratio=None,
               resnetd_shortcut=False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A residual block with BN after convolutions. Modified with padding for 
    the CenterNet model. The input is first padded with 0 along the top, bottom, 
    left, and right prior to the first convolutional layer.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: `float` or None. Ratio of the Squeeze-and-Excitation layer.
      resnetd_shortcut: `bool` if True, apply the resnetd style modification to
        the shortcut connection. Not implemented in residual blocks.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(CenterNetResidualBlock, self).__init__(**kwargs)
    
    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._resnetd_shortcut = resnetd_shortcut
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
    self.pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
    self._conv1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding='valid',
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
    
    super(CenterNetResidualBlock, self).build(input_shape)
  
  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'resnetd_shortcut': self._resnetd_shortcut,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(CenterNetResidualBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)
    
    x = self.pad(inputs)
    x = self._conv1(x)
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
class CenterNetConvBN(tf.keras.layers.Layer):
  def __init__(self,
               filters=1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               dilation_rate=(1, 1),
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               subdivisions=1,
               bias_regularizer=None,
               kernel_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='leaky',
               leaky_alpha=0.1,
               **kwargs):
    """
    Modified Convolution layer based on the DarkNet Library, with changes
    such that it is compatiable with the CenterNet backbone.
    The Layer is a standards combination of Conv BatchNorm Activation,
    however, the use of bias in the conv is determined by the use of batch
    normalization. Modified with padding for the CenterNet model. The input is 
    first padded with 0 along the top, bottom, left, and right prior to the 
    first convolutional layer.
    Cross Stage Partial networks (CSPNets) were proposed in:
    [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu,
          Ping-Yang Chen, Jun-Wei Hsieh
        CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
          arXiv:1911.11929
    Args:
      filters: integer for output depth, or the number of features to learn
      kernel_size: integer or tuple for the shape of the weight matrix or kernel
        to learn
      strides: integer of tuple how much to move the kernel after each kernel
        use
      padding: string 'valid' or 'same', if same, then pad the image, else do
        not
      dialtion_rate: tuple to indicate how much to modulate kernel weights and
        how many pixels in a feature map to skip
      kernel_initializer: string to indicate which function to use to initialize
        weights
      bias_initializer: string to indicate which function to use to initialize
        bias
      kernel_regularizer: string to indicate which function to use to
        regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer
        bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
        of all batch norm layers to the models global statistics
        (across all input batches)
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
        if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      **kwargs: Keyword Arguments
    """
    
    # convolution params
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._dilation_rate = dilation_rate
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions
    
    # batch normalization params
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    
    if tf.keras.backend.image_data_format() == 'channels_last':
      # format: (batch_size, height, width, channels)
      self._bn_axis = -1
    else:
      # format: (batch_size, channels, width, height)
      self._bn_axis = 1
    
    # activation params
    self._activation = activation
    self._leaky_alpha = leaky_alpha
    
    super().__init__(**kwargs)
  
  def build(self, input_shape):
    use_bias = not self._use_bn
    padding = self._padding
    
    kernel_size = self._kernel_size if isinstance(
        self._kernel_size, int) else self._kernel_size[0]
    
    if kernel_size > 1:
      padding = 'valid'
      if kernel_size == 7:
        padding_size = (3, 3)
      if kernel_size == 3:
        padding_size = (1, 1)
      
      self.pad = tf.keras.layers.ZeroPadding2D(padding_size)
    else:
      self.pad = Identity()
    
    self.conv = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding=padding,
        dilation_rate=self._dilation_rate,
        use_bias=use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    
    if self._use_bn:
      if self._use_sync_bn:
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
      else:
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
    
    if self._activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(
          self._activation)  # tf.keras.layers.Activation(self._activation)
  
  def call(self, x):
    x = self.pad(x)
    x = self.conv(x)
    if self._use_bn:
      x = self.bn(x)
    x = self._activation_fn(x)
    return x
  
  def get_config(self):
    # used to store/share parameters to reconstruct the model
    layer_config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'padding': self._padding,
        'dilation_rate': self._dilation_rate,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'leaky_alpha': self._leaky_alpha
    }
    layer_config.update(super().get_config())
    return layer_config


@tf.keras.utils.register_keras_serializable(package='centernet')
class HourglassBlock(tf.keras.layers.Layer):
  """
  Hourglass module
  """
  
  def __init__(self,
               channel_dims_per_stage,
               blocks_per_stage,
               strides=1,
               norm_momentum=0.1,
               norm_epsilon=1e-5,
               **kwargs):
    """
    Args:
      channel_dims_per_stage: list of filter sizes for Residual blocks
      blocks_per_stage: list of residual block repetitions per down/upsample
      strides: integer, stride parameter to the Residual block
      norm_momentum: float, momentum for the batch normalization layers
      norm_episilon: float, epsilon for the batch normalization layers
    """
    self._order = len(channel_dims_per_stage) - 1
    self._channel_dims_per_stage = channel_dims_per_stage
    self._blocks_per_stage = blocks_per_stage
    self._strides = strides
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    
    assert len(channel_dims_per_stage) == len(blocks_per_stage), 'filter ' \
                                                                 'size and residual block repetition lists must have the same length'
    
    self._filters = channel_dims_per_stage[0]
    if self._order > 0:
      self._filters_downsampled = channel_dims_per_stage[1]
    self._reps = blocks_per_stage[0]
    
    super().__init__(**kwargs)
  
  def make_repeated_residual_blocks(self, reps, out_channels,
                                    residual_channels=None,
                                    initial_stride=1):
    """
    Args:
      reps: int for desired number of residual blocks 
      out_channels: int, filter depth of the final residual block
      residual_channels: int, filter depth for the first reps - 1 residual 
        blocks. If None, defaults to the same value as out_channels. If not 
        equal to out_channels, then uses a projection shortcut in the final 
        residual block
      initial_stride: int, stride for the first residual block
    """
    blocks = []
    
    if residual_channels is None:
      residual_channels = out_channels
    
    for i in range(reps - 1):
      stride = initial_stride if i == 0 else 1
      skip_conv = stride > 1
      
      blocks.append(CenterNetResidualBlock(
          filters=residual_channels, strides=stride,
          use_projection=skip_conv, use_sync_bn=True,
          norm_momentum=self._norm_momentum, norm_epsilon=self._norm_epsilon))
    
    if reps == 1:
      stride = initial_stride
      skip_conv = stride > 1
    else:
      stride = 1
      skip_conv = residual_channels != out_channels
    
    blocks.append(CenterNetResidualBlock(
        filters=out_channels, strides=stride,
        use_projection=skip_conv, use_sync_bn=True,
        norm_momentum=self._norm_momentum, norm_epsilon=self._norm_epsilon))
    
    return tf.keras.Sequential(blocks)
  
  def build(self, input_shape):
    if self._order == 0:
      # base case, residual block repetitions in most inner part of hourglass
      self.blocks = self.make_repeated_residual_blocks(reps=self._reps,
                                                       out_channels=self._filters)
    
    else:
      # outer hourglass structures
      self.encoder_block1 = self.make_repeated_residual_blocks(reps=self._reps,
                                                               out_channels=self._filters)
      
      self.encoder_block2 = self.make_repeated_residual_blocks(reps=self._reps,
                                                               out_channels=self._filters_downsampled,
                                                               initial_stride=2)
      
      # recursively define inner hourglasses
      self.inner_hg = type(self)(
          channel_dims_per_stage=self._channel_dims_per_stage[1:],
          blocks_per_stage=self._blocks_per_stage[1:],
          strides=self._strides)
      
      # outer hourglass structures
      self.decoder_block = self.make_repeated_residual_blocks(reps=self._reps,
                                                              residual_channels=self._filters_downsampled,
                                                              out_channels=self._filters)
      
      self.upsample_layer = tf.keras.layers.UpSampling2D(
          size=2, interpolation='nearest')
    
    super().build(input_shape)
  
  def call(self, x):
    if self._order == 0:
      return self.blocks(x)
    else:
      encoded_outputs = self.encoder_block1(x)
      encoded_downsampled_outputs = self.encoder_block2(x)
      inner_outputs = self.inner_hg(encoded_downsampled_outputs)
      hg_output = self.decoder_block(inner_outputs)
      return self.upsample_layer(hg_output) + encoded_outputs
  
  def get_config(self):
    layer_config = {
        'channel_dims_per_stage': self._channel_dims_per_stage,
        'blocks_per_stage': self._blocks_per_stage,
        'strides': self._strides
    }
    layer_config.update(super().get_config())
    return layer_config


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
      output_filters: int, channel depth of layer output
      bias_init: float, value to initialize the bias vector for the final
        convolution layer
      name: string, layer name
    """
    super().__init__(name=name, **kwargs)
    self._output_filters = output_filters
    self._bias_init = bias_init
  
  def build(self, input_shape):
    n_channels = input_shape[-1]
    
    self.conv1 = tf.keras.layers.Conv2D(filters=n_channels,
                                        kernel_size=(3, 3), padding='same')
    
    self.relu = tf.keras.layers.ReLU()
    
    # Initialize bias to the last Conv2D Layer
    self.conv2 = tf.keras.layers.Conv2D(filters=self._output_filters,
                                        kernel_size=(1, 1), padding='valid',
                                        bias_initializer=tf.constant_initializer(
                                          self._bias_init))
  
  def call(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    return x
