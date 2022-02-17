# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Definitions of building blocks for MOSAIC model.

Reference:
   [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
   Context](https://arxiv.org/pdf/2112.11623.pdf)
"""

from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Vision')
class MultiKernelGroupConvBlock(tf.keras.layers.Layer):
  """A multi-kernel grouped convolution block.

  This block is used in the segmentation neck introduced in MOSAIC.
  Reference:
   [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
   Context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def __init__(
      self,
      output_filter_depths: List[int],
      kernel_sizes: List[int],
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout: float = 0.5,
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_depthwise_convolution: bool = True,
      **kwargs):
    """Initializes a Multi-kernel Grouped Convolution Block.

    Args:
      output_filter_depths: A list of integers representing the numbers of
        output channels or filter depths of convolution groups.
      kernel_sizes: A list of integers denoting the convolution kernel sizes in
        each convolution group.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for the activation fuction type. Defaults to 'relu'.
      dropout: A float for the dropout rate before output. Defaults to 0.5.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      use_depthwise_convolution: Allows spatial pooling to be separable
        depthwise convolusions.
      **kwargs: Other keyword arguments for the layer.
    """
    super(MultiKernelGroupConvBlock, self).__init__(**kwargs)

    if len(output_filter_depths) != len(kernel_sizes):
      raise ValueError('The number of output groups must match #kernels.')
    self._output_filter_depths = output_filter_depths
    self._kernel_sizes = kernel_sizes
    self._num_groups = len(self._kernel_sizes)
    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._dropout = dropout
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._use_depthwise_convolution = use_depthwise_convolution
    # To apply BN before activation. Putting BN between conv and activation also
    # helps quantization where conv+bn+activation are fused into a single op.
    self._activation_fn = tf_utils.get_activation(activation)
    if self._use_sync_bn:
      self._bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._bn_op = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
      self._group_split_axis = -1
    else:
      self._bn_axis = 1
      self._group_split_axis = 1

  def build(self, input_shape: List[int]) -> None:
    """Builds the block with the given input shape."""
    input_channels = input_shape[self._group_split_axis]
    if input_channels % self._num_groups != 0:
      raise ValueError('The number of input channels must be divisible by '
                       'the number of groups for evenly group split.')
    self._conv_branches = []
    if self._use_depthwise_convolution:
      for i, conv_kernel_size in enumerate(self._kernel_sizes):
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(conv_kernel_size, conv_kernel_size),
            depth_multiplier=1,
            padding='same',
            depthwise_regularizer=self._kernel_regularizer,
            depthwise_initializer=self._kernel_initializer,
            use_bias=False)
        feature_conv = tf.keras.layers.Conv2D(
            filters=self._output_filter_depths[i],
            kernel_size=(1, 1),
            padding='same',
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._kernel_initializer,
            activation=None,
            use_bias=False)
        batchnorm_op = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        # Use list manually as current QAT API does not support sequential model
        # within a tf.keras.Sequential block, e.g. conv_branch =
        # tf.keras.Sequential([depthwise_conv, feature_conv, batchnorm_op,])
        conv_branch = [depthwise_conv, feature_conv, batchnorm_op]
        self._conv_branches.append(conv_branch)
    else:
      for i, conv_kernel_size in enumerate(self._kernel_sizes):
        norm_conv = tf.keras.layers.Conv2D(
            filters=self._output_filter_depths[i],
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            activation=None,
            use_bias=False)
        batchnorm_op = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        conv_branch = [norm_conv, batchnorm_op]
        self._conv_branches.append(conv_branch)
    self._concat_groups = tf.keras.layers.Concatenate(
        axis=self._group_split_axis)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls this group convolution block with the given inputs."""
    inputs_splits = tf.split(inputs,
                             num_or_size_splits=self._num_groups,
                             axis=self._group_split_axis)
    output_branches = []
    for i, x in enumerate(inputs_splits):
      conv_branch = self._conv_branches[i]
      # Apply layers sequentially and manually.
      for layer in conv_branch:
        x = layer(x)
      # Apply activation function after BN, which also helps quantization
      # where conv+bn+activation are fused into a single op.
      x = self._activation_fn(x)
      output_branches.append(x)
    x = self._concat_groups(output_branches)
    return x

  def get_config(self) -> Dict[str, Any]:
    """Returns a config dictionary for initialization from serialization."""
    config = {
        'output_filter_depths': self._output_filter_depths,
        'kernel_sizes': self._kernel_sizes,
        'num_groups': self._num_groups,
        'use_sync_bn': self._use_sync_bn,
        'batchnorm_momentum': self._batchnorm_momentum,
        'batchnorm_epsilon': self._batchnorm_epsilon,
        'activation': self._activation,
        'dropout': self._dropout,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_depthwise_convolution': self._use_depthwise_convolution,
    }
    base_config = super(MultiKernelGroupConvBlock, self).get_config()
    base_config.update(config)
    return base_config


@tf.keras.utils.register_keras_serializable(package='Vision')
class MosaicEncoderBlock(tf.keras.layers.Layer):
  """Implements the encoder module/block of MOSAIC model.

  Spatial Pyramid Pooling and Multi-kernel Conv layer
  SpatialPyramidPoolingMultiKernelConv
  References:
    [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
    context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def __init__(
      self,
      branch_filter_depths: List[int],
      conv_kernel_sizes: List[int],
      pyramid_pool_bin_nums: List[int],
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout: float = 0.5,
      kernel_initializer: str = 'glorot_uniform',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      use_depthwise_convolution: bool = True,
      **kwargs):
    """Initializes a MOSAIC encoder block which is deployed after a backbone.

    Args:
      branch_filter_depths: A list of integers for the number of convolution
        channels in each branch at a pyramid level after SpatialPyramidPooling.
      conv_kernel_sizes: A list of integers representing the convolution kernel
        sizes in the Multi-kernel Convolution blocks in the encoder.
      pyramid_pool_bin_nums: A list of integers for the number of bins at each
        level of the Spatial Pyramid Pooling.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for the activation function type. Defaults to 'relu'.
      dropout: A float for the dropout rate before output. Defaults to 0.5.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      use_depthwise_convolution: Use depthwise separable convolusions in the
        Multi-kernel Convolution blocks in the encoder.
      **kwargs: Other keyword arguments for the layer.
    """
    super().__init__(**kwargs)

    self._branch_filter_depths = branch_filter_depths
    self._conv_kernel_sizes = conv_kernel_sizes
    self._pyramid_pool_bin_nums = pyramid_pool_bin_nums
    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._dropout = dropout
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._interpolation = interpolation
    self._use_depthwise_convolution = use_depthwise_convolution
    self._activation_fn = tf_utils.get_activation(activation)
    if self._use_sync_bn:
      self._bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._bn_op = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
      self._channel_axis = -1
    else:
      self._bn_axis = 1
      self._channel_axis = 1

  def get_bin_pool_kernel_and_stride(
      self,
      input_size: int,
      num_of_bin: int) -> Tuple[int, int]:
    """Calculates the kernel size and stride for spatial bin pooling.

    Args:
      input_size: Input dimension (a scalar).
      num_of_bin: The number of bins used for spatial bin pooling.

    Returns:
      The Kernel and Stride for spatial bin pooling (a scalar).
    """
    bin_overlap = int(input_size % num_of_bin)
    pooling_stride = int(input_size // num_of_bin)
    pooling_kernel = pooling_stride + bin_overlap
    return pooling_kernel, pooling_stride

  def build(self, input_shape: List[int]) -> None:
    """Builds this MOSAIC encoder block with the given input shape."""
    self._data_format = tf.keras.backend.image_data_format()
    if self._data_format == 'channels_last':
      height = input_shape[1]
      width = input_shape[2]
    else:
      height = input_shape[2]
      width = input_shape[3]

    self._global_pool_branch = None
    self._spatial_pyramid = []

    for pyramid_pool_bin_num in self._pyramid_pool_bin_nums:
      if pyramid_pool_bin_num == 1:
        global_pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self._data_format, keepdims=True)
        global_projection = tf.keras.layers.Conv2D(
            filters=max(self._branch_filter_depths),
            kernel_size=(1, 1),
            padding='same',
            activation=None,
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._kernel_initializer,
            use_bias=False)
        batch_norm_global_branch = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        # Use list manually instead of tf.keras.Sequential([])
        self._global_pool_branch = [
            global_pool,
            global_projection,
            batch_norm_global_branch,
        ]
      else:
        if height < pyramid_pool_bin_num or width < pyramid_pool_bin_num:
          raise ValueError('The number of pooling bins must be smaller than '
                           'input sizes.')
        assert pyramid_pool_bin_num >= 2, (
            'Except for the gloabl pooling, the number of bins in pyramid '
            'pooling must be at least two.')
        pool_height, stride_height = self.get_bin_pool_kernel_and_stride(
            height, pyramid_pool_bin_num)
        pool_width, stride_width = self.get_bin_pool_kernel_and_stride(
            width, pyramid_pool_bin_num)
        bin_pool_level = tf.keras.layers.AveragePooling2D(
            pool_size=(pool_height, pool_width),
            strides=(stride_height, stride_width),
            padding='valid',
            data_format=self._data_format)
        self._spatial_pyramid.append(bin_pool_level)

    # Grouped multi-kernel Convolution.
    self._multi_kernel_group_conv = MultiKernelGroupConvBlock(
        output_filter_depths=self._branch_filter_depths,
        kernel_sizes=self._conv_kernel_sizes,
        use_sync_bn=self._use_sync_bn,
        batchnorm_momentum=self._batchnorm_momentum,
        batchnorm_epsilon=self._batchnorm_epsilon,
        activation=self._activation,
        dropout=self._dropout,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_depthwise_convolution=self._use_depthwise_convolution)

    # Encoder's final 1x1 feature projection.
    # Considering the relatively large #channels merged before projection,
    # enlarge the projection #channels to the sum of the filter depths of
    # branches.
    self._output_channels = sum(self._branch_filter_depths)
    # Use list manually instead of tf.keras.Sequential([]).
    self._encoder_projection = [
        tf.keras.layers.Conv2D(
            filters=self._output_channels,
            kernel_size=(1, 1),
            padding='same',
            activation=None,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_bias=False),
        self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon),
    ]
    # Use the TF2 default feature alignment rule for bilinear resizing.
    self._upsample = tf.keras.layers.Resizing(
        height, width, interpolation='bilinear', crop_to_aspect_ratio=False)
    self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)
    self._concat_layer = tf.keras.layers.Concatenate(axis=self._channel_axis)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    """Calls this MOSAIC encoder block with the given input."""
    if training is None:
      training = tf.keras.backend.learning_phase()
    branches = []
    for bin_pool_level in self._spatial_pyramid:
      x = inputs
      x = bin_pool_level(x)
      x = self._multi_kernel_group_conv(x)
      x = self._upsample(x)
      branches.append(x)
    if self._global_pool_branch is not None:
      x = inputs
      for layer in self._global_pool_branch:
        x = layer(x)
      x = self._activation_fn(x)
      x = self._upsample(x)
      branches.append(x)
    x = self._concat_layer(branches)
    for layer in self._encoder_projection:
      x = layer(x)
    x = self._activation_fn(x)
    return x

  def get_config(self) -> Dict[str, Any]:
    """Returns a config dictionary for initialization from serialization."""
    config = {
        'branch_filter_depths': self._branch_filter_depths,
        'conv_kernel_sizes': self._conv_kernel_sizes,
        'pyramid_pool_bin_nums': self._pyramid_pool_bin_nums,
        'use_sync_bn': self._use_sync_bn,
        'batchnorm_momentum': self._batchnorm_momentum,
        'batchnorm_epsilon': self._batchnorm_epsilon,
        'activation': self._activation,
        'dropout': self._dropout,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'interpolation': self._interpolation,
        'use_depthwise_convolution': self._use_depthwise_convolution,
    }
    base_config = super().get_config()
    base_config.update(config)
    return base_config
