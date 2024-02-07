# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf, tf_keras

from official.modeling import tf_utils


@tf_keras.utils.register_keras_serializable(package='Vision')
class MultiKernelGroupConvBlock(tf_keras.layers.Layer):
  """A multi-kernel grouped convolution block.

  This block is used in the segmentation neck introduced in MOSAIC.
  Reference:
   [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
   Context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def __init__(
      self,
      output_filter_depths: Optional[List[int]] = None,
      kernel_sizes: Optional[List[int]] = None,
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
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
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      use_depthwise_convolution: Allows spatial pooling to be separable
        depthwise convolusions.
      **kwargs: Other keyword arguments for the layer.
    """
    super(MultiKernelGroupConvBlock, self).__init__(**kwargs)

    if output_filter_depths is None:
      output_filter_depths = [64, 64]
    if kernel_sizes is None:
      kernel_sizes = [3, 5]
    if len(output_filter_depths) != len(kernel_sizes):
      raise ValueError('The number of output groups must match #kernels.')
    self._output_filter_depths = output_filter_depths
    self._kernel_sizes = kernel_sizes
    self._num_groups = len(self._kernel_sizes)
    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._use_depthwise_convolution = use_depthwise_convolution
    # To apply BN before activation. Putting BN between conv and activation also
    # helps quantization where conv+bn+activation are fused into a single op.
    self._activation_fn = tf_utils.get_activation(activation)
    if self._use_sync_bn:
      self._bn_op = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._bn_op = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
      self._group_split_axis = -1
    else:
      self._bn_axis = 1
      self._group_split_axis = 1

  def build(self, input_shape: tf.TensorShape) -> None:
    """Builds the block with the given input shape."""
    input_channels = input_shape[self._group_split_axis]
    if input_channels % self._num_groups != 0:
      raise ValueError('The number of input channels must be divisible by '
                       'the number of groups for evenly group split.')
    self._conv_branches = []
    if self._use_depthwise_convolution:
      for i, conv_kernel_size in enumerate(self._kernel_sizes):
        depthwise_conv = tf_keras.layers.DepthwiseConv2D(
            kernel_size=(conv_kernel_size, conv_kernel_size),
            depth_multiplier=1,
            padding='same',
            depthwise_regularizer=self._kernel_regularizer,
            depthwise_initializer=self._kernel_initializer,
            use_bias=False)
        # Add BN->RELU after depthwise convolution.
        batchnorm_op_depthwise = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        activation_depthwise = self._activation_fn
        feature_conv = tf_keras.layers.Conv2D(
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
        # within a tf_keras.Sequential block, e.g. conv_branch =
        # tf_keras.Sequential([depthwise_conv, feature_conv, batchnorm_op,])
        conv_branch = [
            depthwise_conv,
            batchnorm_op_depthwise,
            activation_depthwise,
            feature_conv,
            batchnorm_op,
            ]
        self._conv_branches.append(conv_branch)
    else:
      for i, conv_kernel_size in enumerate(self._kernel_sizes):
        norm_conv = tf_keras.layers.Conv2D(
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
    self._concat_groups = tf_keras.layers.Concatenate(
        axis=self._group_split_axis)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    """Calls this group convolution block with the given inputs."""
    inputs_splits = tf.split(inputs,
                             num_or_size_splits=self._num_groups,
                             axis=self._group_split_axis)
    output_branches = []
    for i, x in enumerate(inputs_splits):
      conv_branch = self._conv_branches[i]
      # Apply layers sequentially and manually.
      for layer in conv_branch:
        if isinstance(layer, tf_keras.layers.Layer):
          x = layer(x, training=training)
        else:
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
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_depthwise_convolution': self._use_depthwise_convolution,
    }
    base_config = super(MultiKernelGroupConvBlock, self).get_config()
    base_config.update(config)
    return base_config


@tf_keras.utils.register_keras_serializable(package='Vision')
class MosaicEncoderBlock(tf_keras.layers.Layer):
  """Implements the encoder module/block of MOSAIC model.

  Spatial Pyramid Pooling and Multi-kernel Conv layer
  SpatialPyramidPoolingMultiKernelConv
  References:
    [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
    context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def __init__(
      self,
      encoder_input_level: Optional[Union[str, int]] = '4',
      branch_filter_depths: Optional[List[int]] = None,
      conv_kernel_sizes: Optional[List[int]] = None,
      pyramid_pool_bin_nums: Optional[List[int]] = None,
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout_rate: float = 0.1,
      kernel_initializer: str = 'glorot_uniform',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      use_depthwise_convolution: bool = True,
      **kwargs):
    """Initializes a MOSAIC encoder block which is deployed after a backbone.

    Args:
      encoder_input_level: An optional `str` or integer specifying the level of
        backbone outputs as the input to the encoder.
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
      dropout_rate: A float between 0 and 1. Fraction of the input units to drop
        out, which will be used directly as the `rate` of the Dropout layer at
        the end of the encoder. Defaults to 0.1.
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

    self._encoder_input_level = str(encoder_input_level)
    if branch_filter_depths is None:
      branch_filter_depths = [64, 64]
    self._branch_filter_depths = branch_filter_depths
    if conv_kernel_sizes is None:
      conv_kernel_sizes = [3, 5]
    self._conv_kernel_sizes = conv_kernel_sizes
    if pyramid_pool_bin_nums is None:
      pyramid_pool_bin_nums = [1, 4, 8, 16]
    self._pyramid_pool_bin_nums = pyramid_pool_bin_nums
    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._interpolation = interpolation
    self._use_depthwise_convolution = use_depthwise_convolution
    self._activation_fn = tf_utils.get_activation(activation)

    if self._use_sync_bn:
      self._bn_op = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._bn_op = tf_keras.layers.BatchNormalization

    self._dropout_rate = dropout_rate
    if dropout_rate:
      self._encoder_end_dropout_layer = tf_keras.layers.Dropout(
          rate=dropout_rate)
    else:
      self._encoder_end_dropout_layer = None

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
      self._channel_axis = -1
    else:
      self._bn_axis = 1
      self._channel_axis = 1

  def _get_bin_pool_kernel_and_stride(
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

  def build(
      self, input_shape: Union[tf.TensorShape, Dict[str,
                                                    tf.TensorShape]]) -> None:
    """Builds this MOSAIC encoder block with the given single input shape."""
    input_shape = (
        input_shape[self._encoder_input_level]
        if isinstance(input_shape, dict) else input_shape)
    self._data_format = tf_keras.backend.image_data_format()
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
        global_pool = tf_keras.layers.GlobalAveragePooling2D(
            data_format=self._data_format, keepdims=True)
        global_projection = tf_keras.layers.Conv2D(
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
        # Use list manually instead of tf_keras.Sequential([])
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
        pool_height, stride_height = self._get_bin_pool_kernel_and_stride(
            height, pyramid_pool_bin_num)
        pool_width, stride_width = self._get_bin_pool_kernel_and_stride(
            width, pyramid_pool_bin_num)
        bin_pool_level = tf_keras.layers.AveragePooling2D(
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
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_depthwise_convolution=self._use_depthwise_convolution)

    # Encoder's final 1x1 feature projection.
    # Considering the relatively large #channels merged before projection,
    # enlarge the projection #channels to the sum of the filter depths of
    # branches.
    self._output_channels = sum(self._branch_filter_depths)
    # Use list manually instead of tf_keras.Sequential([]).
    self._encoder_projection = [
        tf_keras.layers.Conv2D(
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
    self._upsample = tf_keras.layers.Resizing(
        height,
        width,
        interpolation=self._interpolation,
        crop_to_aspect_ratio=False)
    self._concat_layer = tf_keras.layers.Concatenate(axis=self._channel_axis)

  def call(self,
           inputs: Union[tf.Tensor, Dict[str, tf.Tensor]],
           training: Optional[bool] = None) -> tf.Tensor:
    """Calls this MOSAIC encoder block with the given input."""
    if training is None:
      training = tf_keras.backend.learning_phase()
    input_from_backbone_output = (
        inputs[self._encoder_input_level]
        if isinstance(inputs, dict) else inputs)
    branches = []
    # Original features from the final output of the backbone.
    branches.append(input_from_backbone_output)
    if self._spatial_pyramid:
      for bin_pool_level in self._spatial_pyramid:
        x = input_from_backbone_output
        x = bin_pool_level(x)
        x = self._multi_kernel_group_conv(x, training=training)
        x = self._upsample(x)
        branches.append(x)
    if self._global_pool_branch is not None:
      x = input_from_backbone_output
      for layer in self._global_pool_branch:
        x = layer(x, training=training)
      x = self._activation_fn(x)
      x = self._upsample(x)
      branches.append(x)
    x = self._concat_layer(branches)
    for layer in self._encoder_projection:
      x = layer(x, training=training)
    x = self._activation_fn(x)
    if self._encoder_end_dropout_layer is not None:
      x = self._encoder_end_dropout_layer(x, training=training)
    return x

  def get_config(self) -> Dict[str, Any]:
    """Returns a config dictionary for initialization from serialization."""
    config = {
        'encoder_input_level': self._encoder_input_level,
        'branch_filter_depths': self._branch_filter_depths,
        'conv_kernel_sizes': self._conv_kernel_sizes,
        'pyramid_pool_bin_nums': self._pyramid_pool_bin_nums,
        'use_sync_bn': self._use_sync_bn,
        'batchnorm_momentum': self._batchnorm_momentum,
        'batchnorm_epsilon': self._batchnorm_epsilon,
        'activation': self._activation,
        'dropout_rate': self._dropout_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'interpolation': self._interpolation,
        'use_depthwise_convolution': self._use_depthwise_convolution,
    }
    base_config = super().get_config()
    base_config.update(config)
    return base_config


@tf_keras.utils.register_keras_serializable(package='Vision')
class DecoderSumMergeBlock(tf_keras.layers.Layer):
  """Implements the decoder feature sum merge block of MOSAIC model.

  This block is used in the decoder of segmentation head introduced in MOSAIC.
  It essentially merges a high-resolution feature map of a low semantic level
  and a low-resolution feature map of a higher semantic level by 'Sum-Merge'.
  """

  def __init__(
      self,
      decoder_projected_depth: int,
      output_size: Tuple[int, int] = (0, 0),
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      **kwargs):
    """Initialize a sum-merge block for one decoder stage.

    Args:
      decoder_projected_depth: An integer representing the number of output
        channels of this sum-merge block in the decoder.
      output_size: A Tuple of integers representing the output height and width
        of the feature maps from this sum-merge block. Defaults to (0, 0),
        where the output size is set the same as the high-resolution branch.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for the activation function type. Defaults to 'relu'.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      **kwargs: Other keyword arguments for the layer.
    """
    super(DecoderSumMergeBlock, self).__init__(**kwargs)

    self._decoder_projected_depth = decoder_projected_depth
    self._output_size = output_size
    self._low_res_branch = []
    self._upsample_low_res = None
    self._high_res_branch = []
    self._upsample_high_res = None

    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._interpolation = interpolation
    # Apply BN before activation. Putting BN between conv and activation also
    # helps quantization where conv+bn+activation are fused into a single op.
    self._activation_fn = tf_utils.get_activation(activation)
    if self._use_sync_bn:
      self._bn_op = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._bn_op = tf_keras.layers.BatchNormalization

    self._bn_axis = (
        -1
        if tf_keras.backend.image_data_format() == 'channels_last' else 1)
    self._channel_axis = (
        -1
        if tf_keras.backend.image_data_format() == 'channels_last' else 1)
    self._add_layer = tf_keras.layers.Add()

  def build(
      self,
      input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
    """Builds the block with the given input shape."""
    # Assume backbone features of the same level are concated before input.
    low_res_input_shape = input_shape[0]
    high_res_input_shape = input_shape[1]
    low_res_channels = low_res_input_shape[self._channel_axis]
    high_res_channels = high_res_input_shape[self._channel_axis]

    if low_res_channels != self._decoder_projected_depth:
      low_res_feature_conv = tf_keras.layers.Conv2D(
          filters=self._decoder_projected_depth,
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
      self._low_res_branch.extend([
          low_res_feature_conv,
          batchnorm_op,
      ])
    if high_res_channels != self._decoder_projected_depth:
      high_res_feature_conv = tf_keras.layers.Conv2D(
          filters=self._decoder_projected_depth,
          kernel_size=(1, 1),
          padding='same',
          kernel_regularizer=self._kernel_regularizer,
          kernel_initializer=self._kernel_initializer,
          activation=None,
          use_bias=False)
      batchnorm_op_high = self._bn_op(
          axis=self._bn_axis,
          momentum=self._batchnorm_momentum,
          epsilon=self._batchnorm_epsilon)
      self._high_res_branch.extend([
          high_res_feature_conv,
          batchnorm_op_high,
      ])
    # Resize feature maps.
    if tf_keras.backend.image_data_format() == 'channels_last':
      low_res_height = low_res_input_shape[1]
      low_res_width = low_res_input_shape[2]
      high_res_height = high_res_input_shape[1]
      high_res_width = high_res_input_shape[2]
    else:
      low_res_height = low_res_input_shape[2]
      low_res_width = low_res_input_shape[3]
      high_res_height = high_res_input_shape[2]
      high_res_width = high_res_input_shape[3]
    if (self._output_size[0] == 0 or self._output_size[1] == 0):
      self._output_size = (high_res_height, high_res_width)
    if (low_res_height != self._output_size[0] or
        low_res_width != self._output_size[1]):
      self._upsample_low_res = tf_keras.layers.Resizing(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    if (high_res_height != self._output_size[0] or
        high_res_width != self._output_size[1]):
      self._upsample_high_res = tf_keras.layers.Resizing(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)

  def call(self,
           inputs: Tuple[tf.Tensor, tf.Tensor],
           training: Optional[bool] = None) -> tf.Tensor:
    """Calls this decoder sum-merge block with the given input.

    Args:
      inputs: A Tuple of tensors consisting of a low-resolution higher-semantic
        level feature map from the encoder as the first item and a higher
        resolution lower-level feature map from the backbone as the second item.
      training: a `bool` indicating whether it is in `training` mode.
    Note: the first item of the input Tuple takes a lower-resolution feature map
    and the second item of the input Tuple takes a higher-resolution branch.

    Returns:
      A tensor representing the sum-merged decoder feature map.
    """
    if training is None:
      training = tf_keras.backend.learning_phase()
    x_low_res = inputs[0]
    x_high_res = inputs[1]
    if self._low_res_branch:
      for layer in self._low_res_branch:
        x_low_res = layer(x_low_res, training=training)
      x_low_res = self._activation_fn(x_low_res)
    if self._high_res_branch:
      for layer in self._high_res_branch:
        x_high_res = layer(x_high_res, training=training)
      x_high_res = self._activation_fn(x_high_res)
    if self._upsample_low_res is not None:
      x_low_res = self._upsample_low_res(x_low_res)
    if self._upsample_high_res is not None:
      x_high_res = self._upsample_high_res(x_high_res)
    output = self._add_layer([x_low_res, x_high_res])
    return output

  def get_config(self) -> Dict[str, Any]:
    """Returns a config dictionary for initialization from serialization."""
    config = {
        'decoder_projected_depth': self._decoder_projected_depth,
        'output_size': self._output_size,
        'use_sync_bn': self._use_sync_bn,
        'batchnorm_momentum': self._batchnorm_momentum,
        'batchnorm_epsilon': self._batchnorm_epsilon,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'interpolation': self._interpolation,
    }
    base_config = super(DecoderSumMergeBlock, self).get_config()
    base_config.update(config)
    return base_config


@tf_keras.utils.register_keras_serializable(package='Vision')
class DecoderConcatMergeBlock(tf_keras.layers.Layer):
  """Implements the decoder feature concat merge block of MOSAIC model.

  This block is used in the decoder of segmentation head introduced in MOSAIC.
  It essentially merges a high-resolution feature map of a low semantic level
  and a low-resolution feature of a higher semantic level by 'Concat-Merge'.
  """

  def __init__(
      self,
      decoder_internal_depth: int,
      decoder_projected_depth: int,
      output_size: Tuple[int, int] = (0, 0),
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      **kwargs):
    """Initializes a concat-merge block for one decoder stage.

    Args:
      decoder_internal_depth: An integer representing the number of internal
        channels of this concat-merge block in the decoder.
      decoder_projected_depth: An integer representing the number of output
        channels of this concat-merge block in the decoder.
      output_size: A Tuple of integers representing the output height and width
        of the feature maps from this concat-merge block. Defaults to (0, 0),
        where the output size is set the same as the high-resolution branch.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for the activation function type. Defaults to 'relu'.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      **kwargs: Other keyword arguments for the layer.
    """
    super(DecoderConcatMergeBlock, self).__init__(**kwargs)

    self._decoder_internal_depth = decoder_internal_depth
    self._decoder_projected_depth = decoder_projected_depth
    self._output_size = output_size
    self._upsample_low_res = None
    self._upsample_high_res = None

    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._interpolation = interpolation
    # Apply BN before activation. Putting BN between conv and activation also
    # helps quantization where conv+bn+activation are fused into a single op.
    self._activation_fn = tf_utils.get_activation(activation)
    if self._use_sync_bn:
      self._bn_op = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._bn_op = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
      self._channel_axis = -1
    else:
      self._bn_axis = 1
      self._channel_axis = 1

  def build(
      self,
      input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
    """Builds this block with the given input shape."""
    # Assume backbone features of the same level are concated before input.
    low_res_input_shape = input_shape[0]
    high_res_input_shape = input_shape[1]
    # Set up resizing feature maps before concat.
    if tf_keras.backend.image_data_format() == 'channels_last':
      low_res_height = low_res_input_shape[1]
      low_res_width = low_res_input_shape[2]
      high_res_height = high_res_input_shape[1]
      high_res_width = high_res_input_shape[2]
    else:
      low_res_height = low_res_input_shape[2]
      low_res_width = low_res_input_shape[3]
      high_res_height = high_res_input_shape[2]
      high_res_width = high_res_input_shape[3]
    if (self._output_size[0] == 0 or self._output_size[1] == 0):
      self._output_size = (high_res_height, high_res_width)
    if (low_res_height != self._output_size[0] or
        low_res_width != self._output_size[1]):
      self._upsample_low_res = tf_keras.layers.Resizing(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    if (high_res_height != self._output_size[0] or
        high_res_width != self._output_size[1]):
      self._upsample_high_res = tf_keras.layers.Resizing(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    # Set up a 3-layer separable convolution blocks, i.e.
    # 1x1->BN->RELU + Depthwise->BN->RELU + 1x1->BN->RELU.
    initial_feature_conv = tf_keras.layers.Conv2D(
        filters=self._decoder_internal_depth,
        kernel_size=(1, 1),
        padding='same',
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        activation=None,
        use_bias=False)
    batchnorm_op1 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)
    activation1 = self._activation_fn
    depthwise_conv = tf_keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depth_multiplier=1,
        padding='same',
        depthwise_regularizer=self._kernel_regularizer,
        depthwise_initializer=self._kernel_initializer,
        use_bias=False)
    batchnorm_op2 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)
    activation2 = self._activation_fn
    project_feature_conv = tf_keras.layers.Conv2D(
        filters=self._decoder_projected_depth,
        kernel_size=(1, 1),
        padding='same',
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        activation=None,
        use_bias=False)
    batchnorm_op3 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)
    activation3 = self._activation_fn
    self._feature_fusion_block = [
        initial_feature_conv,
        batchnorm_op1,
        activation1,
        depthwise_conv,
        batchnorm_op2,
        activation2,
        project_feature_conv,
        batchnorm_op3,
        activation3,
        ]
    self._concat_layer = tf_keras.layers.Concatenate(axis=self._channel_axis)

  def call(self,
           inputs: Tuple[tf.Tensor, tf.Tensor],
           training: Optional[bool] = None) -> tf.Tensor:
    """Calls this concat-merge block with the given inputs.

    Args:
      inputs: A Tuple of tensors consisting of a lower-level higher-resolution
        feature map from the backbone as the first item and a higher-level
        lower-resolution feature map from the encoder as the second item.
      training: a `Boolean` indicating whether it is in `training` mode.

    Returns:
      A tensor representing the concat-merged decoder feature map.
    """
    low_res_input = inputs[0]
    high_res_input = inputs[1]
    if self._upsample_low_res is not None:
      low_res_input = self._upsample_low_res(low_res_input)
    if self._upsample_high_res is not None:
      high_res_input = self._upsample_high_res(high_res_input)
    decoder_feature_list = [low_res_input, high_res_input]
    x = self._concat_layer(decoder_feature_list)
    for layer in self._feature_fusion_block:
      if isinstance(layer, tf_keras.layers.Layer):
        x = layer(x, training=training)
      else:
        x = layer(x)
    return x

  def get_config(self) -> Dict[str, Any]:
    """Returns a config dictionary for initialization from serialization."""
    config = {
        'decoder_internal_depth': self._decoder_internal_depth,
        'decoder_projected_depth': self._decoder_projected_depth,
        'output_size': self._output_size,
        'use_sync_bn': self._use_sync_bn,
        'batchnorm_momentum': self._batchnorm_momentum,
        'batchnorm_epsilon': self._batchnorm_epsilon,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'interpolation': self._interpolation,
    }
    base_config = super(DecoderConcatMergeBlock, self).get_config()
    base_config.update(config)
    return base_config
