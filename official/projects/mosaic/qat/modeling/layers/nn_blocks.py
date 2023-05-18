# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Contains quantized neural blocks for the QAT."""

from typing import Dict, Tuple, Union

import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.modeling import tf_utils
from official.projects.mosaic.modeling import mosaic_blocks
from official.projects.qat.vision.quantization import configs
from official.projects.qat.vision.quantization import helper


@tf.keras.utils.register_keras_serializable(package='Vision')
class MultiKernelGroupConvBlockQuantized(mosaic_blocks.MultiKernelGroupConvBlock
                                        ):
  """A quantized multi-kernel grouped convolution block.

  This block is used in the segmentation neck introduced in MOSAIC.
  Reference:
   [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
   Context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def build(self, input_shape: tf.TensorShape) -> None:
    """Builds the block with the given input shape."""
    input_channels = input_shape[self._group_split_axis]
    if input_channels % self._num_groups != 0:
      raise ValueError('The number of input channels must be divisible by '
                       'the number of groups for evenly group split.')

    # Override the activation and bn with their quantized version.
    self._activation_fn = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    norm_layer = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._use_sync_bn else tf.keras.layers.BatchNormalization)
    norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    norm_no_quantize = helper.BatchNormalizationNoQuantized(norm_layer)
    self._bn_op = helper.norm_by_activation(
        self._activation, norm_with_quantize, norm_no_quantize)

    self._conv_branches = []
    if self._use_depthwise_convolution:
      for i, conv_kernel_size in enumerate(self._kernel_sizes):
        depthwise_conv = helper.DepthwiseConv2DQuantized(
            kernel_size=(conv_kernel_size, conv_kernel_size),
            depth_multiplier=1,
            padding='same',
            depthwise_regularizer=self._kernel_regularizer,
            depthwise_initializer=self._kernel_initializer,
            use_bias=False,
            activation=helper.NoOpActivation())
        # Add BN->RELU after depthwise convolution.
        batchnorm_op_depthwise = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        activation_depthwise = self._activation_fn
        feature_conv = helper.Conv2DQuantized(
            filters=self._output_filter_depths[i],
            kernel_size=(1, 1),
            padding='same',
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._kernel_initializer,
            activation=helper.NoOpActivation(),
            use_bias=False)
        batchnorm_op = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        # Use list manually as current QAT API does not support sequential model
        # within a tf.keras.Sequential block, e.g. conv_branch =
        # tf.keras.Sequential([depthwise_conv, feature_conv, batchnorm_op,])
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
        norm_conv = helper.Conv2DQuantized(
            filters=self._output_filter_depths[i],
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            activation=helper.NoOpActivation(),
            use_bias=False)
        batchnorm_op = self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
        conv_branch = [norm_conv, batchnorm_op]
        self._conv_branches.append(conv_branch)

    self._concat_groups = helper.ConcatenateQuantized(
        axis=self._group_split_axis)


@tf.keras.utils.register_keras_serializable(package='Vision')
class MosaicEncoderBlockQuantized(mosaic_blocks.MosaicEncoderBlock):
  """Implements the encoder module/block of MOSAIC model.

  Spatial Pyramid Pooling and Multi-kernel Conv layer
  SpatialPyramidPoolingMultiKernelConv
  References:
    [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded
    context](https://arxiv.org/pdf/2112.11623.pdf)
  """

  def build(
      self, input_shape: Union[tf.TensorShape, Dict[str,
                                                    tf.TensorShape]]) -> None:
    """Builds this MOSAIC encoder block with the given single input shape."""
    input_shape = (
        input_shape[self._encoder_input_level]
        if isinstance(input_shape, dict) else input_shape)
    self._data_format = tf.keras.backend.image_data_format()
    if self._data_format == 'channels_last':
      height = input_shape[1]
      width = input_shape[2]
    else:
      height = input_shape[2]
      width = input_shape[3]

    self._global_pool_branch = None
    self._spatial_pyramid = []

    # Override the activation and bn with their quantized version.
    self._activation_fn = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    norm_layer = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._use_sync_bn else tf.keras.layers.BatchNormalization)
    norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    norm_no_quantize = helper.BatchNormalizationNoQuantized(norm_layer)
    self._bn_op = helper.norm_by_activation(
        self._activation, norm_with_quantize, norm_no_quantize)

    for pyramid_pool_bin_num in self._pyramid_pool_bin_nums:
      if pyramid_pool_bin_num == 1:
        global_pool = helper.GlobalAveragePooling2DQuantized(
            data_format=self._data_format, keepdims=True)

        global_projection = helper.Conv2DQuantized(
            filters=max(self._branch_filter_depths),
            kernel_size=(1, 1),
            padding='same',
            activation=helper.NoOpActivation(),
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
        pool_height, stride_height = self._get_bin_pool_kernel_and_stride(
            height, pyramid_pool_bin_num)
        pool_width, stride_width = self._get_bin_pool_kernel_and_stride(
            width, pyramid_pool_bin_num)
        bin_pool_level = helper.AveragePooling2DQuantized(
            pool_size=(pool_height, pool_width),
            strides=(stride_height, stride_width),
            padding='valid',
            data_format=self._data_format)
        self._spatial_pyramid.append(bin_pool_level)

    # Grouped multi-kernel Convolution.
    self._multi_kernel_group_conv = MultiKernelGroupConvBlockQuantized(
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
    # Use list manually instead of tf.keras.Sequential([]).
    self._encoder_projection = [
        helper.Conv2DQuantized(
            filters=self._output_channels,
            kernel_size=(1, 1),
            padding='same',
            activation=helper.NoOpActivation(),
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_bias=False),
        self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon),
    ]
    # Use the TF2 default feature alignment rule for bilinear resizing.
    self._upsample = helper.ResizingQuantized(
        height,
        width,
        interpolation=self._interpolation,
        crop_to_aspect_ratio=False)
    self._concat_layer = helper.ConcatenateQuantized(axis=self._channel_axis)


@tf.keras.utils.register_keras_serializable(package='Vision')
class DecoderSumMergeBlockQuantized(mosaic_blocks.DecoderSumMergeBlock):
  """Implements the decoder feature sum merge block of MOSAIC model.

  This block is used in the decoder of segmentation head introduced in MOSAIC.
  It essentially merges a high-resolution feature map of a low semantic level
  and a low-resolution feature map of a higher semantic level by 'Sum-Merge'.
  """

  def build(
      self,
      input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
    """Builds the block with the given input shape."""
    # Assume backbone features of the same level are concated before input.
    low_res_input_shape = input_shape[0]
    high_res_input_shape = input_shape[1]
    low_res_channels = low_res_input_shape[self._channel_axis]
    high_res_channels = high_res_input_shape[self._channel_axis]

    # Override the activation and bn with their quantized version.
    self._activation_fn = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    norm_layer = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._use_sync_bn else tf.keras.layers.BatchNormalization)
    norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    norm_no_quantize = helper.BatchNormalizationNoQuantized(norm_layer)
    self._bn_op = helper.norm_by_activation(
        self._activation, norm_with_quantize, norm_no_quantize)

    if low_res_channels != self._decoder_projected_depth:
      low_res_feature_conv = helper.Conv2DQuantized(
          filters=self._decoder_projected_depth,
          kernel_size=(1, 1),
          padding='same',
          kernel_regularizer=self._kernel_regularizer,
          kernel_initializer=self._kernel_initializer,
          activation=helper.NoOpActivation(),
          use_bias=False)
      batchnorm_op = self._bn_op(
          axis=self._bn_axis,
          momentum=self._batchnorm_momentum,
          epsilon=self._batchnorm_epsilon)
      self._low_res_branch = [
          low_res_feature_conv,
          batchnorm_op,
      ]
    if high_res_channels != self._decoder_projected_depth:
      high_res_feature_conv = helper.Conv2DQuantized(
          filters=self._decoder_projected_depth,
          kernel_size=(1, 1),
          padding='same',
          kernel_regularizer=self._kernel_regularizer,
          kernel_initializer=self._kernel_initializer,
          activation=helper.NoOpActivation(),
          use_bias=False)
      batchnorm_op_high = self._bn_op(
          axis=self._bn_axis,
          momentum=self._batchnorm_momentum,
          epsilon=self._batchnorm_epsilon)
      self._high_res_branch = [
          high_res_feature_conv,
          batchnorm_op_high,
      ]
    # Resize feature maps.
    if tf.keras.backend.image_data_format() == 'channels_last':
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
      self._upsample_low_res = helper.ResizingQuantized(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    if (high_res_height != self._output_size[0] or
        high_res_width != self._output_size[1]):
      self._upsample_high_res = helper.ResizingQuantized(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    self._add_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf.keras.layers.Add(), configs.Default8BitQuantizeConfig([], [], True))


@tf.keras.utils.register_keras_serializable(package='Vision')
class DecoderConcatMergeBlockQuantized(mosaic_blocks.DecoderConcatMergeBlock):
  """Implements the decoder feature concat merge block of MOSAIC model.

  This block is used in the decoder of segmentation head introduced in MOSAIC.
  It essentially merges a high-resolution feature map of a low semantic level
  and a low-resolution feature of a higher semantic level by 'Concat-Merge'.
  """

  def build(
      self,
      input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
    """Builds this block with the given input shape."""
    # Assume backbone features of the same level are concated before input.
    low_res_input_shape = input_shape[0]
    high_res_input_shape = input_shape[1]
    # Set up resizing feature maps before concat.
    if tf.keras.backend.image_data_format() == 'channels_last':
      low_res_height = low_res_input_shape[1]
      low_res_width = low_res_input_shape[2]
      high_res_height = high_res_input_shape[1]
      high_res_width = high_res_input_shape[2]
    else:
      low_res_height = low_res_input_shape[2]
      low_res_width = low_res_input_shape[3]
      high_res_height = high_res_input_shape[2]
      high_res_width = high_res_input_shape[3]

    self._concat_layer = helper.ConcatenateQuantized(axis=self._channel_axis)

    # Override the activation and bn with their quantized version.
    self._activation_fn = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    norm_layer = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._use_sync_bn else tf.keras.layers.BatchNormalization)
    norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    norm_no_quantize = helper.BatchNormalizationNoQuantized(norm_layer)
    self._bn_op = helper.norm_by_activation(
        self._activation, norm_with_quantize, norm_no_quantize)

    if (self._output_size[0] == 0 or self._output_size[1] == 0):
      self._output_size = (high_res_height, high_res_width)
    if (low_res_height != self._output_size[0] or
        low_res_width != self._output_size[1]):
      self._upsample_low_res = helper.ResizingQuantized(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    if (high_res_height != self._output_size[0] or
        high_res_width != self._output_size[1]):
      self._upsample_high_res = helper.ResizingQuantized(
          self._output_size[0],
          self._output_size[1],
          interpolation=self._interpolation,
          crop_to_aspect_ratio=False)
    # Set up a 3-layer separable convolution blocks, i.e.
    # 1x1->BN->RELU + Depthwise->BN->RELU + 1x1->BN->RELU.
    initial_feature_conv = helper.Conv2DQuantized(
        filters=self._decoder_internal_depth,
        kernel_size=(1, 1),
        padding='same',
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        activation=helper.NoOpActivation(),
        use_bias=False)
    batchnorm_op1 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)
    activation1 = self._activation_fn
    depthwise_conv = helper.DepthwiseConv2DQuantized(
        kernel_size=(3, 3),
        depth_multiplier=1,
        padding='same',
        depthwise_regularizer=self._kernel_regularizer,
        depthwise_initializer=self._kernel_initializer,
        use_bias=False,
        activation=helper.NoOpActivation())
    batchnorm_op2 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)
    activation2 = self._activation_fn
    project_feature_conv = helper.Conv2DQuantized(
        filters=self._decoder_projected_depth,
        kernel_size=(1, 1),
        padding='same',
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        activation=helper.NoOpActivation(),
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
    self._concat_layer = helper.ConcatenateQuantized(axis=self._channel_axis)
