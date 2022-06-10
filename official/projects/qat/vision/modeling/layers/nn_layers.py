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

"""Contains common building blocks for neural networks."""

import enum
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Any

import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.modeling import tf_utils
from official.projects.qat.vision.quantization import configs
from official.projects.qat.vision.quantization import helper
from official.vision.modeling.decoders import aspp
from official.vision.modeling.layers import nn_layers


# Type annotations.
States = Dict[str, tf.Tensor]
Activation = Union[str, Callable]


# String constants.
class FeatureFusion(str, enum.Enum):
  PYRAMID_FUSION = 'pyramid_fusion'
  PANOPTIC_FPN_FUSION = 'panoptic_fpn_fusion'
  DEEPLABV3PLUS = 'deeplabv3plus'
  DEEPLABV3PLUS_SUM_TO_MERGE = 'deeplabv3plus_sum_to_merge'


@tf.keras.utils.register_keras_serializable(package='Vision')
class SqueezeExcitationQuantized(
    helper.LayerQuantizerHelper,
    tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               out_filters,
               se_ratio,
               divisible_by=1,
               use_3d_input=False,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               gating_activation='sigmoid',
               round_down_protect=True,
               **kwargs):
    """Initializes a squeeze and excitation layer.

    Args:
      in_filters: An `int` number of filters of the input tensor.
      out_filters: An `int` number of filters of the output tensor.
      se_ratio: A `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      use_3d_input: A `bool` of whether input is 2D or 3D image.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      gating_activation: A `str` name of the activation function for final
        gating function.
      round_down_protect: A `bool` of whether round down more than 10% will be
        allowed.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._round_down_protect = round_down_protect
    self._use_3d_input = use_3d_input
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      if not use_3d_input:
        self._spatial_axis = [1, 2]
      else:
        self._spatial_axis = [1, 2, 3]
    else:
      if not use_3d_input:
        self._spatial_axis = [2, 3]
      else:
        self._spatial_axis = [2, 3, 4]

  def _create_gating_activation_layer(self):
    if self._gating_activation == 'hard_sigmoid':
      # Convert hard_sigmoid activation to quantizable keras layers so each op
      # can be properly quantized.
      # Formula is hard_sigmoid(x) = relu6(x + 3) * 0.16667.
      self._add_quantizer('add_three')
      self._add_quantizer('divide_six')
      self._relu6 = tfmot.quantization.keras.QuantizeWrapperV2(
          tf_utils.get_activation('relu6', use_keras_layer=True),
          configs.Default8BitActivationQuantizeConfig())
    else:
      self._gating_activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
          tf_utils.get_activation(
              self._gating_activation, use_keras_layer=True),
          configs.Default8BitActivationQuantizeConfig())

  def _apply_gating_activation_layer(
      self, x: tf.Tensor, training: bool) -> tf.Tensor:
    if self._gating_activation == 'hard_sigmoid':
      x = self._apply_quantizer('add_three', x + 3.0, training)
      x = self._relu6(x)
      x = self._apply_quantizer('divide_six', x * 1.6667, training)
    else:
      x = self._gating_activation_layer(x)
    return x

  def build(self, input_shape):
    num_reduced_filters = nn_layers.make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by,
        round_down_protect=self._round_down_protect)

    self._se_reduce = helper.Conv2DQuantized(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())

    self._se_expand = helper.Conv2DOutputQuantized(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=helper.NoOpActivation())

    self._multiply = tfmot.quantization.keras.QuantizeWrapperV2(
        tf.keras.layers.Multiply(),
        configs.Default8BitQuantizeConfig([], [], True))
    self._reduce_mean_quantizer = (
        tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))
    self._reduce_mean_quantizer_vars = self._reduce_mean_quantizer.build(
        None, 'reduce_mean_quantizer_vars', self)

    self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(self._activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    self._create_gating_activation_layer()

    self._build_quantizer_vars()
    super().build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'se_ratio': self._se_ratio,
        'divisible_by': self._divisible_by,
        'use_3d_input': self._use_3d_input,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'gating_activation': self._gating_activation,
        'round_down_protect': self._round_down_protect,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._reduce_mean_quantizer(
        x, training, self._reduce_mean_quantizer_vars)
    x = self._activation_layer(self._se_reduce(x))
    x = self._apply_gating_activation_layer(self._se_expand(x), training)
    x = self._multiply([x, inputs])
    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class SegmentationHeadQuantized(tf.keras.layers.Layer):
  """Creates a segmentation head."""

  def __init__(
      self,
      num_classes: int,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      use_depthwise_convolution: bool = False,
      prediction_kernel_size: int = 1,
      upsample_factor: int = 1,
      feature_fusion: Optional[str] = None,
      decoder_min_level: Optional[int] = None,
      decoder_max_level: Optional[int] = None,
      low_level: int = 2,
      low_level_num_filters: int = 48,
      num_decoder_filters: int = 256,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a segmentation head.

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      level: An `int` or `str`, level to use to build segmentation head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      prediction_kernel_size: An `int` number to specify the kernel size of the
        prediction layer.
      upsample_factor: An `int` number to specify the upsampling factor to
        generate finer mask. Default 1 means no upsampling is applied.
      feature_fusion: One of `deeplabv3plus`, `deeplabv3plus_sum_to_merge`,
        `pyramid_fusion`, or None. If  `deeplabv3plus`, features from
        decoder_features[level] will be fused with low level feature maps from
        backbone. If `pyramid_fusion`, multiscale features will be resized and
        fused at the target level.
      decoder_min_level: An `int` of minimum level from decoder to use in
        feature fusion. It is only used when feature_fusion is set to
        `panoptic_fpn_fusion`.
      decoder_max_level: An `int` of maximum level from decoder to use in
        feature fusion. It is only used when feature_fusion is set to
        `panoptic_fpn_fusion`.
      low_level: An `int` of backbone level to be used for feature fusion. It is
        used when feature_fusion is set to `deeplabv3plus`.
      low_level_num_filters: An `int` of reduced number of filters for the low
        level features before fusing it with higher level features. It is only
        used when feature_fusion is set to `deeplabv3plus`.
      num_decoder_filters: An `int` of number of filters in the decoder outputs.
        It is only used when feature_fusion is set to `panoptic_fpn_fusion`.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'level': level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_depthwise_convolution': use_depthwise_convolution,
        'prediction_kernel_size': prediction_kernel_size,
        'upsample_factor': upsample_factor,
        'feature_fusion': feature_fusion,
        'decoder_min_level': decoder_min_level,
        'decoder_max_level': decoder_max_level,
        'low_level': low_level,
        'low_level_num_filters': low_level_num_filters,
        'num_decoder_filters': num_decoder_filters,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())

  def build(self, input_shape: Sequence[tf.TensorShape]):
    """Creates the variables of the segmentation head."""
    # When input_shape is a list/tuple, the first corresponds to backbone
    # features used for resizing the decoder features (the second) if feature
    # fusion type is `deeplabv3plus`.
    backbone_shape = input_shape[0]
    use_depthwise_convolution = self._config_dict['use_depthwise_convolution']
    random_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    conv_kwargs = {
        'kernel_size': 3 if not use_depthwise_convolution else 1,
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': random_initializer,
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
    }

    norm_layer = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._config_dict['use_sync_bn'] else
        tf.keras.layers.BatchNormalization)
    norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    norm_no_quantize = helper.BatchNormalizationNoQuantized(norm_layer)
    norm = helper.norm_by_activation(self._config_dict['activation'],
                                     norm_with_quantize, norm_no_quantize)

    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    if self._config_dict['feature_fusion'] in [
        FeatureFusion.DEEPLABV3PLUS, FeatureFusion.DEEPLABV3PLUS_SUM_TO_MERGE
    ]:
      # Deeplabv3+ feature fusion layers.
      self._dlv3p_conv = helper.Conv2DQuantized(
          kernel_size=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf_utils.clone_initializer(random_initializer),
          kernel_regularizer=self._config_dict['kernel_regularizer'],
          name='segmentation_head_deeplabv3p_fusion_conv',
          filters=self._config_dict['low_level_num_filters'],
          activation=helper.NoOpActivation())

      self._dlv3p_norm = norm(
          name='segmentation_head_deeplabv3p_fusion_norm', **bn_kwargs)

    # Segmentation head layers.
    self._convs = []
    self._norms = []
    for i in range(self._config_dict['num_convs']):
      if use_depthwise_convolution:
        self._convs.append(
            helper.DepthwiseConv2DQuantized(
                name='segmentation_head_depthwise_conv_{}'.format(i),
                kernel_size=3,
                padding='same',
                use_bias=False,
                depthwise_initializer=tf_utils.clone_initializer(
                    random_initializer),
                depthwise_regularizer=self._config_dict['kernel_regularizer'],
                depth_multiplier=1,
                activation=helper.NoOpActivation()))
        norm_name = 'segmentation_head_depthwise_norm_{}'.format(i)
        self._norms.append(norm(name=norm_name, **bn_kwargs))
      conv_name = 'segmentation_head_conv_{}'.format(i)
      self._convs.append(
          helper.Conv2DQuantized(
              name=conv_name,
              filters=self._config_dict['num_filters'],
              activation=helper.NoOpActivation(),
              **conv_kwargs))
      norm_name = 'segmentation_head_norm_{}'.format(i)
      self._norms.append(norm(name=norm_name, **bn_kwargs))

    self._classifier = helper.Conv2DOutputQuantized(
        name='segmentation_output',
        filters=self._config_dict['num_classes'],
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf_utils.clone_initializer(random_initializer),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        activation=helper.NoOpActivation())

    self._upsampling_layer = helper.UpSampling2DQuantized(
        size=(self._config_dict['upsample_factor'],
              self._config_dict['upsample_factor']),
        interpolation='nearest')
    self._resizing_layer = helper.ResizingQuantized(
        backbone_shape[1], backbone_shape[2], interpolation='bilinear')

    self._concat_layer = helper.ConcatenateQuantized(axis=self._bn_axis)
    self._add_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf.keras.layers.Add(), configs.Default8BitQuantizeConfig([], [], True))

    super().build(input_shape)

  def call(self, inputs: Tuple[Union[tf.Tensor, Mapping[str, tf.Tensor]],
                               Union[tf.Tensor, Mapping[str, tf.Tensor]]]):
    """Forward pass of the segmentation head.

    It supports both a tuple of 2 tensors or 2 dictionaries. The first is
    backbone endpoints, and the second is decoder endpoints. When inputs are
    tensors, they are from a single level of feature maps. When inputs are
    dictionaries, they contain multiple levels of feature maps, where the key
    is the index of feature map.

    Args:
      inputs: A tuple of 2 feature map tensors of shape
        [batch, height_l, width_l, channels] or 2 dictionaries of tensors:
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
          [batch, height_l, width_l, channels].

    Returns:
      segmentation prediction mask: A `tf.Tensor` of the segmentation mask
        scores predicted from input features.
    """
    if self._config_dict['feature_fusion'] in (
        FeatureFusion.PYRAMID_FUSION, FeatureFusion.PANOPTIC_FPN_FUSION):
      raise ValueError(
          'The feature fusion method `pyramid_fusion` is not supported in QAT.')

    backbone_output = inputs[0]
    decoder_output = inputs[1]
    if self._config_dict['feature_fusion'] in {
        FeatureFusion.DEEPLABV3PLUS, FeatureFusion.DEEPLABV3PLUS_SUM_TO_MERGE
    }:
      # deeplabv3+ feature fusion.
      x = decoder_output[str(self._config_dict['level'])] if isinstance(
          decoder_output, dict) else decoder_output
      y = backbone_output[str(self._config_dict['low_level'])] if isinstance(
          backbone_output, dict) else backbone_output
      y = self._dlv3p_norm(self._dlv3p_conv(y))
      y = self._activation_layer(y)
      x = self._resizing_layer(x)
      x = tf.cast(x, dtype=y.dtype)
      if self._config_dict['feature_fusion'] == FeatureFusion.DEEPLABV3PLUS:
        x = self._concat_layer([x, y])
      else:
        x = self._add_layer([x, y])
    else:
      x = decoder_output[str(self._config_dict['level'])] if isinstance(
          decoder_output, dict) else decoder_output

    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x)
      x = self._activation_layer(x)
    if self._config_dict['upsample_factor'] > 1:
      # Use keras layer for nearest upsampling so it is QAT compatible.
      x = self._upsampling_layer(x)

    return self._classifier(x)

  def get_config(self):
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(self._config_dict.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Vision')
class SpatialPyramidPoolingQuantized(nn_layers.SpatialPyramidPooling):
  """Implements the quantized Atrous Spatial Pyramid Pooling.

  References:
    [Rethinking Atrous Convolution for Semantic Image Segmentation](
      https://arxiv.org/pdf/1706.05587.pdf)
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
  """

  def __init__(
      self,
      output_channels: int,
      dilation_rates: List[int],
      pool_kernel_size: Optional[List[int]] = None,
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout: float = 0.5,
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      use_depthwise_convolution: bool = False,
      **kwargs):
    """Initializes `SpatialPyramidPooling`.

    Args:
      output_channels: Number of channels produced by SpatialPyramidPooling.
      dilation_rates: A list of integers for parallel dilated conv.
      pool_kernel_size: A list of integers or None. If None, global average
        pooling is applied, otherwise an average pooling of pool_kernel_size is
        applied.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for type of activation to be used. Defaults to 'relu'.
      dropout: A float for the dropout rate before output. Defaults to 0.5.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      use_depthwise_convolution: Allows spatial pooling to be separable
        depthwise convolusions. [Encoder-Decoder with Atrous Separable
        Convolution for Semantic Image Segmentation](
         https://arxiv.org/pdf/1802.02611.pdf)
      **kwargs: Other keyword arguments for the layer.
    """
    super().__init__(
        output_channels=output_channels,
        dilation_rates=dilation_rates,
        use_sync_bn=use_sync_bn,
        batchnorm_momentum=batchnorm_momentum,
        batchnorm_epsilon=batchnorm_epsilon,
        activation=activation,
        dropout=dropout,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        interpolation=interpolation,
        pool_kernel_size=pool_kernel_size,
        use_depthwise_convolution=use_depthwise_convolution)

    self._activation_fn = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(activation, use_keras_layer=True),
        configs.Default8BitActivationQuantizeConfig())
    self._activation_fn_no_quant = (
        tf_utils.get_activation(activation, use_keras_layer=True))

  def build(self, input_shape):
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    norm_layer = (
        tf.keras.layers.experimental.SyncBatchNormalization
        if self._use_sync_bn else tf.keras.layers.BatchNormalization)
    norm_with_quantize = helper.BatchNormalizationQuantized(norm_layer)
    norm_no_quantize = helper.BatchNormalizationNoQuantized(norm_layer)
    norm = helper.norm_by_activation(self._activation, norm_with_quantize,
                                     norm_no_quantize)

    self.aspp_layers = []

    conv1 = helper.Conv2DQuantized(
        filters=self._output_channels,
        kernel_size=(1, 1),
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        use_bias=False,
        activation=helper.NoOpActivation())
    norm1 = norm(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)

    self.aspp_layers.append([conv1, norm1])

    for dilation_rate in self._dilation_rates:
      leading_layers = []
      kernel_size = (3, 3)
      if self._use_depthwise_convolution:
        leading_layers += [
            helper.DepthwiseConv2DOutputQuantized(
                depth_multiplier=1,
                kernel_size=kernel_size,
                padding='same',
                depthwise_regularizer=self._kernel_regularizer,
                depthwise_initializer=tf_utils.clone_initializer(
                    self._kernel_initializer),
                dilation_rate=dilation_rate,
                use_bias=False,
                activation=helper.NoOpActivation())
        ]
        kernel_size = (1, 1)
      conv_dilation = leading_layers + [
          helper.Conv2DQuantized(
              filters=self._output_channels,
              kernel_size=kernel_size,
              padding='same',
              kernel_regularizer=self._kernel_regularizer,
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
              dilation_rate=dilation_rate,
              use_bias=False,
              activation=helper.NoOpActivation())
      ]
      norm_dilation = norm(
          axis=self._bn_axis,
          momentum=self._batchnorm_momentum,
          epsilon=self._batchnorm_epsilon)

      self.aspp_layers.append(conv_dilation + [norm_dilation])

    if self._pool_kernel_size is None:
      pooling = [
          helper.GlobalAveragePooling2DQuantized(),
          helper.ReshapeQuantized((1, 1, channels))
      ]
    else:
      pooling = [helper.AveragePooling2DQuantized(self._pool_kernel_size)]

    conv2 = helper.Conv2DQuantized(
        filters=self._output_channels,
        kernel_size=(1, 1),
        kernel_initializer=tf_utils.clone_initializer(
            self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        use_bias=False,
        activation=helper.NoOpActivation())
    norm2 = norm(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon)

    self.aspp_layers.append(pooling + [conv2, norm2])
    self._resizing_layer = helper.ResizingQuantized(
        height, width, interpolation=self._interpolation)

    self._projection = [
        helper.Conv2DQuantized(
            filters=self._output_channels,
            kernel_size=(1, 1),
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer,
            use_bias=False,
            activation=helper.NoOpActivation()),
        norm(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon)
    ]
    self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)
    self._concat_layer = helper.ConcatenateQuantized(axis=-1)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    if training is None:
      training = tf.keras.backend.learning_phase()
    result = []
    for i, layers in enumerate(self.aspp_layers):
      x = inputs
      for layer in layers:
        # Apply layers sequentially.
        x = layer(x, training=training)
      x = self._activation_fn(x)

      # Apply resize layer to the end of the last set of layers.
      if i == len(self.aspp_layers) - 1:
        x = self._resizing_layer(x)

      result.append(tf.cast(x, inputs.dtype))
    x = self._concat_layer(result)
    for layer in self._projection:
      x = layer(x, training=training)
    x = self._activation_fn(x)
    return self._dropout_layer(x)


@tf.keras.utils.register_keras_serializable(package='Vision')
class ASPPQuantized(aspp.ASPP):
  """Creates a quantized Atrous Spatial Pyramid Pooling (ASPP) layer."""

  def __init__(
      self,
      level: int,
      dilation_rates: List[int],
      num_filters: int = 256,
      pool_kernel_size: Optional[int] = None,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout_rate: float = 0.0,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      use_depthwise_convolution: bool = False,
      spp_layer_version: str = 'v1',
      output_tensor: bool = True,
      **kwargs):
    """Initializes an Atrous Spatial Pyramid Pooling (ASPP) layer.

    Args:
      level: An `int` level to apply ASPP.
      dilation_rates: A `list` of dilation rates.
      num_filters: An `int` number of output filters in ASPP.
      pool_kernel_size: A `list` of [height, width] of pooling kernel size or
        None. Pooling size is with respect to original image size, it will be
        scaled down by 2**level. If None, global average pooling is used.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      activation: A `str` activation to be used in ASPP.
      dropout_rate: A `float` rate for dropout regularization.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      interpolation: A `str` of interpolation method. It should be one of
        `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
        `gaussian`, or `mitchellcubic`.
      use_depthwise_convolution: If True depthwise separable convolutions will
        be added to the Atrous spatial pyramid pooling.
      spp_layer_version: A `str` of spatial pyramid pooling layer version.
      output_tensor: Whether to output a single tensor or a dictionary of
        tensor. Default is true.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(
        level=level,
        dilation_rates=dilation_rates,
        num_filters=num_filters,
        pool_kernel_size=pool_kernel_size,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        activation=activation,
        dropout_rate=dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        interpolation=interpolation,
        use_depthwise_convolution=use_depthwise_convolution,
        spp_layer_version=spp_layer_version,
        output_tensor=output_tensor,
        **kwargs)

    self._aspp_layer = SpatialPyramidPoolingQuantized

  def call(self, inputs: Union[tf.Tensor, Mapping[str,
                                                  tf.Tensor]]) -> tf.Tensor:
    """Calls the Atrous Spatial Pyramid Pooling (ASPP) layer on an input.

    The output of ASPP will be a dict of {`level`, `tf.Tensor`} even if only one
    level is present, if output_tensor is false. Hence, this will be compatible
    with the rest of the segmentation model interfaces.
    If output_tensor is true, a single tensot is output.

    Args:
      inputs: A `tf.Tensor` of shape [batch, height_l, width_l, filter_size] or
        a `dict` of `tf.Tensor` where
        - key: A `str` of the level of the multilevel feature maps.
        - values: A `tf.Tensor` of shape [batch, height_l, width_l,
          filter_size].

    Returns:
      A `tf.Tensor` of shape [batch, height_l, width_l, filter_size] or a `dict`
        of `tf.Tensor` where
        - key: A `str` of the level of the multilevel feature maps.
        - values: A `tf.Tensor` of output of ASPP module.
    """
    level = str(self._config_dict['level'])
    backbone_output = inputs[level] if isinstance(inputs, dict) else inputs
    return self.aspp(backbone_output)


class BatchNormalizationWrapper(tf.keras.layers.Wrapper):
  """A BatchNormalizationWrapper that explicitly not folded.

  It just added an identity depthwise conv right before the normalization.
  As a result, given normalization op just folded into the identity depthwise
  conv layer.

  Note that it only used when the batch normalization folding is not working.
  It makes quantize them as a 1x1 depthwise conv layer that just work as same
  as inference mode for the normalization. (Basically mult and add for the BN.)
  """

  def call(self, inputs: tf.Tensor, *args: Any, **kwargs: Any) -> tf.Tensor:
    channels = tf.shape(inputs)[-1]
    x = tf.nn.depthwise_conv2d(
        inputs, tf.ones([1, 1, channels, 1]), [1, 1, 1, 1], 'VALID')
    outputs = self.layer.call(x, *args, **kwargs)
    return outputs
