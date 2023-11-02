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

"""Contains definitions of segmentation heads."""
from typing import List, Union, Optional, Mapping, Tuple, Any
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.vision.modeling.layers import nn_layers
from official.vision.ops import spatial_transform_ops


class MaskScoring(tf_keras.Model):
  """Creates a mask scoring layer.

  This implements mask scoring layer from the paper:

  Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, Xinggang Wang.
  Mask Scoring R-CNN.
  (https://arxiv.org/pdf/1903.00241.pdf)
  """

  def __init__(
      self,
      num_classes: int,
      fc_input_size: List[int],
      num_convs: int = 3,
      num_filters: int = 256,
      use_depthwise_convolution: bool = False,
      fc_dims: int = 1024,
      num_fcs: int = 2,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):

    """Initializes mask scoring layer.

    Args:
      num_classes: An `int` for number of classes.
      fc_input_size: A List of `int` for the input size of the
        fully connected layers.
      num_convs: An`int` for number of conv layers.
      num_filters: An `int` for the number of filters for conv layers.
      use_depthwise_convolution: A `bool`, whether or not using depthwise convs.
      fc_dims: An `int` number of filters for each fully connected layers.
      num_fcs: An `int` for number of fully connected layers.
      activation: A `str` name of the activation function.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      norm_momentum: A float for the momentum in BatchNorm. Defaults to 0.99.
      norm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(MaskScoring, self).__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'fc_input_size': fc_input_size,
        'fc_dims': fc_dims,
        'num_fcs': num_fcs,
        'use_sync_bn': use_sync_bn,
        'use_depthwise_convolution': use_depthwise_convolution,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the mask scoring head."""
    conv_op = tf_keras.layers.Conv2D
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
    }
    conv_kwargs.update({
        'kernel_initializer': tf_keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        'bias_initializer': tf.zeros_initializer(),
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
        'bias_regularizer': self._config_dict['bias_regularizer'],
    })
    bn_op = tf_keras.layers.BatchNormalization
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
        'synchronized': self._config_dict['use_sync_bn'],
    }

    self._convs = []
    self._conv_norms = []
    for i in range(self._config_dict['num_convs']):
      if self._config_dict['use_depthwise_convolution']:
        self._convs.append(
            tf_keras.layers.DepthwiseConv2D(
                name='mask-scoring-depthwise-conv-{}'.format(i),
                kernel_size=3,
                padding='same',
                use_bias=False,
                depthwise_initializer=tf_keras.initializers.RandomNormal(
                    stddev=0.01),
                depthwise_regularizer=self._config_dict['kernel_regularizer'],
                depth_multiplier=1))
        norm_name = 'mask-scoring-depthwise-bn-{}'.format(i)
        self._conv_norms.append(bn_op(name=norm_name, **bn_kwargs))
      conv_name = 'mask-scoring-conv-{}'.format(i)
      if 'kernel_initializer' in conv_kwargs:
        conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
            conv_kwargs['kernel_initializer'])
      if self._config_dict['use_depthwise_convolution']:
        conv_kwargs['kernel_size'] = 1
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'mask-scoring-bn-{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._fcs = []
    self._fc_norms = []
    for i in range(self._config_dict['num_fcs']):
      fc_name = 'mask-scoring-fc-{}'.format(i)
      self._fcs.append(
          tf_keras.layers.Dense(
              units=self._config_dict['fc_dims'],
              kernel_initializer=tf_keras.initializers.VarianceScaling(
                  scale=1 / 3.0, mode='fan_out', distribution='uniform'),
              kernel_regularizer=self._config_dict['kernel_regularizer'],
              bias_regularizer=self._config_dict['bias_regularizer'],
              name=fc_name))
      bn_name = 'mask-scoring-fc-bn-{}'.format(i)
      self._fc_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._classifier = tf_keras.layers.Dense(
        units=self._config_dict['num_classes'],
        kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='iou-scores')

    super(MaskScoring, self).build(input_shape)

  def call(self, inputs: tf.Tensor, training: bool = None):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Forward pass mask scoring head.

    Args:
      inputs: A `tf.Tensor` of the shape [batch_size, width, size, num_classes],
      representing the segmentation logits.
      training: a `bool` indicating whether it is in `training` mode.

    Returns:
      mask_scores: A `tf.Tensor` of predicted mask scores
        [batch_size, num_classes].
    """
    x = tf.stop_gradient(inputs)
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation(x)

    # Casts feat to float32 so the resize op can be run on TPU.
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, size=self._config_dict['fc_input_size'],
                        method=tf.image.ResizeMethod.BILINEAR)
    # Casts it back to be compatible with the rest opetations.
    x = tf.cast(x, inputs.dtype)

    _, h, w, filters = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * filters])

    for fc, bn in zip(self._fcs, self._fc_norms):
      x = fc(x)
      x = bn(x)
      x = self._activation(x)

    ious = self._classifier(x)
    return ious

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@tf_keras.utils.register_keras_serializable(package='Vision')
class SegmentationHead(tf_keras.layers.Layer):
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
      logit_activation: Optional[str] = None,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
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
      feature_fusion: One of the constants in nn_layers.FeatureFusion, namely
        `deeplabv3plus`, `pyramid_fusion`, `panoptic_fpn_fusion`,
        `deeplabv3plus_sum_to_merge`, or None. If `deeplabv3plus`, features from
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
        used when feature_fusion is set to `deeplabv3plus` or
        `deeplabv3plus_sum_to_merge`.
      low_level_num_filters: An `int` of reduced number of filters for the low
        level features before fusing it with higher level features. It is only
        used when feature_fusion is set to `deeplabv3plus` or
        `deeplabv3plus_sum_to_merge`.
      num_decoder_filters: An `int` of number of filters in the decoder outputs.
        It is only used when feature_fusion is set to `panoptic_fpn_fusion`.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      logit_activation: Activation applied to the final classifier layer logits,
        e.g. 'sigmoid', 'softmax'. Can be useful in cases when the task does not
        use only cross entropy loss.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SegmentationHead, self).__init__(**kwargs)

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
        'logit_activation': logit_activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer
    }
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the segmentation head."""
    use_depthwise_convolution = self._config_dict['use_depthwise_convolution']
    conv_op = tf_keras.layers.Conv2D
    bn_op = tf_keras.layers.BatchNormalization
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
        'synchronized': self._config_dict['use_sync_bn'],
    }

    if self._config_dict['feature_fusion'] in {'deeplabv3plus',
                                               'deeplabv3plus_sum_to_merge'}:
      # Deeplabv3+ feature fusion layers.
      self._dlv3p_conv = conv_op(
          kernel_size=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
          kernel_regularizer=self._config_dict['kernel_regularizer'],
          name='segmentation_head_deeplabv3p_fusion_conv',
          filters=self._config_dict['low_level_num_filters'])

      self._dlv3p_norm = bn_op(
          name='segmentation_head_deeplabv3p_fusion_norm', **bn_kwargs)

    elif self._config_dict['feature_fusion'] == 'panoptic_fpn_fusion':
      self._panoptic_fpn_fusion = nn_layers.PanopticFPNFusion(
          min_level=self._config_dict['decoder_min_level'],
          max_level=self._config_dict['decoder_max_level'],
          target_level=self._config_dict['level'],
          num_filters=self._config_dict['num_filters'],
          num_fpn_filters=self._config_dict['num_decoder_filters'],
          activation=self._config_dict['activation'],
          kernel_regularizer=self._config_dict['kernel_regularizer'],
          bias_regularizer=self._config_dict['bias_regularizer'])

    # Segmentation head layers.
    self._convs = []
    self._norms = []
    for i in range(self._config_dict['num_convs']):
      if use_depthwise_convolution:
        self._convs.append(
            tf_keras.layers.DepthwiseConv2D(
                name='segmentation_head_depthwise_conv_{}'.format(i),
                kernel_size=3,
                padding='same',
                use_bias=False,
                depthwise_initializer=tf_keras.initializers.RandomNormal(
                    stddev=0.01),
                depthwise_regularizer=self._config_dict['kernel_regularizer'],
                depth_multiplier=1))
        norm_name = 'segmentation_head_depthwise_norm_{}'.format(i)
        self._norms.append(bn_op(name=norm_name, **bn_kwargs))
      conv_name = 'segmentation_head_conv_{}'.format(i)
      self._convs.append(
          conv_op(
              name=conv_name,
              filters=self._config_dict['num_filters'],
              kernel_size=3 if not use_depthwise_convolution else 1,
              padding='same',
              use_bias=False,
              kernel_initializer=tf_keras.initializers.RandomNormal(
                  stddev=0.01),
              kernel_regularizer=self._config_dict['kernel_regularizer']))
      norm_name = 'segmentation_head_norm_{}'.format(i)
      self._norms.append(bn_op(name=norm_name, **bn_kwargs))

    self._classifier = conv_op(
        name='segmentation_output',
        filters=self._config_dict['num_classes'],
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        activation=self._config_dict['logit_activation'],
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

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
        The first is backbone endpoints, and the second is decoder endpoints.
    Returns:
      segmentation prediction mask: A `tf.Tensor` of the segmentation mask
        scores predicted from input features.
    """

    backbone_output = inputs[0]
    decoder_output = inputs[1]
    if self._config_dict['feature_fusion'] in {'deeplabv3plus',
                                               'deeplabv3plus_sum_to_merge'}:
      # deeplabv3+ feature fusion
      x = decoder_output[str(self._config_dict['level'])] if isinstance(
          decoder_output, dict) else decoder_output
      y = backbone_output[str(self._config_dict['low_level'])] if isinstance(
          backbone_output, dict) else backbone_output
      y = self._dlv3p_norm(self._dlv3p_conv(y))
      y = self._activation(y)

      x = tf.image.resize(
          x, tf.shape(y)[1:3], method=tf.image.ResizeMethod.BILINEAR)
      x = tf.cast(x, dtype=y.dtype)
      if self._config_dict['feature_fusion'] == 'deeplabv3plus':
        x = tf.concat([x, y], axis=self._bn_axis)
      else:
        x = tf_keras.layers.Add()([x, y])
    elif self._config_dict['feature_fusion'] == 'pyramid_fusion':
      if not isinstance(decoder_output, dict):
        raise ValueError('Only support dictionary decoder_output.')
      x = nn_layers.pyramid_feature_fusion(decoder_output,
                                           self._config_dict['level'])
    elif self._config_dict['feature_fusion'] == 'panoptic_fpn_fusion':
      x = self._panoptic_fpn_fusion(decoder_output)
    else:
      x = decoder_output[str(self._config_dict['level'])] if isinstance(
          decoder_output, dict) else decoder_output

    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x)
      x = self._activation(x)
    if self._config_dict['upsample_factor'] > 1:
      x = spatial_transform_ops.nearest_upsampling(
          x, scale=self._config_dict['upsample_factor'])

    return self._classifier(x)

  def get_config(self):
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(self._config_dict.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)
