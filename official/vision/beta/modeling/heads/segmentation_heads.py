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

"""Contains definitions of segmentation heads."""

import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers
from official.vision.beta.ops import spatial_transform_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class SegmentationHead(tf.keras.layers.Layer):
  """Creates a segmentation head."""

  def __init__(self,
               num_classes,
               level,
               num_convs=2,
               num_filters=256,
               upsample_factor=1,
               feature_fusion=None,
               low_level=2,
               low_level_num_filters=48,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_regularizer=None,
               bias_regularizer=None,
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
      upsample_factor: An `int` number to specify the upsampling factor to
        generate finer mask. Default 1 means no upsampling is applied.
      feature_fusion: One of `deeplabv3plus`, `pyramid_fusion`, or None. If
        `deeplabv3plus`, features from decoder_features[level] will be fused
        with low level feature maps from backbone. If `pyramid_fusion`,
        multiscale features will be resized and fused at the target level.
      low_level: An `int` of backbone level to be used for feature fusion. It is
        used when feature_fusion is set to `deeplabv3plus`.
      low_level_num_filters: An `int` of reduced number of filters for the low
        level features before fusing it with higher level features. It is only
        used when feature_fusion is set to `deeplabv3plus`.
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
    super(SegmentationHead, self).__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'level': level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'upsample_factor': upsample_factor,
        'feature_fusion': feature_fusion,
        'low_level': low_level,
        'low_level_num_filters': low_level_num_filters,
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
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape):
    """Creates the variables of the segmentation head."""
    conv_op = tf.keras.layers.Conv2D
    conv_kwargs = {
        'kernel_size': 3,
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=0.01),
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
    }
    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    if self._config_dict['feature_fusion'] == 'deeplabv3plus':
      # Deeplabv3+ feature fusion layers.
      self._dlv3p_conv = conv_op(
          kernel_size=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          kernel_regularizer=self._config_dict['kernel_regularizer'],
          name='segmentation_head_deeplabv3p_fusion_conv',
          filters=self._config_dict['low_level_num_filters'])

      self._dlv3p_norm = bn_op(
          name='segmentation_head_deeplabv3p_fusion_norm', **bn_kwargs)

    # Segmentation head layers.
    self._convs = []
    self._norms = []
    for i in range(self._config_dict['num_convs']):
      conv_name = 'segmentation_head_conv_{}'.format(i)
      self._convs.append(
          conv_op(
              name=conv_name,
              filters=self._config_dict['num_filters'],
              **conv_kwargs))
      norm_name = 'segmentation_head_norm_{}'.format(i)
      self._norms.append(bn_op(name=norm_name, **bn_kwargs))

    self._classifier = conv_op(
        name='segmentation_output',
        filters=self._config_dict['num_classes'],
        kernel_size=1,
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    super(SegmentationHead, self).build(input_shape)

  def call(self, backbone_output, decoder_output):
    """Forward pass of the segmentation head.

    Args:
      backbone_output: A `dict` of tensors
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].
      decoder_output: A `dict` of tensors
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].
    Returns:
      segmentation prediction mask: A `tf.Tensor` of the segmentation mask
        scores predicted from input features.
    """
    if self._config_dict['feature_fusion'] == 'deeplabv3plus':
      # deeplabv3+ feature fusion
      x = decoder_output[str(self._config_dict['level'])]
      y = backbone_output[str(
          self._config_dict['low_level'])]
      y = self._dlv3p_norm(self._dlv3p_conv(y))
      y = self._activation(y)

      x = tf.image.resize(
          x, tf.shape(y)[1:3], method=tf.image.ResizeMethod.BILINEAR)
      x = tf.cast(x, dtype=y.dtype)
      x = tf.concat([x, y], axis=self._bn_axis)
    elif self._config_dict['feature_fusion'] == 'pyramid_fusion':
      x = nn_layers.pyramid_feature_fusion(decoder_output,
                                           self._config_dict['level'])
    else:
      x = decoder_output[str(self._config_dict['level'])]

    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x)
      x = self._activation(x)
    x = spatial_transform_ops.nearest_upsampling(
        x, scale=self._config_dict['upsample_factor'])
    return self._classifier(x)

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
