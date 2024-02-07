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

"""Contains feature fusion blocks for panoptic segmentation models."""
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import tensorflow as tf, tf_keras

from official.modeling import tf_utils


# Type annotations.
States = Dict[str, tf.Tensor]
Activation = Union[str, Callable]


class PanopticDeepLabFusion(tf_keras.layers.Layer):
  """Creates a Panoptic DeepLab feature Fusion layer.

  This implements the feature fusion introduced in the paper:
  Cheng et al. Panoptic-DeepLab
  (https://arxiv.org/pdf/1911.10194.pdf)
  """

  def __init__(
      self,
      level: int,
      low_level: List[int],
      num_projection_filters: List[int],
      num_output_filters: int = 256,
      use_depthwise_convolution: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      **kwargs):
    """Initializes panoptic FPN feature fusion layer.

    Args:
      level: An `int` level at which the decoder was appled at.
      low_level: A list of `int` of minimum level to use in feature fusion.
      num_projection_filters: A list of `int` with number of filters for
        projection conv2d layers.
      num_output_filters: An `int` number of filters in output conv2d layers.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      interpolation: A `str` interpolation method for upsampling. Defaults to
        `bilinear`.
      **kwargs: Additional keyword arguments to be passed.
    Returns:
      A `float` `tf.Tensor` of shape [batch_size, feature_height, feature_width,
        feature_channel].
    """
    super(PanopticDeepLabFusion, self).__init__(**kwargs)

    self._config_dict = {
        'level': level,
        'low_level': low_level,
        'num_projection_filters': num_projection_filters,
        'num_output_filters': num_output_filters,
        'use_depthwise_convolution': use_depthwise_convolution,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'interpolation': interpolation
    }
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: List[tf.TensorShape]):
    conv_op = tf_keras.layers.Conv2D
    conv_kwargs = {
        'padding': 'same',
        'use_bias': True,
        'kernel_initializer': tf.initializers.VarianceScaling(),
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
    }
    bn_op = (tf_keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf_keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._channel_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._projection_convs = []
    self._projection_norms = []
    self._fusion_convs = []
    self._fusion_norms = []
    for i in range(len(self._config_dict['low_level'])):
      self._projection_convs.append(
          conv_op(
              filters=self._config_dict['num_projection_filters'][i],
              kernel_size=1,
              **conv_kwargs))
      if self._config_dict['use_depthwise_convolution']:
        depthwise_initializer = tf_keras.initializers.RandomNormal(stddev=0.01)
        fusion_conv = tf_keras.Sequential([
            tf_keras.layers.DepthwiseConv2D(
                kernel_size=5,
                padding='same',
                use_bias=True,
                depthwise_initializer=depthwise_initializer,
                depthwise_regularizer=self._config_dict['kernel_regularizer'],
                depth_multiplier=1),
            bn_op(**bn_kwargs),
            conv_op(
                filters=self._config_dict['num_output_filters'],
                kernel_size=1,
                **conv_kwargs)])
      else:
        fusion_conv = conv_op(
            filters=self._config_dict['num_output_filters'],
            kernel_size=5,
            **conv_kwargs)
      self._fusion_convs.append(fusion_conv)
      self._projection_norms.append(bn_op(**bn_kwargs))
      self._fusion_norms.append(bn_op(**bn_kwargs))

  def call(self, inputs, training=None):
    if training is None:
      training = tf_keras.backend.learning_phase()

    backbone_output = inputs[0]
    decoder_output = inputs[1][str(self._config_dict['level'])]

    x = decoder_output
    for i in range(len(self._config_dict['low_level'])):
      feature = backbone_output[str(self._config_dict['low_level'][i])]
      feature = self._projection_convs[i](feature)
      feature = self._projection_norms[i](feature, training=training)
      feature = self._activation(feature)

      shape = tf.shape(feature)
      x = tf.image.resize(
          x, size=[shape[1], shape[2]],
          method=self._config_dict['interpolation'])
      x = tf.cast(x, dtype=feature.dtype)
      x = tf.concat([x, feature], axis=self._channel_axis)

      x = self._fusion_convs[i](x)
      x = self._fusion_norms[i](x, training=training)
      x = self._activation(x)
    return x

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
