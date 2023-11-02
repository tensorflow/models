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

"""Contains definitions for Panoptic Deeplab heads."""

from typing import List, Mapping, Optional, Tuple, Union
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.projects.panoptic.modeling.layers import fusion_layers
from official.vision.ops import spatial_transform_ops


class PanopticDeeplabHead(tf_keras.layers.Layer):
  """Creates a panoptic deeplab head."""

  def __init__(
      self,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      kernel_size: int = 3,
      use_depthwise_convolution: bool = False,
      upsample_factor: int = 1,
      low_level: Optional[List[int]] = None,
      low_level_num_filters: Optional[List[int]] = None,
      fusion_num_output_filters: int = 256,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a panoptic deeplab head.

    Args:
      level: An `int` or `str`, level to use to build head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      kernel_size:  An `int` number to specify the kernel size of the
        stacked convolutions before the last prediction layer.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      upsample_factor: An `int` number to specify the upsampling factor to
        generate finer mask. Default 1 means no upsampling is applied.
      low_level: An `int` of backbone level to be used for feature fusion. It is
        used when feature_fusion is set to `deeplabv3plus`.
      low_level_num_filters: An `int` of reduced number of filters for the low
        level features before fusing it with higher level features. It is only
        used when feature_fusion is set to `deeplabv3plus`.
      fusion_num_output_filters: An `int` number to specify the number of
        filters used by output layer of fusion module. Default is 256.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(PanopticDeeplabHead, self).__init__(**kwargs)

    self._config_dict = {
        'level': level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'kernel_size': kernel_size,
        'use_depthwise_convolution': use_depthwise_convolution,
        'upsample_factor': upsample_factor,
        'low_level': low_level,
        'low_level_num_filters': low_level_num_filters,
        'fusion_num_output_filters': fusion_num_output_filters,
        'activation': activation,
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
    """Creates the variables of the head."""
    kernel_size = self._config_dict['kernel_size']
    use_depthwise_convolution = self._config_dict['use_depthwise_convolution']
    random_initializer = tf_keras.initializers.RandomNormal(stddev=0.01)
    conv_op = tf_keras.layers.Conv2D
    conv_kwargs = {
        'kernel_size': kernel_size if not use_depthwise_convolution else 1,
        'padding': 'same',
        'use_bias': True,
        'kernel_initializer': random_initializer,
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
    }
    bn_op = (tf_keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf_keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    self._panoptic_deeplab_fusion = fusion_layers.PanopticDeepLabFusion(
        level=self._config_dict['level'],
        low_level=self._config_dict['low_level'],
        num_projection_filters=self._config_dict['low_level_num_filters'],
        num_output_filters=self._config_dict['fusion_num_output_filters'],
        use_depthwise_convolution=self
        ._config_dict['use_depthwise_convolution'],
        activation=self._config_dict['activation'],
        use_sync_bn=self._config_dict['use_sync_bn'],
        norm_momentum=self._config_dict['norm_momentum'],
        norm_epsilon=self._config_dict['norm_epsilon'],
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    # Stacked convolutions layers.
    self._convs = []
    self._norms = []
    for i in range(self._config_dict['num_convs']):
      if use_depthwise_convolution:
        self._convs.append(
            tf_keras.layers.DepthwiseConv2D(
                name='panoptic_deeplab_head_depthwise_conv_{}'.format(i),
                kernel_size=kernel_size,
                padding='same',
                use_bias=True,
                depthwise_initializer=random_initializer,
                depthwise_regularizer=self._config_dict['kernel_regularizer'],
                depth_multiplier=1))
        norm_name = 'panoptic_deeplab_head_depthwise_norm_{}'.format(i)
        self._norms.append(bn_op(name=norm_name, **bn_kwargs))
      conv_name = 'panoptic_deeplab_head_conv_{}'.format(i)
      self._convs.append(
          conv_op(
              name=conv_name,
              filters=self._config_dict['num_filters'],
              **conv_kwargs))
      norm_name = 'panoptic_deeplab_head_norm_{}'.format(i)
      self._norms.append(bn_op(name=norm_name, **bn_kwargs))

    super().build(input_shape)

  def call(self, inputs: Tuple[Union[tf.Tensor, Mapping[str, tf.Tensor]],
                               Union[tf.Tensor, Mapping[str, tf.Tensor]]],
           training=None):
    """Forward pass of the head.

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
      training: A bool, runs the model in training/eval mode.

    Returns:
      A `tf.Tensor` of the fused backbone and decoder features.
    """
    if training is None:
      training = tf_keras.backend.learning_phase()

    x = self._panoptic_deeplab_fusion(inputs, training=training)

    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x, training=training)
      x = self._activation(x)

    if self._config_dict['upsample_factor'] > 1:
      x = spatial_transform_ops.nearest_upsampling(
          x, scale=self._config_dict['upsample_factor'])

    return x

  def get_config(self):
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(self._config_dict.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf_keras.utils.register_keras_serializable(package='Vision')
class SemanticHead(PanopticDeeplabHead):
  """Creates a semantic head."""

  def __init__(
      self,
      num_classes: int,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      kernel_size: int = 3,
      prediction_kernel_size: int = 3,
      use_depthwise_convolution: bool = False,
      upsample_factor: int = 1,
      low_level: Optional[List[int]] = None,
      low_level_num_filters: Optional[List[int]] = None,
      fusion_num_output_filters: int = 256,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a instance center head.

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      level: An `int` or `str`, level to use to build head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      kernel_size:  An `int` number to specify the kernel size of the
        stacked convolutions before the last prediction layer.
      prediction_kernel_size: An `int` number to specify the kernel size of the
        prediction layer.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      upsample_factor: An `int` number to specify the upsampling factor to
        generate finer mask. Default 1 means no upsampling is applied.
      low_level: An `int` of backbone level to be used for feature fusion. It is
        used when feature_fusion is set to `deeplabv3plus`.
      low_level_num_filters: An `int` of reduced number of filters for the low
        level features before fusing it with higher level features. It is only
        used when feature_fusion is set to `deeplabv3plus`.
      fusion_num_output_filters: An `int` number to specify the number of
        filters used by output layer of fusion module. Default is 256.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SemanticHead, self).__init__(
        level=level,
        num_convs=num_convs,
        num_filters=num_filters,
        use_depthwise_convolution=use_depthwise_convolution,
        kernel_size=kernel_size,
        upsample_factor=upsample_factor,
        low_level=low_level,
        low_level_num_filters=low_level_num_filters,
        fusion_num_output_filters=fusion_num_output_filters,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        **kwargs)
    self._config_dict.update({
        'num_classes': num_classes,
        'prediction_kernel_size': prediction_kernel_size})

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the semantic head."""
    super(SemanticHead, self).build(input_shape)
    self._classifier = tf_keras.layers.Conv2D(
        name='semantic_output',
        filters=self._config_dict['num_classes'],
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

  def call(self, inputs: Tuple[Union[tf.Tensor, Mapping[str, tf.Tensor]],
                               Union[tf.Tensor, Mapping[str, tf.Tensor]]],
           training=None):
    """Forward pass of the head."""

    if training is None:
      training = tf_keras.backend.learning_phase()
    x = super(SemanticHead, self).call(inputs, training=training)
    outputs = self._classifier(x)
    return outputs


@tf_keras.utils.register_keras_serializable(package='Vision')
class InstanceHead(PanopticDeeplabHead):
  """Creates a instance head."""

  def __init__(
      self,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      kernel_size: int = 3,
      prediction_kernel_size: int = 3,
      use_depthwise_convolution: bool = False,
      upsample_factor: int = 1,
      low_level: Optional[List[int]] = None,
      low_level_num_filters: Optional[List[int]] = None,
      fusion_num_output_filters: int = 256,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a instance center head.

    Args:
      level: An `int` or `str`, level to use to build head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      kernel_size:  An `int` number to specify the kernel size of the
        stacked convolutions before the last prediction layer.
      prediction_kernel_size: An `int` number to specify the kernel size of the
        prediction layer.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      upsample_factor: An `int` number to specify the upsampling factor to
        generate finer mask. Default 1 means no upsampling is applied.
      low_level: An `int` of backbone level to be used for feature fusion. It is
        used when feature_fusion is set to `deeplabv3plus`.
      low_level_num_filters: An `int` of reduced number of filters for the low
        level features before fusing it with higher level features. It is only
        used when feature_fusion is set to `deeplabv3plus`.
      fusion_num_output_filters: An `int` number to specify the number of
        filters used by output layer of fusion module. Default is 256.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(InstanceHead, self).__init__(
        level=level,
        num_convs=num_convs,
        num_filters=num_filters,
        use_depthwise_convolution=use_depthwise_convolution,
        kernel_size=kernel_size,
        upsample_factor=upsample_factor,
        low_level=low_level,
        low_level_num_filters=low_level_num_filters,
        fusion_num_output_filters=fusion_num_output_filters,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        **kwargs)
    self._config_dict.update({
        'prediction_kernel_size': prediction_kernel_size})

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the instance head."""
    super(InstanceHead, self).build(input_shape)
    self._instance_center_prediction_conv = tf_keras.layers.Conv2D(
        name='instance_centers_heatmap',
        filters=1,
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    self._instance_center_regression_conv = tf_keras.layers.Conv2D(
        name='instance_centers_offset',
        filters=2,
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

  def call(self, inputs: Tuple[Union[tf.Tensor, Mapping[str, tf.Tensor]],
                               Union[tf.Tensor, Mapping[str, tf.Tensor]]],
           training=None):
    """Forward pass of the head."""

    if training is None:
      training = tf_keras.backend.learning_phase()

    x = super(InstanceHead, self).call(inputs, training=training)
    instance_centers_heatmap = self._instance_center_prediction_conv(x)
    instance_centers_offset = self._instance_center_regression_conv(x)
    outputs = {
        'instance_centers_heatmap': instance_centers_heatmap,
        'instance_centers_offset': instance_centers_offset
    }
    return outputs
