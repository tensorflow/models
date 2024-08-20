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

"""Contains definition for postprocessing layer to genrate panoptic segmentations."""

from typing import Any, List, Optional, Union
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.vision.modeling.layers import nn_layers
from official.vision.ops import spatial_transform_ops


@tf_keras.utils.register_keras_serializable(package='Vision')
class MaskConverHead(tf_keras.layers.Layer):
  """Creates a MaskConver head."""

  def __init__(
      self,
      num_classes: int,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      use_depthwise_convolution: bool = False,
      depthwise_kernel_size: int = 3,
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
      use_layer_norm: bool = False,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_initializer: Optional[Any] = tf.constant_initializer(0.0),
      **kwargs):
    """Initializes a maskconver head.

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      level: An `int` or `str`, level to use to build maskconver head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      depthwise_kernel_size: An `int` for depthwise kernel size.
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
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      use_layer_norm: A `bool` for whether to use layer norm.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      bias_initializer: Bias initializer for the classification layer.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'level': level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_depthwise_convolution': use_depthwise_convolution,
        'depthwise_kernel_size': depthwise_kernel_size,
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
        'bias_initializer': bias_initializer,
        'use_layer_norm': use_layer_norm,
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
    if self._config_dict['use_layer_norm']:
      bn_layer = lambda: tf_keras.layers.LayerNormalization(epsilon=1e-6)
    else:
      bn_kwargs = {
          'axis': self._bn_axis,
          'momentum': self._config_dict['norm_momentum'],
          'epsilon': self._config_dict['norm_epsilon'],
      }
      if self._config_dict['use_sync_bn']:
        bn_layer = lambda: tf_keras.layers.experimental.SyncBatchNormalization(  # pylint: disable=g-long-lambda
            **bn_kwargs)
      else:
        bn_layer = lambda: tf_keras.layers.BatchNormalization(**bn_kwargs)

    if self._config_dict['feature_fusion'] in {'deeplabv3plus',
                                               'deeplabv3plus_sum_to_merge'}:
      # Deeplabv3+ feature fusion layers.
      self._dlv3p_conv = conv_op(
          kernel_size=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf_keras.initializers.he_normal(),
          kernel_regularizer=self._config_dict['kernel_regularizer'],
          name='segmentation_head_deeplabv3p_fusion_conv',
          filters=self._config_dict['low_level_num_filters'])

      self._dlv3p_norm = bn_layer()

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
                kernel_size=self._config_dict['depthwise_kernel_size'],
                padding='same',
                use_bias=False,
                depth_multiplier=1))
        self._norms.append(bn_layer())
      conv_name = 'segmentation_head_conv_{}'.format(i)
      self._convs.append(
          conv_op(
              name=conv_name,
              filters=self._config_dict['num_filters'],
              kernel_size=3 if not use_depthwise_convolution else 1,
              padding='same',
              use_bias=False,
              kernel_initializer=tf_keras.initializers.he_normal(),
              kernel_regularizer=self._config_dict['kernel_regularizer']))
      self._norms.append(bn_layer())

    self._classifier = conv_op(
        name='segmentation_output',
        filters=self._config_dict['num_classes'],
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        bias_initializer=self._config_dict['bias_initializer'],
        kernel_initializer=tf_keras.initializers.truncated_normal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    super().build(input_shape)

  def call(self, inputs):
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


class MLP(tf_keras.Model):
  """MLP."""

  def __init__(
      self,
      hidden_dim: int = 256,
      output_dim: int = 256,
      num_layers: int = 2,
      activation: str = 'swish',
      l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    dims = [hidden_dim] * (num_layers - 1)
    # pylint: disable=g-complex-comprehension
    bn_layer = lambda: tf_keras.layers.LayerNormalization(epsilon=1e-6)
    self.dense = [
        tf_keras.layers.Dense(d, kernel_regularizer=l2_regularizer)
        for d in dims + [output_dim]
    ]
    self.norms = [bn_layer() for _ in dims]
    self.activation = tf_keras.activations.get(activation)

  def call(
      self, inputs: tf.Tensor, training: Any = None, mask: Any = None
  ) -> tf.Tensor:
    x = inputs
    for i, layer in enumerate(self.dense):
      x = (
          self.activation(self.norms[i](layer(x)))
          if i < self.num_layers - 1
          else layer(x)
      )
    return x
