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

"""Contains definition of instance center heads."""
from typing import List, Union, Optional, Mapping, Tuple
import tensorflow as tf

from official.vision.beta.modeling.heads import segmentation_heads
from official.vision.beta.ops import spatial_transform_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class InstanceCenterHead(segmentation_heads.SegmentationHead):
  """Creates a segmentation head."""

  def __init__(
      self,
      level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      use_depthwise_convolution: bool = False,
      kernel_size: int = 3,
      prediction_kernel_size: int = 1,
      upsample_factor: int = 1,
      feature_fusion: Optional[str] = None,
      decoder_min_level: Optional[int] = None,
      decoder_max_level: Optional[int] = None,
      low_level: Union[int, List[int]] = 2,
      low_level_num_filters: Union[int, List[int]] = 48,
      num_decoder_filters: int = 256,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a instance center head.

    Args:
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
      feature_fusion: One of `deeplabv3plus`, `pyramid_fusion`,
        `panoptic_fpn_fusion`, or None. If `deeplabv3plus`, features from
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
    super(InstanceCenterHead, self).__init__(
        num_classes=2,
        level=level,
        num_convs=num_convs,
        num_filters=num_filters,
        use_depthwise_convolution=use_depthwise_convolution,
        kernel_size=kernel_size,
        prediction_kernel_size=prediction_kernel_size,
        upsample_factor=upsample_factor,
        feature_fusion=feature_fusion,
        decoder_min_level=decoder_min_level,
        decoder_max_level=decoder_max_level,
        low_level=low_level,
        low_level_num_filters=low_level_num_filters,
        num_decoder_filters=num_decoder_filters,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        **kwargs)


  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    self._instance_center_prediction_conv = tf.keras.layers.Conv2D(
        name='instance_center_prediction',
        filters=1,
        kernel_size=self._config_dict['prediction_kernel_size'],
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])
    super(InstanceCenterHead, self).build(input_shape)


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
    x = self._fuse_features(inputs)

    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x)
      x = self._activation(x)
    if self._config_dict['upsample_factor'] > 1:
      x = spatial_transform_ops.nearest_upsampling(
          x, scale=self._config_dict['upsample_factor'])

    instance_center_prediction = self._instance_center_prediction_conv(x)
    instance_center_regression = self._prediction_conv(x)
    outputs = {
        'instance_center_prediction': instance_center_prediction,
        'instance_center_regression': instance_center_regression
    }
    return outputs

  def get_config(self):
    config_dict = super(InstanceCenterHead, self).get_config().copy()
    config_dict.pop('num_classes')
    return config_dict
