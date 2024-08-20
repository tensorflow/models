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

"""Contains definition for multi-scale MaskConver head."""

from typing import Any, List, Optional, Union
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.vision.ops import spatial_transform_ops


@tf_keras.utils.register_keras_serializable(package='Vision')
class MultiScaleMaskConverHead(tf_keras.layers.Layer):
  """Creates a MaskConver head."""

  def __init__(
      self,
      num_classes: int,
      min_level: Union[int, str],
      max_level: Union[int, str],
      num_convs: int = 2,
      num_filters: int = 256,
      use_depthwise_convolution: bool = False,
      depthwise_kernel_size: int = 3,
      prediction_kernel_size: int = 1,
      upsample_factor: int = 1,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      use_layer_norm: bool = True,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_initializer: Optional[Any] = tf.constant_initializer(0.0),
      **kwargs):
    """Initializes a maskconver head.

    Args:
      num_classes: An `int` number of mask classification categories. The number
        of classes does not include background class.
      min_level: An `int` or `str`, min level to use to build maskconver head.
      max_level: An `int` or `str`, max level to use to build maskconver head.
      num_convs: An `int` number of stacked convolution before the last
        prediction layer.
      num_filters: An `int` number to specify the number of filters used.
        Default is 256.
      use_depthwise_convolution: A bool to specify if use depthwise separable
        convolutions.
      depthwise_kernel_size: An `int` for the depthwise kernel size.
      prediction_kernel_size: An `int` number to specify the kernel size of the
      prediction layer.
      upsample_factor: An `int` number to specify the upsampling factor to
        generate finer mask. Default 1 means no upsampling is applied.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      use_layer_norm: A `bool` whether to use layer norm.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      bias_initializer: Bias initializer for the classification layer.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'min_level': min_level,
        'max_level': max_level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_depthwise_convolution': use_depthwise_convolution,
        'depthwise_kernel_size': depthwise_kernel_size,
        'prediction_kernel_size': prediction_kernel_size,
        'upsample_factor': upsample_factor,
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

    # Segmentation head layers.
    self._convs = []
    self._norms = []
    for level in range(
        self._config_dict['min_level'], self._config_dict['max_level'] + 1
    ):
      level_norms = []
      for i in range(self._config_dict['num_convs']):
        # We use shared convolution layers across levels.
        if use_depthwise_convolution:
          if level == self._config_dict['min_level']:
            self._convs.append(
                tf_keras.layers.DepthwiseConv2D(
                    name='segmentation_head_depthwise_conv_{}'.format(i),
                    kernel_size=self._config_dict['depthwise_kernel_size'],
                    padding='same',
                    use_bias=False,
                    depth_multiplier=1))
          level_norms.append(bn_layer())
        if level == self._config_dict['min_level']:
          conv_name = 'segmentation_head_conv_{}'.format(i)
          self._convs.append(
              conv_op(
                  name=conv_name,
                  filters=self._config_dict['num_filters'],
                  kernel_size=1 if use_depthwise_convolution else 3,
                  padding='same',
                  use_bias=False,
                  kernel_initializer=tf_keras.initializers.he_normal(),
                  kernel_regularizer=self._config_dict['kernel_regularizer']))
        level_norms.append(bn_layer())
      self._norms.append(level_norms)

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
    """Forward pass of the multiscale maskconver head."""
    outputs = {}
    for i, level in enumerate(
        range(self._config_dict['min_level'],
              self._config_dict['max_level'] + 1)):
      x = inputs[str(level)]
      for conv, norm in zip(self._convs, self._norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      if self._config_dict['upsample_factor'] > 1:
        x = spatial_transform_ops.nearest_upsampling(
            x, scale=self._config_dict['upsample_factor'])
      outputs[level] = self._classifier(x)

    return outputs

  def get_config(self):
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(self._config_dict.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)
