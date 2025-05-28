# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Segmentation heads."""

from typing import Any, Union, Sequence, Mapping, Tuple
import tensorflow as tf, tf_keras

from official.modeling import tf_utils


@tf_keras.utils.register_keras_serializable(package='Vision')
class SegmentationHead3D(tf_keras.layers.Layer):
  """Segmentation head for 3D input."""

  def __init__(self,
               num_classes: int,
               level: Union[int, str],
               num_convs: int = 2,
               num_filters: int = 256,
               upsample_factor: int = 1,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               use_batch_normalization: bool = False,
               kernel_regularizer: tf_keras.regularizers.Regularizer = None,
               bias_regularizer: tf_keras.regularizers.Regularizer = None,
               output_logits: bool = True,  # pytype: disable=annotation-type-mismatch  # typed-keras
               **kwargs):
    """Initialize params to build segmentation head.

    Args:
      num_classes: `int` number of mask classification categories. The number of
        classes does not include background class.
      level: `int` or `str`, level to use to build segmentation head.
      num_convs: `int` number of stacked convolution before the last prediction
        layer.
      num_filters: `int` number to specify the number of filters used. Default
        is 256.
      upsample_factor: `int` number to specify the upsampling factor to generate
        finer mask. Default 1 means no upsampling is applied.
      activation: `string`, indicating which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: `bool`, whether to use synchronized batch normalization
        across different replicas.
      norm_momentum: `float`, the momentum parameter of the normalization
        layers.
      norm_epsilon: `float`, the epsilon parameter of the normalization layers.
      use_batch_normalization: A bool of whether to use batch normalization or
        not.
      kernel_regularizer: `tf_keras.regularizers.Regularizer` object for layer
        kernel.
      bias_regularizer: `tf_keras.regularizers.Regularizer` object for bias.
      output_logits: A `bool` of whether to output logits or not. Default
        is True. If set to False, output softmax.
      **kwargs: other keyword arguments passed to Layer.
    """
    super(SegmentationHead3D, self).__init__(**kwargs)

    self._config_dict = {
        'num_classes': num_classes,
        'level': level,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'upsample_factor': upsample_factor,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'use_batch_normalization': use_batch_normalization,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'output_logits': output_logits
    }
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation, use_keras_layer=True)

  def build(self, input_shape: Union[tf.TensorShape, Sequence[tf.TensorShape]]):
    """Creates the variables of the segmentation head."""
    conv_op = tf_keras.layers.Conv3D
    conv_kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'use_bias': False,
        'kernel_initializer': tf_keras.initializers.RandomNormal(stddev=0.01),
        'kernel_regularizer': self._config_dict['kernel_regularizer'],
    }
    final_kernel_size = (1, 1, 1)

    bn_op = (
        tf_keras.layers.experimental.SyncBatchNormalization
        if self._config_dict['use_sync_bn'] else
        tf_keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

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
      if self._config_dict['use_batch_normalization']:
        self._norms.append(bn_op(name=norm_name, **bn_kwargs))

    self._classifier = conv_op(
        name='segmentation_output',
        filters=self._config_dict['num_classes'],
        kernel_size=final_kernel_size,
        padding='valid',
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf_keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'])

    super(SegmentationHead3D, self).build(input_shape)

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
    decoder_output = inputs[1]
    x = decoder_output[str(self._config_dict['level'])] if isinstance(
        decoder_output, dict) else decoder_output

    for i, conv in enumerate(self._convs):
      x = conv(x)
      if self._norms:
        x = self._norms[i](x)
      x = self._activation(x)

    x = tf_keras.layers.UpSampling3D(size=self._config_dict['upsample_factor'])(
        x)
    x = self._classifier(x)
    return x if self._config_dict['output_logits'] else tf_keras.layers.Softmax(
        dtype='float32')(
            x)

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]):
    return cls(**config)
