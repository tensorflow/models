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

"""Contains common building blocks for neural networks."""

from typing import Any, Callable, Dict, Union

import tensorflow as tf, tf_keras
import tensorflow_model_optimization as tfmot

from official.modeling import tf_utils
from official.projects.qat.vision.n_bit import configs
from official.vision.modeling.layers import nn_layers

# Type annotations.
States = Dict[str, tf.Tensor]
Activation = Union[str, Callable]


class NoOpActivation:
  """No-op activation which simply returns the incoming tensor.

  This activation is required to distinguish between `keras.activations.linear`
  which does the same thing. The main difference is that NoOpActivation should
  not have any quantize operation applied to it.
  """

  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    return x

  def get_config(self) -> Dict[str, Any]:
    """Get a config of this object."""
    return {}

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, NoOpActivation)

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)


def _quantize_wrapped_layer(cls, quantize_config):
  def constructor(*arg, **kwargs):
    return tfmot.quantization.keras.QuantizeWrapperV2(
        cls(*arg, **kwargs),
        quantize_config)
  return constructor


@tf_keras.utils.register_keras_serializable(package='Vision')
class SqueezeExcitationNBitQuantized(tf_keras.layers.Layer):
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
               num_bits_weight=8,
               num_bits_activation=8,
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
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      gating_activation: A `str` name of the activation function for final
        gating function.
      num_bits_weight: An `int` number of bits for the weight. Default to 8.
      num_bits_activation: An `int` number of bits for the weight. Default to 8.
      **kwargs: Additional keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._use_3d_input = use_3d_input
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation
    if tf_keras.backend.image_data_format() == 'channels_last':
      if not use_3d_input:
        self._spatial_axis = [1, 2]
      else:
        self._spatial_axis = [1, 2, 3]
    else:
      if not use_3d_input:
        self._spatial_axis = [2, 3]
      else:
        self._spatial_axis = [2, 3, 4]
    self._activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(activation, use_keras_layer=True),
        configs.DefaultNBitActivationQuantizeConfig(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    self._gating_activation_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_utils.get_activation(gating_activation, use_keras_layer=True),
        configs.DefaultNBitActivationQuantizeConfig(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))

  def build(self, input_shape):
    conv2d_quantized = _quantize_wrapped_layer(
        tf_keras.layers.Conv2D,
        configs.DefaultNBitConvQuantizeConfig(
            ['kernel'], ['activation'], False,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    conv2d_quantized_output_quantized = _quantize_wrapped_layer(
        tf_keras.layers.Conv2D,
        configs.DefaultNBitConvQuantizeConfig(
            ['kernel'], ['activation'], True,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    num_reduced_filters = nn_layers.make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by)

    self._se_reduce = conv2d_quantized(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())

    self._se_expand = conv2d_quantized_output_quantized(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=NoOpActivation())

    self._multiply = tfmot.quantization.keras.QuantizeWrapperV2(
        tf_keras.layers.Multiply(),
        configs.DefaultNBitQuantizeConfig(
            [], [], True, num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation))
    self._reduce_mean_quantizer = (
        tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=self._num_bits_activation, per_axis=False,
            symmetric=False, narrow_range=False))  # activation/output
    self._reduce_mean_quantizer_vars = self._reduce_mean_quantizer.build(
        None, 'reduce_mean_quantizer_vars', self)

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
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._reduce_mean_quantizer(
        x, training, self._reduce_mean_quantizer_vars)
    x = self._activation_layer(self._se_reduce(x))
    x = self._gating_activation_layer(self._se_expand(x))
    x = self._multiply([x, inputs])
    return x
