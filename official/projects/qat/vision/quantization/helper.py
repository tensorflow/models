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

"""Quantization helpers."""
from typing import Any, Dict

import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.projects.qat.vision.quantization import configs


_QUANTIZATION_WEIGHT_NAMES = [
    'output_max', 'output_min', 'optimizer_step', 'kernel_min', 'kernel_max',
    'add_three_min', 'add_three_max', 'divide_six_min', 'divide_six_max',
    'depthwise_kernel_min', 'depthwise_kernel_max',
    'reduce_mean_quantizer_vars_min', 'reduce_mean_quantizer_vars_max',
    'quantize_layer_min', 'quantize_layer_max',
    'quantize_layer_1_min', 'quantize_layer_1_max',
    'quantize_layer_2_min', 'quantize_layer_2_max',
    'quantize_layer_3_min', 'quantize_layer_3_max',
    'post_activation_min', 'post_activation_max',
]

_ORIGINAL_WEIGHT_NAME = [
    'kernel', 'depthwise_kernel', 'gamma', 'beta', 'moving_mean',
    'moving_variance', 'bias'
]


def is_quantization_weight_name(name: str) -> bool:
  simple_name = name.split('/')[-1].split(':')[0]
  if simple_name in _QUANTIZATION_WEIGHT_NAMES:
    return True
  if simple_name in _ORIGINAL_WEIGHT_NAME:
    return False
  raise ValueError('Variable name {} is not supported.'.format(simple_name))


def copy_original_weights(original_model: tf.keras.Model,
                          quantized_model: tf.keras.Model):
  """Helper function that copy the original model weights to quantized model."""
  original_weight_value = original_model.get_weights()
  weight_values = quantized_model.get_weights()

  original_idx = 0
  for idx, weight in enumerate(quantized_model.weights):
    if not is_quantization_weight_name(weight.name):
      if original_idx >= len(original_weight_value):
        raise ValueError('Not enought original model weights.')
      weight_values[idx] = original_weight_value[original_idx]
      original_idx = original_idx + 1

  if original_idx < len(original_weight_value):
    raise ValueError('Not enought quantized model weights.')

  quantized_model.set_weights(weight_values)


class LayerQuantizerHelper(object):
  """Helper class that handles quantizers."""

  def __init__(self, *args, **kwargs):
    self._quantizers = {}
    self._quantizer_vars = {}
    super().__init__(*args, **kwargs)

  def _all_value_quantizer(self):
    return tfmot.quantization.keras.quantizers.AllValuesQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def _moving_average_quantizer(self):
    return tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def _add_quantizer(self, name, all_value_quantizer=False):
    if all_value_quantizer:
      self._quantizers[name] = self._all_value_quantizer()
    else:
      self._quantizers[name] = self._moving_average_quantizer()

  def _apply_quantizer(self, name, inputs, training, **kwargs):
    return self._quantizers[name](
        inputs, training, self._quantizer_vars[name], **kwargs)

  def _build_quantizer_vars(self):
    for name in self._quantizers:
      self._quantizer_vars[name] = self._quantizers[name].build(
          tensor_shape=None, name=name, layer=self)


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
    if not other or not isinstance(other, NoOpActivation):
      return False

    return True

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)


def quantize_wrapped_layer(cls, quantize_config):

  def constructor(*arg, **kwargs):
    return tfmot.quantization.keras.QuantizeWrapperV2(
        cls(*arg, **kwargs), quantize_config)

  return constructor


def norm_by_activation(activation, norm_quantized, norm_no_quantized):
  if activation not in ['relu', 'relu6']:
    return norm_quantized
  else:
    return norm_no_quantized


Conv2DQuantized = quantize_wrapped_layer(
    tf.keras.layers.Conv2D,
    configs.Default8BitConvQuantizeConfig(['kernel'], ['activation'], False))
Conv2DOutputQuantized = quantize_wrapped_layer(
    tf.keras.layers.Conv2D,
    configs.Default8BitConvQuantizeConfig(['kernel'], ['activation'], True))
DepthwiseConv2DQuantized = quantize_wrapped_layer(
    tf.keras.layers.DepthwiseConv2D,
    configs.Default8BitConvQuantizeConfig(['depthwise_kernel'], ['activation'],
                                          False))
DepthwiseConv2DOutputQuantized = quantize_wrapped_layer(
    tf.keras.layers.DepthwiseConv2D,
    configs.Default8BitConvQuantizeConfig(['depthwise_kernel'], ['activation'],
                                          True))
GlobalAveragePooling2DQuantized = quantize_wrapped_layer(
    tf.keras.layers.GlobalAveragePooling2D,
    configs.Default8BitQuantizeConfig([], [], True))
AveragePooling2DQuantized = quantize_wrapped_layer(
    tf.keras.layers.AveragePooling2D,
    configs.Default8BitQuantizeConfig([], [], True))
ResizingQuantized = quantize_wrapped_layer(
    tf.keras.layers.Resizing, configs.Default8BitQuantizeConfig([], [], True))
ConcatenateQuantized = quantize_wrapped_layer(
    tf.keras.layers.Concatenate, configs.Default8BitQuantizeConfig([], [],
                                                                   True))
UpSampling2DQuantized = quantize_wrapped_layer(
    tf.keras.layers.UpSampling2D, configs.Default8BitQuantizeConfig([], [],
                                                                    True))
ReshapeQuantized = quantize_wrapped_layer(
    tf.keras.layers.Reshape, configs.Default8BitQuantizeConfig([], [], True))

# pylint:disable=g-long-lambda
BatchNormalizationQuantized = lambda norm_layer: quantize_wrapped_layer(
    norm_layer, configs.Default8BitOutputQuantizeConfig())
BatchNormalizationNoQuantized = lambda norm_layer: quantize_wrapped_layer(
    norm_layer, configs.NoOpQuantizeConfig())
