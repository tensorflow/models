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

"""Quantization helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Type, Union

import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official.projects.qat.vision.quantization import configs


_QUANTIZATION_WEIGHT_NAMES = [
    'output_max',
    'output_min',
    'optimizer_step',
    'kernel_min',
    'kernel_max',
    'add_three_min',
    'add_three_max',
    'divide_six_min',
    'divide_six_max',
    'depthwise_kernel_min',
    'depthwise_kernel_max',
    'pointwise_kernel_min',
    'pointwise_kernel_max',
    'reduce_mean_quantizer_vars_min',
    'reduce_mean_quantizer_vars_max',
    'quantize_layer_min',
    'quantize_layer_max',
    'quantize_layer_1_min',
    'quantize_layer_1_max',
    'quantize_layer_2_min',
    'quantize_layer_2_max',
    'quantize_layer_3_min',
    'quantize_layer_3_max',
    'post_activation_min',
    'post_activation_max',
]

_ORIGINAL_WEIGHT_NAME = [
    'kernel',
    'depthwise_kernel',
    'pointwise_kernel',
    'gamma',
    'beta',
    'moving_mean',
    'moving_variance',
    'bias',
]


def is_quantization_weight_name(name: str) -> bool:
  simple_name = name.split('/')[-1].split(':')[0]
  if simple_name in _QUANTIZATION_WEIGHT_NAMES:
    return True
  if simple_name in _ORIGINAL_WEIGHT_NAME:
    return False
  raise ValueError('Variable name {} is not supported.'.format(simple_name))


def copy_original_weights(original_model: tf_keras.Model,
                          quantized_model: tf_keras.Model):
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


class SeparableConv2DQuantized(tf_keras.layers.Layer):
  """Quantized SeperableConv2D."""

  def __init__(
      self,
      name: Optional[str] = None,
      last_quantize: bool = False,
      **conv_kwargs,
  ):
    """Initializes a SeparableConv2DQuantized.

    Args:
      name: The name of the layer.
      last_quantize: A `bool` indicates whether add quantization for the output.
      **conv_kwargs: A keyword arguments to be used for conv and dwconv.
    """

    super().__init__(name=name)
    self._conv_kwargs = copy.deepcopy(conv_kwargs)
    self._name = name
    self._last_quantize = last_quantize

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the child layers of the layer."""
    depthwise_conv2d_quantized = quantize_wrapped_layer(
        tf_keras.layers.DepthwiseConv2D,
        configs.Default8BitConvQuantizeConfig(['depthwise_kernel'], [], True),
    )
    conv2d_quantized = quantize_wrapped_layer(
        tf_keras.layers.Conv2D,
        configs.Default8BitConvQuantizeConfig(
            ['kernel'], [], self._last_quantize
        ),
    )

    dwconv_kwargs = self._conv_kwargs.copy()
    # Depthwise conv input filters is always equal to output filters.
    # This filters argument only needed for the point-wise conv2d op.
    del dwconv_kwargs['filters']
    dwconv_kwargs.update({
        'activation': None,
        'use_bias': False,
    })
    self.dw_conv = depthwise_conv2d_quantized(name='dw', **dwconv_kwargs)

    conv_kwargs = self._conv_kwargs.copy()
    conv_kwargs.update({
        'kernel_size': (1, 1),
        'strides': (1, 1),
        'padding': 'valid',
        'groups': 1,
    })

    self.conv = conv2d_quantized(name='pw', **conv_kwargs)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Call the separable conv layer."""
    x = self.dw_conv(inputs)
    outputs = self.conv(x)
    return outputs

  def get_config(self) -> Dict[str, Any]:
    """Returns the config of the layer."""
    config = self._conv_kwargs.copy()
    config.update({
        'name': self._name,
        'last_quantize': self._last_quantize,
    })
    return config

  @classmethod
  def from_config(
      cls: Type[SeparableConv2DQuantized], config: Dict[str, Any]
  ) -> SeparableConv2DQuantized:
    """Creates a layer from its config."""
    return cls(**config)


Conv2DQuantized = quantize_wrapped_layer(
    tf_keras.layers.Conv2D,
    configs.Default8BitConvQuantizeConfig(['kernel'], ['activation'], False))
Conv2DOutputQuantized = quantize_wrapped_layer(
    tf_keras.layers.Conv2D,
    configs.Default8BitConvQuantizeConfig(['kernel'], ['activation'], True))
DepthwiseConv2DQuantized = quantize_wrapped_layer(
    tf_keras.layers.DepthwiseConv2D,
    configs.Default8BitConvQuantizeConfig(['depthwise_kernel'], ['activation'],
                                          False))
DepthwiseConv2DOutputQuantized = quantize_wrapped_layer(
    tf_keras.layers.DepthwiseConv2D,
    configs.Default8BitConvQuantizeConfig(['depthwise_kernel'], ['activation'],
                                          True))
GlobalAveragePooling2DQuantized = quantize_wrapped_layer(
    tf_keras.layers.GlobalAveragePooling2D,
    configs.Default8BitQuantizeConfig([], [], True))
AveragePooling2DQuantized = quantize_wrapped_layer(
    tf_keras.layers.AveragePooling2D,
    configs.Default8BitQuantizeConfig([], [], True))
ResizingQuantized = quantize_wrapped_layer(
    tf_keras.layers.Resizing, configs.Default8BitQuantizeConfig([], [], True))
ConcatenateQuantized = quantize_wrapped_layer(
    tf_keras.layers.Concatenate, configs.Default8BitQuantizeConfig([], [],
                                                                   True))
UpSampling2DQuantized = quantize_wrapped_layer(
    tf_keras.layers.UpSampling2D, configs.Default8BitQuantizeConfig([], [],
                                                                    True))
ReshapeQuantized = quantize_wrapped_layer(
    tf_keras.layers.Reshape, configs.Default8BitQuantizeConfig([], [], True))
DenseQuantized = quantize_wrapped_layer(
    tf_keras.layers.Dense,
    configs.Default8BitQuantizeConfig(['kernel'], ['activation'], False),
)
DenseOutputQuantized = quantize_wrapped_layer(
    tf_keras.layers.Dense,
    configs.Default8BitQuantizeConfig(['kernel'], ['activation'], True),
)
IdentityQuantized = quantize_wrapped_layer(
    tf_keras.layers.Identity, configs.Default8BitQuantizeConfig([], [], True)
)

# pylint:disable=g-long-lambda
BatchNormalizationQuantized = lambda norm_layer: quantize_wrapped_layer(
    norm_layer, configs.Default8BitOutputQuantizeConfig())
BatchNormalizationNoQuantized = lambda norm_layer: quantize_wrapped_layer(
    norm_layer, configs.NoOpQuantizeConfig())
