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

"""Quantization schemes."""
from typing import Type

# Import libraries

import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official.projects.qat.vision.n_bit import configs
from official.projects.qat.vision.n_bit import nn_blocks

keras = tf.keras
default_n_bit_transforms = tfmot.quantization.keras.experimental.default_n_bit.default_n_bit_transforms
_LayerNode = tfmot.quantization.keras.graph_transformations.transforms.LayerNode
_LayerPattern = tfmot.quantization.keras.graph_transformations.transforms.LayerPattern
_ModelTransformer = tfmot.quantization.keras.graph_transformations.model_transformer.ModelTransformer

_QUANTIZATION_WEIGHT_NAMES = [
    'output_max', 'output_min', 'optimizer_step',
    'kernel_min', 'kernel_max',
    'depthwise_kernel_min', 'depthwise_kernel_max',
    'reduce_mean_quantizer_vars_min', 'reduce_mean_quantizer_vars_max']

_ORIGINAL_WEIGHT_NAME = [
    'kernel', 'depthwise_kernel',
    'gamma', 'beta', 'moving_mean', 'moving_variance',
    'bias']


class CustomLayerQuantize(
    tfmot.quantization.keras.graph_transformations.transforms.Transform):
  """Add QAT support for Keras Custom layer."""

  def __init__(self,
               original_layer_pattern: str,
               quantized_layer_class: Type[keras.layers.Layer],
               num_bits_weight: int = 8,
               num_bits_activation: int = 8):
    super().__init__()
    self._original_layer_pattern = original_layer_pattern
    self._quantized_layer_class = quantized_layer_class
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def pattern(self) -> _LayerPattern:
    """See base class."""
    return _LayerPattern(self._original_layer_pattern)

  def _is_quantization_weight_name(self, name):
    simple_name = name.split('/')[-1].split(':')[0]
    if simple_name in _QUANTIZATION_WEIGHT_NAMES:
      return True
    if simple_name in _ORIGINAL_WEIGHT_NAME:
      return False
    raise ValueError(f'Variable name {simple_name} is not supported on '
                     'CustomLayerQuantize({self._original_layer_pattern}) '
                     'transform.')

  def replacement(self, match_layer: _LayerNode) -> _LayerNode:
    """See base class."""
    bottleneck_layer = match_layer.layer
    bottleneck_config = bottleneck_layer['config']
    bottleneck_config['num_bits_weight'] = self._num_bits_weight
    bottleneck_config['num_bits_activation'] = self._num_bits_activation
    bottleneck_names_and_weights = list(match_layer.names_and_weights)
    quantized_layer = self._quantized_layer_class(
        **bottleneck_config)
    dummy_input_shape = [1, 1, 1, 1]
    quantized_layer.compute_output_shape(dummy_input_shape)
    quantized_names_and_weights = zip(
        [weight.name for weight in quantized_layer.weights],
        quantized_layer.get_weights())
    match_idx = 0
    names_and_weights = []
    for name_and_weight in quantized_names_and_weights:
      if not self._is_quantization_weight_name(name=name_and_weight[0]):
        name_and_weight = bottleneck_names_and_weights[match_idx]
        match_idx = match_idx + 1
      names_and_weights.append(name_and_weight)

    if match_idx != len(bottleneck_names_and_weights):
      raise ValueError('{}/{} of Bottleneck weights is transformed.'.format(
          match_idx, len(bottleneck_names_and_weights)))
    quantized_layer_config = keras.layers.serialize(quantized_layer)
    quantized_layer_config['name'] = quantized_layer_config['config']['name']
    layer_metadata = {
        'quantize_config':
            configs.DefaultNBitOutputQuantizeConfig(
                num_bits_weight=self._num_bits_weight,
                num_bits_activation=self._num_bits_activation)}

    return _LayerNode(
        quantized_layer_config,
        metadata=layer_metadata,
        names_and_weights=names_and_weights)


class QuantizeLayoutTransform(
    tfmot.quantization.keras.QuantizeLayoutTransform):
  """Default model transformations."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def apply(self, model, layer_quantize_map):
    """Implement default 8-bit transforms.

    Currently this means the following.
      1. Pull activations into layers, and apply fuse activations. (TODO)
      2. Modify range in incoming layers for Concat. (TODO)
      3. Fuse Conv2D/DepthwiseConv2D + BN into single layer.

    Args:
      model: Keras model to be quantized.
      layer_quantize_map: Map with keys as layer names, and values as dicts
        containing custom `QuantizeConfig`s which may have been passed with
        layers.

    Returns:
      (Transformed Keras model to better match TensorFlow Lite backend, updated
      layer quantize map.)
    """

    transforms = [
        default_n_bit_transforms.InputLayerQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.SeparableConv1DQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.SeparableConvQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DReshapeBatchNormReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DReshapeBatchNormActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DBatchNormReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DBatchNormActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DReshapeBatchNormQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.Conv2DBatchNormQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform6Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform5Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform4Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform3Inputs(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.ConcatTransform(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.LayerReLUQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        default_n_bit_transforms.LayerReluActivationQuantize(
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        CustomLayerQuantize(
            'Vision>BottleneckBlock',
            nn_blocks.BottleneckBlockNBitQuantized,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        CustomLayerQuantize(
            'Vision>InvertedBottleneckBlock',
            nn_blocks.InvertedBottleneckBlockNBitQuantized,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation),
        CustomLayerQuantize(
            'Vision>Conv2DBNBlock',
            nn_blocks.Conv2DBNBlockNBitQuantized,
            num_bits_weight=self._num_bits_weight,
            num_bits_activation=self._num_bits_activation)
    ]
    return _ModelTransformer(model, transforms, set(layer_quantize_map.keys()),
                             layer_quantize_map).transform()


class DefaultNBitQuantizeScheme(tfmot.quantization.keras.experimental
                                .default_n_bit.DefaultNBitQuantizeScheme):
  """Default N-bit Scheme."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    super(DefaultNBitQuantizeScheme, self).__init__(
        num_bits_weight=num_bits_weight,
        num_bits_activation=num_bits_activation)
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def get_layout_transformer(self):
    return QuantizeLayoutTransform(
        num_bits_weight=self._num_bits_weight,
        num_bits_activation=self._num_bits_activation)

