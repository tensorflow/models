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

"""Quantization schemes."""
from typing import Type

# Import libraries

import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.projects.qat.vision.modeling.layers import nn_blocks as quantized_nn_blocks
from official.projects.qat.vision.modeling.layers import nn_layers as quantized_nn_layers
from official.projects.qat.vision.quantization import configs


keras = tf.keras
default_8bit_transforms = tfmot.quantization.keras.default_8bit.default_8bit_transforms
LayerNode = tfmot.quantization.keras.graph_transformations.transforms.LayerNode
LayerPattern = tfmot.quantization.keras.graph_transformations.transforms.LayerPattern

_QUANTIZATION_WEIGHT_NAMES = [
    'output_max', 'output_min', 'optimizer_step',
    'kernel_min', 'kernel_max',
    'add_three_min', 'add_three_max',
    'divide_six_min', 'divide_six_max',
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
               quantized_layer_class: Type[keras.layers.Layer]):
    super(CustomLayerQuantize, self).__init__()
    self._original_layer_pattern = original_layer_pattern
    self._quantized_layer_class = quantized_layer_class

  def pattern(self) -> LayerPattern:
    """See base class."""
    return LayerPattern(self._original_layer_pattern)

  def _is_quantization_weight_name(self, name):
    simple_name = name.split('/')[-1].split(':')[0]
    if simple_name in _QUANTIZATION_WEIGHT_NAMES:
      return True
    if simple_name in _ORIGINAL_WEIGHT_NAME:
      return False
    raise ValueError('Variable name {} is not supported on '
                     'CustomLayerQuantize({}) transform.'.format(
                         simple_name,
                         self._original_layer_pattern))

  def replacement(self, match_layer: LayerNode) -> LayerNode:
    """See base class."""
    bottleneck_layer = match_layer.layer
    bottleneck_config = bottleneck_layer['config']
    bottleneck_names_and_weights = list(match_layer.names_and_weights)
    quantized_layer = self._quantized_layer_class(
        **bottleneck_config)
    dummy_input_shape = [1, 64, 128, 1]
    # SegmentationHead layer requires a tuple of 2 tensors.
    if isinstance(quantized_layer,
                  quantized_nn_layers.SegmentationHeadQuantized):
      dummy_input_shape = ([1, 1, 1, 1], [1, 1, 1, 1])
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
    if bottleneck_layer['class_name'] in [
        'Vision>Conv2DBNBlock', 'Vision>InvertedBottleneckBlock',
        'Vision>SegmentationHead', 'Vision>SpatialPyramidPooling',
        'Vision>ASPP'
    ]:
      layer_metadata = {'quantize_config': configs.NoOpQuantizeConfig()}
    else:
      layer_metadata = {
          'quantize_config': configs.Default8BitOutputQuantizeConfig()
      }

    return LayerNode(
        quantized_layer_config,
        metadata=layer_metadata,
        names_and_weights=names_and_weights)


class QuantizeLayoutTransform(
    tfmot.quantization.keras.QuantizeLayoutTransform):
  """Default model transformations."""

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
        default_8bit_transforms.InputLayerQuantize(),
        default_8bit_transforms.SeparableConv1DQuantize(),
        default_8bit_transforms.SeparableConvQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormReLUQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormActivationQuantize(),
        default_8bit_transforms.Conv2DBatchNormReLUQuantize(),
        default_8bit_transforms.Conv2DBatchNormActivationQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormQuantize(),
        default_8bit_transforms.Conv2DBatchNormQuantize(),
        default_8bit_transforms.ConcatTransform6Inputs(),
        default_8bit_transforms.ConcatTransform5Inputs(),
        default_8bit_transforms.ConcatTransform4Inputs(),
        default_8bit_transforms.ConcatTransform3Inputs(),
        default_8bit_transforms.ConcatTransform(),
        default_8bit_transforms.LayerReLUQuantize(),
        default_8bit_transforms.LayerReluActivationQuantize(),
        CustomLayerQuantize('Vision>BottleneckBlock',
                            quantized_nn_blocks.BottleneckBlockQuantized),
        CustomLayerQuantize(
            'Vision>InvertedBottleneckBlock',
            quantized_nn_blocks.InvertedBottleneckBlockQuantized),
        CustomLayerQuantize('Vision>Conv2DBNBlock',
                            quantized_nn_blocks.Conv2DBNBlockQuantized),
        CustomLayerQuantize('Vision>SegmentationHead',
                            quantized_nn_layers.SegmentationHeadQuantized),
        CustomLayerQuantize('Vision>SpatialPyramidPooling',
                            quantized_nn_layers.SpatialPyramidPoolingQuantized),
        CustomLayerQuantize('Vision>ASPP', quantized_nn_layers.ASPPQuantized)
    ]
    return tfmot.quantization.keras.graph_transformations.model_transformer.ModelTransformer(
        model, transforms,
        set(layer_quantize_map.keys()), layer_quantize_map).transform()


class Default8BitQuantizeScheme(
    tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme):

  def get_layout_transformer(self):
    return QuantizeLayoutTransform()
