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

"""Contains custom quantization layer transforms."""
from typing import Any, Type, Mapping, List, Union, Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from official.modeling import tf_utils
from official.projects.qat.vision.modeling.layers import nn_blocks as quantized_nn_blocks
from official.projects.qat.vision.modeling.layers import nn_layers as quantized_nn_layers
from official.projects.qat.vision.quantization import configs
from official.projects.qat.vision.quantization import helper

keras = tf.keras
LayerNode = tfmot.quantization.keras.graph_transformations.transforms.LayerNode
LayerPattern = tfmot.quantization.keras.graph_transformations.transforms.LayerPattern

_LAYER_NAMES = [
    'Vision>Conv2DBNBlock', 'Vision>InvertedBottleneckBlock',
    'Vision>SegmentationHead', 'Vision>SpatialPyramidPooling', 'Vision>ASPP'
]


class CustomLayerQuantize(
    tfmot.quantization.keras.graph_transformations.transforms.Transform):
  """Add QAT support for Keras Custom layer."""

  def __init__(self, original_layer_pattern: str,
               quantized_layer_class: Type[keras.layers.Layer]):
    super(CustomLayerQuantize, self).__init__()
    self._original_layer_pattern = original_layer_pattern
    self._quantized_layer_class = quantized_layer_class

  def pattern(self) -> LayerPattern:
    """See base class."""
    return LayerPattern(self._original_layer_pattern)

  def _create_layer_metadata(
      self, layer_class_name: str
  ) -> Mapping[str, tfmot.quantization.keras.QuantizeConfig]:
    if layer_class_name in _LAYER_NAMES:
      layer_metadata = {'quantize_config': configs.NoOpQuantizeConfig()}
    else:
      layer_metadata = {
          'quantize_config': configs.Default8BitOutputQuantizeConfig()
      }
    return layer_metadata

  def _create_dummy_input_shape(
      self, quantized_layer: tf.keras.layers.Layer
  ) -> Union[List[int], Tuple[Any, Any]]:
    dummy_input_shape = [1, 128, 128, 1]
    # SegmentationHead layer requires a tuple of 2 tensors.
    if isinstance(quantized_layer,
                  quantized_nn_layers.SegmentationHeadQuantized):
      dummy_input_shape = ([1, 1, 1, 1], [1, 1, 1, 1])
    return dummy_input_shape

  def replacement(self, match_layer: LayerNode) -> LayerNode:
    """See base class."""
    bottleneck_layer = match_layer.layer
    bottleneck_config = bottleneck_layer['config']
    bottleneck_names_and_weights = list(match_layer.names_and_weights)
    quantized_layer = self._quantized_layer_class(**bottleneck_config)
    dummy_input_shape = self._create_dummy_input_shape(quantized_layer)
    quantized_layer.compute_output_shape(dummy_input_shape)
    quantized_names_and_weights = zip(
        [weight.name for weight in quantized_layer.weights],
        quantized_layer.get_weights())
    match_idx = 0
    names_and_weights = []
    for name_and_weight in quantized_names_and_weights:
      if not helper.is_quantization_weight_name(name=name_and_weight[0]):
        name_and_weight = bottleneck_names_and_weights[match_idx]
        match_idx = match_idx + 1
      names_and_weights.append(name_and_weight)

    if match_idx != len(bottleneck_names_and_weights):
      raise ValueError('{}/{} of Bottleneck weights is transformed.'.format(
          match_idx, len(bottleneck_names_and_weights)))
    quantized_layer_config = tf_utils.serialize_layer(
        quantized_layer, use_legacy_format=True
    )
    quantized_layer_config['name'] = quantized_layer_config['config']['name']

    layer_metadata = self._create_layer_metadata(bottleneck_layer['class_name'])

    return LayerNode(
        quantized_layer_config,
        metadata=layer_metadata,
        names_and_weights=names_and_weights)


CUSTOM_TRANSFORMS = [
    CustomLayerQuantize('Vision>BottleneckBlock',
                        quantized_nn_blocks.BottleneckBlockQuantized),
    CustomLayerQuantize('Vision>InvertedBottleneckBlock',
                        quantized_nn_blocks.InvertedBottleneckBlockQuantized),
    CustomLayerQuantize('Vision>Conv2DBNBlock',
                        quantized_nn_blocks.Conv2DBNBlockQuantized),
    CustomLayerQuantize('Vision>SegmentationHead',
                        quantized_nn_layers.SegmentationHeadQuantized),
    CustomLayerQuantize('Vision>SpatialPyramidPooling',
                        quantized_nn_layers.SpatialPyramidPoolingQuantized),
    CustomLayerQuantize('Vision>ASPP', quantized_nn_layers.ASPPQuantized)
]
