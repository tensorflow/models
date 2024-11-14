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

"""Quantization schemes."""
import numpy as np
import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot

from official.modeling import tf_utils
from official.projects.qat.nlp.modeling.layers import mobile_bert_layers
from official.projects.qat.nlp.modeling.layers import transformer_encoder_block
from official.projects.qat.nlp.quantization import configs

keras = tf.keras
default_8bit_transforms = tfmot.quantization.keras.default_8bit.default_8bit_transforms
LayerNode = tfmot.quantization.keras.graph_transformations.transforms.LayerNode
LayerPattern = tfmot.quantization.keras.graph_transformations.transforms.LayerPattern


class TransformerEncoderBlockQuantize(
    tfmot.quantization.keras.graph_transformations.transforms.Transform):
  """Add QAT support for Keras Custom layer."""

  _QUANTIZATION_AWARE_TRAINING_WEIGHT_NAMES = frozenset({
      'optimizer_step',
      'output_max', 'output_min',
      'kernel_min', 'kernel_max',
      'depthwise_kernel_min', 'depthwise_kernel_max',
      'query_min', 'query_max',
      'attention_scores_min', 'attention_scores_max',
      'attention_output_min', 'attention_output_max',
      'masked_softmax_attention_mask_min',
      'masked_softmax_attention_mask_max',
      'masked_softmax_sub1_min', 'masked_softmax_sub1_max',
      'masked_softmax_mask1_min', 'masked_softmax_mask1_max',
      'masked_softmax_sub2_min', 'masked_softmax_sub2_max',
      'masked_softmax_clamp_min', 'masked_softmax_clamp_max',
      'masked_softmax_mask2_min', 'masked_softmax_mask2_max',
      'masked_softmax_adder_sub_min', 'masked_softmax_adder_sub_max',
      'masked_softmax_adder_mul_min', 'masked_softmax_adder_mul_max',
      'masked_softmax_add_min', 'masked_softmax_add_max',
      'post_activation_min', 'post_activation_max',
      'word_embedding_out_min', 'word_embedding_out_max',
      'pos_embedding_out_min', 'pos_embedding_out_max',
      'type_embedding_out_min', 'type_embedding_out_max',
      'bias_min', 'bias_max'
  })

  _SUPPOTRED_MODEL_WEIGHT_NAMES = frozenset({
      'kernel', 'depthwise_kernel', 'bias',
      'gamma', 'beta', 'moving_mean', 'moving_variance',
      'embeddings'
  })

  def __init__(self):
    super().__init__()
    self._original_layer_pattern = 'modeling>TransformerEncoderBlock'
    self._quantized_layer_class = transformer_encoder_block.TransformerEncoderBlockQuantized

  def pattern(self) -> LayerPattern:
    """See base class."""
    return LayerPattern(self._original_layer_pattern)

  def _is_quantization_weight_name(self, name):
    simple_name = name.split('/')[-1].split(':')[0]
    if simple_name in self._QUANTIZATION_AWARE_TRAINING_WEIGHT_NAMES:
      return True
    if simple_name in self._SUPPOTRED_MODEL_WEIGHT_NAMES:
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

    quantized_layer_config = quantized_layer.get_config()
    if 'hidden_size' in quantized_layer_config:
      dummy_input_shape = [
          1, 1, quantized_layer_config['hidden_size']]
      quantized_layer.compute_output_shape(dummy_input_shape)
    elif 'num_attention_heads' in quantized_layer_config:
      dummy_input_shape = [
          1, 1, quantized_layer_config['num_attention_heads']]
      quantized_layer.compute_output_shape(dummy_input_shape)
    else:
      dummy_input_shape = [1, 1]
      quantized_layer(np.zeros(shape=dummy_input_shape, dtype=np.int32),
                      np.zeros(shape=dummy_input_shape, dtype=np.int32),
                      training=False)

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
    quantized_layer_config = tf_utils.serialize_layer(
        quantized_layer, use_legacy_format=True
    )
    quantized_layer_config['name'] = quantized_layer_config['config']['name']
    layer_metadata = {
        'quantize_config':
            configs.NoQuantizeConfig()}

    return LayerNode(
        quantized_layer_config,
        metadata=layer_metadata,
        names_and_weights=names_and_weights)


class MobileBertTransformerQuantize(TransformerEncoderBlockQuantize):

  def __init__(self):
    super().__init__()
    self._original_layer_pattern = 'Text>MobileBertTransformer'
    self._quantized_layer_class = mobile_bert_layers.MobileBertTransformerQuantized


class MobileBertEmbeddingQuantize(TransformerEncoderBlockQuantize):

  def __init__(self):
    super().__init__()
    self._original_layer_pattern = 'Text>MobileBertEmbedding'
    self._quantized_layer_class = mobile_bert_layers.MobileBertEmbeddingQuantized


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
        TransformerEncoderBlockQuantize(),
        MobileBertTransformerQuantize(),
        MobileBertEmbeddingQuantize(),
    ]
    return tfmot.quantization.keras.graph_transformations.model_transformer.ModelTransformer(
        model, transforms,
        set(layer_quantize_map.keys()), layer_quantize_map).transform()


class Default8BitQuantizeScheme(
    tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme):

  def get_layout_transformer(self):
    return QuantizeLayoutTransform()
