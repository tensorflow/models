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

"""Custom quantize configs."""
from typing import Sequence, Callable, Tuple, Any, Dict

import tensorflow as tf, tf_keras
import tensorflow_model_optimization as tfmot


Quantizer = tfmot.quantization.keras.quantizers.Quantizer
Layer = tf_keras.layers.Layer
Activation = Callable[[tf.Tensor], tf.Tensor]
WeightAndQuantizer = Tuple[tf.Variable, Quantizer]
ActivationAndQuantizer = Tuple[Activation, Quantizer]


class _QuantizeHelper(object):
  """Mixin with helper functions for quantizers."""

  def _add_range_weights(self, layer, name, per_axis=False, tensor_shape=None):
    """Add min and max vars to layer."""
    # Added naming index to avoid duplicated.
    if hasattr(layer, 'quantize_helper_weight_idx'):
      layer.quantize_helper_weight_idx += 1
      name = '{}/{}'.format(layer.quantize_helper_weight_idx, name)
    else:
      layer.quantize_helper_weight_idx = 0

    shape = None
    if per_axis and tensor_shape is not None:
      shape = (tensor_shape[-1])

    min_weight = layer.add_weight(
        name + '_min',
        initializer=tf_keras.initializers.Constant(-6.0),
        trainable=False,
        shape=shape)
    max_weight = layer.add_weight(
        name + '_max',
        initializer=tf_keras.initializers.Constant(6.0),
        trainable=False,
        shape=shape)

    return {'min_var': min_weight, 'max_var': max_weight}


class LastValueQuantizer(
    _QuantizeHelper,
    tfmot.quantization.keras.quantizers.LastValueQuantizer):
  pass


class MovingAverageQuantizer(
    _QuantizeHelper,
    tfmot.quantization.keras.quantizers.MovingAverageQuantizer):
  pass


class NoQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
  """Dummy quantizer do nothing."""

  def __call__(self, inputs, training, weights, **kwargs):
    return tf.identity(inputs)

  def get_config(self):
    return {}

  def build(self, tensor_shape, name, layer):
    return {}

  def __eq__(self, other):
    if not isinstance(other, NoQuantizer):
      return False

    return True

  def __ne__(self, other):
    return not self.__eq__(other)


class DefaultEinsumDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig for EinsumDense layer."""

  # Configure how to quantize weights.
  def get_weights_and_quantizers(self, layer):
    return [(layer.kernel, LastValueQuantizer(
        num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

  # Configure how to quantize activations.
  def get_activations_and_quantizers(self, layer):
    return [(layer.activation, MovingAverageQuantizer(
        num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

  def set_quantize_weights(self, layer, quantize_weights):
    # Add this line for each item returned in `get_weights_and_quantizers`
    # , in the same order
    layer.kernel = quantize_weights[0]

  def set_quantize_activations(self, layer, quantize_activations):
    # Add this line for each item returned in `get_activations_and_quantizers`
    # , in the same order.
    layer.activation = quantize_activations[0]

  # Configure how to quantize outputs (may be equivalent to activations).
  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


# pylint: disable=protected-access
class DefaultMultiHeadAttentionQuantizeConfig(
    tfmot.quantization.keras.QuantizeConfig):
  """Default quantize config for MultiHeadAttention layer.

   It only quantize child EinsumDense layers. It should be applied to
  MultiHeadAttentionQuantized layer.
  """

  def __init__(self):
    self.einsum_dense_config = DefaultEinsumDenseQuantizeConfig()
    self.num_weight_per_einsum_dense = 1
    self.num_activation_per_einsum_dense = 1

  def _get_einsum_dense_layers(self, layer):
    return [
        layer._query_dense,
        layer._key_dense,
        layer._value_dense,
        layer._output_dense]

  def get_weights_and_quantizers(self, layer):
    ret = []
    for einsum_dense_layer in self._get_einsum_dense_layers(layer):
      ret += self.einsum_dense_config.get_weights_and_quantizers(
          einsum_dense_layer)
    return ret

  def get_activations_and_quantizers(self, layer):
    ret = []
    for einsum_dense_layer in self._get_einsum_dense_layers(layer):
      ret += self.einsum_dense_config.get_activations_and_quantizers(
          einsum_dense_layer)
    return ret

  def set_quantize_weights(self, layer, quantize_weights):
    idx = 0
    for einsum_dense_layer in self._get_einsum_dense_layers(layer):
      self.einsum_dense_config.set_quantize_weights(
          einsum_dense_layer,
          quantize_weights[idx:idx+self.num_weight_per_einsum_dense])
      idx += self.num_weight_per_einsum_dense

  def set_quantize_activations(self, layer, quantize_activations):
    idx = 0
    for einsum_dense_layer in self._get_einsum_dense_layers(layer):
      self.einsum_dense_config.set_quantize_activations(
          einsum_dense_layer,
          quantize_activations[idx:idx+self.num_activation_per_einsum_dense])
      idx += self.num_activation_per_einsum_dense

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}
# pylint: enable=protected-access


class Default8BitOutputQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def get_weights_and_quantizers(
      self, layer: Layer) -> Sequence[WeightAndQuantizer]:
    return []

  def get_activations_and_quantizers(
      self, layer: Layer) -> Sequence[ActivationAndQuantizer]:
    return []

  def set_quantize_weights(self,
                           layer: Layer,
                           quantize_weights: Sequence[tf.Tensor]):
    pass

  def set_quantize_activations(self,
                               layer: Layer,
                               quantize_activations: Sequence[Activation]):
    pass

  def get_output_quantizers(self, layer: Layer) -> Sequence[Quantizer]:
    return [
        MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False)
    ]

  def get_config(self) -> Dict[str, Any]:
    return {}


class Default8BitActivationQuantizeConfig(
    tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig for keras.layers.Activation.

  `keras.layers.Activation` needs a separate `QuantizeConfig` since the
  decision to quantize depends on the specific activation type.
  """

  def _assert_activation_layer(self, layer: Layer):
    if not isinstance(layer, tf_keras.layers.Activation):
      raise RuntimeError(
          'Default8BitActivationQuantizeConfig can only be used with '
          '`keras.layers.Activation`.')

  def get_weights_and_quantizers(
      self, layer: Layer) -> Sequence[WeightAndQuantizer]:
    """See base class."""
    self._assert_activation_layer(layer)
    return []

  def get_activations_and_quantizers(
      self, layer: Layer) -> Sequence[ActivationAndQuantizer]:
    """See base class."""
    self._assert_activation_layer(layer)
    return []

  def set_quantize_weights(
      self,
      layer: Layer,
      quantize_weights: Sequence[tf.Tensor]):
    """See base class."""
    self._assert_activation_layer(layer)

  def set_quantize_activations(
      self,
      layer: Layer,
      quantize_activations: Sequence[Activation]):
    """See base class."""
    self._assert_activation_layer(layer)

  def get_output_quantizers(self, layer: Layer) -> Sequence[Quantizer]:
    """See base class."""
    self._assert_activation_layer(layer)

    if not hasattr(layer.activation, '__name__'):
      raise ValueError('Activation {} not supported by '
                       'Default8BitActivationQuantizeConfig.'.format(
                           layer.activation))

    # This code is copied from TFMOT repo, but added relu6 to support mobilenet.
    if layer.activation.__name__ in ['relu', 'relu6']:
      # 'relu' should generally get fused into the previous layer.
      return [MovingAverageQuantizer(
          num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]
    elif layer.activation.__name__ in ['linear', 'softmax', 'sigmoid']:
      return []

    raise ValueError('Activation {} not supported by '
                     'Default8BitActivationQuantizeConfig.'.format(
                         layer.activation))

  def get_config(self) -> Dict[str, Any]:
    """Get a config for this quantizer config."""
    return {}


class NoQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """Empty quantize config."""

  # Configure how to quantize weights.
  def get_weights_and_quantizers(self, layer):
    return []

  # Configure how to quantize activations.
  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    # Add this line for each item returned in `get_weights_and_quantizers`
    # , in the same order
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    # Add this line for each item returned in `get_activations_and_quantizers`
    # , in the same order.
    pass

  # Configure how to quantize outputs (may be equivalent to activations).
  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


class Default8BitQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig for non recurrent Keras layers."""

  def __init__(self, weight_attrs, activation_attrs, quantize_output):
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = LastValueQuantizer(
        num_bits=8, per_axis=False, symmetric=True, narrow_range=True)
    self.activation_quantizer = MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def get_weights_and_quantizers(self, layer):
    return [(getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs]

  def get_activations_and_quantizers(self, layer):
    return [(getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs]

  def set_quantize_weights(self, layer, quantize_weights):
    if len(self.weight_attrs) != len(quantize_weights):
      raise ValueError(
          '`set_quantize_weights` called on layer {} with {} '
          'weight parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_weights), len(self.weight_attrs)))

    for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
      current_weight = getattr(layer, weight_attr)
      if current_weight.shape != weight.shape:
        raise ValueError('Existing layer weight shape {} is incompatible with'
                         'provided weight shape {}'.format(
                             current_weight.shape, weight.shape))

      setattr(layer, weight_attr, weight)

  def set_quantize_activations(self, layer, quantize_activations):
    if len(self.activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(self.activation_attrs)))

    for activation_attr, activation in zip(
        self.activation_attrs, quantize_activations):
      setattr(layer, activation_attr, activation)

  def get_output_quantizers(self, layer):
    if self.quantize_output:
      return [self.activation_quantizer]
    return []

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Default8BitQuantizeConfig` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Default8BitQuantizeConfig` instance.
    """
    return cls(**config)

  def get_config(self):
    # TODO(pulkitb): Add weight and activation quantizer to config.
    # Currently it's created internally, but ideally the quantizers should be
    # part of the constructor and passed in from the registry.
    return {
        'weight_attrs': self.weight_attrs,
        'activation_attrs': self.activation_attrs,
        'quantize_output': self.quantize_output
    }

  def __eq__(self, other):
    if not isinstance(other, Default8BitQuantizeConfig):
      return False

    return (self.weight_attrs == other.weight_attrs and
            self.activation_attrs == self.activation_attrs and
            self.weight_quantizer == other.weight_quantizer and
            self.activation_quantizer == other.activation_quantizer and
            self.quantize_output == other.quantize_output)

  def __ne__(self, other):
    return not self.__eq__(other)


def _types_dict():
  return {
      'NoQuantizer':
          NoQuantizer,
      'LastValueQuantizer':
          LastValueQuantizer,
      'MovingAverageQuantizer':
          MovingAverageQuantizer,
      'DefaultEinsumDenseQuantizeConfig':
          DefaultEinsumDenseQuantizeConfig,
      'DefaultMultiHeadAttentionQuantizeConfig':
          DefaultMultiHeadAttentionQuantizeConfig,
      'Default8BitOutputQuantizeConfig':
          Default8BitOutputQuantizeConfig,
      'Default8BitActivationQuantizeConfig':
          Default8BitActivationQuantizeConfig,
      'NoQuantizeConfig':
          NoQuantizeConfig,
      'Default8BitQuantizeConfig':
          Default8BitQuantizeConfig,
  }
