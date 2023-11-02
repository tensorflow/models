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

"""Default 8-bit QuantizeConfigs."""
from typing import Sequence, Callable, Tuple, Any, Dict

import tensorflow as tf, tf_keras
import tensorflow_model_optimization as tfmot


Quantizer = tfmot.quantization.keras.quantizers.Quantizer
Layer = tf_keras.layers.Layer
Activation = Callable[[tf.Tensor], tf.Tensor]
WeightAndQuantizer = Tuple[tf.Variable, Quantizer]
ActivationAndQuantizer = Tuple[Activation, Quantizer]


class DefaultNBitOutputQuantizeConfig(
    tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

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
        tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=self._num_bits_activation, per_axis=False,
            symmetric=False, narrow_range=False)  # activation/output
    ]

  def get_config(self) -> Dict[str, Any]:
    return {
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation,
    }


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def get_weights_and_quantizers(
      self, layer: Layer) -> Sequence[WeightAndQuantizer]:
    return []

  def get_activations_and_quantizers(
      self, layer: Layer) -> Sequence[ActivationAndQuantizer]:
    return []

  def set_quantize_weights(
      self,
      layer: Layer,
      quantize_weights: Sequence[tf.Tensor]):
    pass

  def set_quantize_activations(
      self,
      layer: Layer,
      quantize_activations: Sequence[Activation]):
    pass

  def get_output_quantizers(self, layer: Layer) -> Sequence[Quantizer]:
    return []

  def get_config(self) -> Dict[str, Any]:
    return {
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation,
    }


class DefaultNBitQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig for non recurrent Keras layers."""

  def __init__(self,
               weight_attrs: Sequence[str],
               activation_attrs: Sequence[str],
               quantize_output: bool,
               num_bits_weight: int = 8,
               num_bits_activation: int = 8):
    """Initializes a default N-bit quantize config."""
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(
        num_bits=num_bits_weight, per_axis=False,
        symmetric=True, narrow_range=True)  # weight
    self.activation_quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=num_bits_activation, per_axis=False,
        symmetric=False, narrow_range=False)  # activation/output

  def get_weights_and_quantizers(
      self, layer: Layer) -> Sequence[WeightAndQuantizer]:
    """See base class."""
    return [(getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs]

  def get_activations_and_quantizers(
      self, layer: Layer) -> Sequence[ActivationAndQuantizer]:
    """See base class."""
    return [(getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs]

  def set_quantize_weights(
      self,
      layer: Layer,
      quantize_weights: Sequence[tf.Tensor]):
    """See base class."""
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

  def set_quantize_activations(
      self,
      layer: Layer,
      quantize_activations: Sequence[Activation]):
    """See base class."""
    if len(self.activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(self.activation_attrs)))

    for activation_attr, activation in zip(
        self.activation_attrs, quantize_activations):
      setattr(layer, activation_attr, activation)

  def get_output_quantizers(self, layer: Layer) -> Sequence[Quantizer]:
    """See base class."""
    if self.quantize_output:
      return [self.activation_quantizer]
    return []

  @classmethod
  def from_config(cls, config: Dict[str, Any]) -> object:
    """Instantiates a `DefaultNBitQuantizeConfig` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `DefaultNBitQuantizeConfig` instance.
    """
    return cls(**config)

  def get_config(self) -> Dict[str, Any]:
    """Get a config for this quantize config."""
    # TODO(pulkitb): Add weight and activation quantizer to config.
    # Currently it's created internally, but ideally the quantizers should be
    # part of the constructor and passed in from the registry.
    return {
        'weight_attrs': self.weight_attrs,
        'activation_attrs': self.activation_attrs,
        'quantize_output': self.quantize_output,
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation
    }

  def __eq__(self, other):
    if not isinstance(other, DefaultNBitQuantizeConfig):
      return False

    return (self.weight_attrs == other.weight_attrs and
            self.activation_attrs == self.activation_attrs and
            self.weight_quantizer == other.weight_quantizer and
            self.activation_quantizer == other.activation_quantizer and
            self.quantize_output == other.quantize_output)

  def __ne__(self, other):
    return not self.__eq__(other)


class DefaultNBitConvWeightsQuantizer(
    tfmot.quantization.keras.quantizers.LastValueQuantizer):
  """Quantizer for handling weights in Conv2D/DepthwiseConv2D layers."""

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    """Construct LastValueQuantizer with params specific for TFLite Convs."""

    super(DefaultNBitConvWeightsQuantizer, self).__init__(
        num_bits=num_bits_weight, per_axis=True,
        symmetric=True, narrow_range=True)  # weight
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def build(self,
            tensor_shape: tf.TensorShape,
            name: str,
            layer: Layer):
    """Build min/max quantization variables."""
    min_weight = layer.add_weight(
        name + '_min',
        shape=(tensor_shape[-1],),
        initializer=tf_keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        shape=(tensor_shape[-1],),
        initializer=tf_keras.initializers.Constant(6.0),
        trainable=False)

    return {'min_var': min_weight, 'max_var': max_weight}


class NoQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
  """Dummy quantizer for explicitly not quantize."""

  def __call__(self, inputs, training, weights, **kwargs):
    return tf.identity(inputs)

  def get_config(self):
    return {}

  def build(self, tensor_shape, name, layer):
    return {}


class DefaultNBitConvQuantizeConfig(DefaultNBitQuantizeConfig):
  """QuantizeConfig for Conv2D/DepthwiseConv2D layers."""

  def __init__(self,
               weight_attrs: Sequence[str],
               activation_attrs: Sequence[str],
               quantize_output: bool,
               num_bits_weight: int = 8,
               num_bits_activation: int = 8):
    """Initializes default N-bit quantization config for the conv layer."""
    super().__init__(weight_attrs=weight_attrs,
                     activation_attrs=activation_attrs,
                     quantize_output=quantize_output,
                     num_bits_weight=num_bits_weight,
                     num_bits_activation=num_bits_activation)
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

    self.weight_quantizer = DefaultNBitConvWeightsQuantizer(
        num_bits_weight=num_bits_weight,
        num_bits_activation=num_bits_activation)


class DefaultNBitActivationQuantizeConfig(
    tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig for keras.layers.Activation.

  `keras.layers.Activation` needs a separate `QuantizeConfig` since the
  decision to quantize depends on the specific activation type.
  """

  def __init__(self, num_bits_weight: int = 8, num_bits_activation: int = 8):
    self._num_bits_weight = num_bits_weight
    self._num_bits_activation = num_bits_activation

  def _assert_activation_layer(self, layer: Layer):
    if not isinstance(layer, tf_keras.layers.Activation):
      raise RuntimeError(
          'DefaultNBitActivationQuantizeConfig can only be used with '
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
                       'DefaultNBitActivationQuantizeConfig.'.format(
                           layer.activation))

    # This code is copied from TFMOT repo, but added relu6 to support mobilenet.
    if layer.activation.__name__ in ['relu', 'relu6', 'swish']:
      # 'relu' should generally get fused into the previous layer.
      return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
          num_bits=self._num_bits_activation, per_axis=False,
          symmetric=False, narrow_range=False)]  # activation/output
    elif layer.activation.__name__ in ['linear', 'softmax', 'sigmoid']:
      return []

    raise ValueError('Activation {} not supported by '
                     'DefaultNBitActivationQuantizeConfig.'.format(
                         layer.activation))

  def get_config(self) -> Dict[str, Any]:
    """Get a config for this quantizer config."""
    return {
        'num_bits_weight': self._num_bits_weight,
        'num_bits_activation': self._num_bits_activation,
    }


def _types_dict():
  return {
      'DefaultNBitOutputQuantizeConfig':
          DefaultNBitOutputQuantizeConfig,
      'NoOpQuantizeConfig':
          NoOpQuantizeConfig,
      'DefaultNBitQuantizeConfig':
          DefaultNBitQuantizeConfig,
      'DefaultNBitConvWeightsQuantizer':
          DefaultNBitConvWeightsQuantizer,
      'DefaultNBitConvQuantizeConfig':
          DefaultNBitConvQuantizeConfig,
      'DefaultNBitActivationQuantizeConfig':
          DefaultNBitActivationQuantizeConfig,
  }
