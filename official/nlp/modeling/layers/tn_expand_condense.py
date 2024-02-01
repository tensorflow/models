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

"""ExpandCondense tensor network layer used in TN-BERT."""
# pylint: disable=g-classes-have-attributes
from typing import List, Optional, Text, Any, Dict
import tensorflow as tf, tf_keras

from official.modeling import tf_utils

Layer = tf_keras.layers.Layer
activations = tf_keras.activations
initializers = tf_keras.initializers


@tf_keras.utils.register_keras_serializable(package='Text')
class TNExpandCondense(Layer):
  """A TPU-optimized TensorNetwork layer.

  Designed for use in models that currently use Dense layers to achieve
  up projection followed by down projection.

  This layer is a TPU-optimized combination of 3 operations:
  Expand, Apply Activation, and Condense. The layer projects up from
  `input_shape[-1]` to `input_shape[-1] * proj_multiplier`, applies
  `self.activation`, and then condenses back to `input_shape[-1]`.

  Note the input shape and output shape will be identical.

  Args:
    proj_multiplier: Positive integer, multiple of `input_shape[-1]` to project
      up to. Must be one of `[2, 4, 6, 8]`.
    use_bias: Boolean, whether the layer uses a bias vector.
    activation: Activation function to use between Expand and Condense. If you
      don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    kernel_initializer: Initializer for the weight matrices.
    bias_initializer: Initializer for the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_shape[-1])`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., input_shape[-1])`.
  """

  def __init__(self,
               proj_multiplier: int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = 'relu',
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               **kwargs) -> None:

    # Allow specification of input_dim instead of input_shape,
    # for compatability with Keras layers that support this
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super().__init__(**kwargs)

    assert proj_multiplier in [
        2, 4, 6, 8, 10, 12
    ], 'proj_multiplier needs to be one of [2, 4, 6, 8, 10, 12]'
    self.proj_multiplier = proj_multiplier

    self.use_bias = use_bias
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape: List[int]) -> None:
    # Disable the attribute-defined-outside-init violations in this function
    # pylint: disable=attribute-defined-outside-init
    if input_shape[-1] is None:
      raise ValueError(
          'The last dimension of the inputs to `TNExpandCondense` '
          'should be defined. Found `None`.')

    super().build(input_shape)

    self.proj_size = self.proj_multiplier * input_shape[-1]

    assert (self.proj_size // input_shape[-1]) * input_shape[
        -1] == self.proj_size, (f'{self.proj_size} / {input_shape[-1]} must be '
                                f'round')
    assert (input_shape[-1] // 128
           ) * 128 == input_shape[-1], f'{input_shape[-1]} / 128 must be round'

    self.w1 = self.add_weight(
        name='w1',
        shape=(input_shape[-1], input_shape[-1]),
        trainable=True,
        initializer=tf_utils.clone_initializer(self.kernel_initializer))

    self.w2 = self.add_weight(
        name='w2',
        shape=(128, (128 * (self.proj_size // input_shape[-1]))),
        trainable=True,
        initializer=tf_utils.clone_initializer(self.kernel_initializer))

    self.w3 = self.add_weight(
        name='w3',
        shape=(128 * (self.proj_size // input_shape[-1]), 128),
        trainable=True,
        initializer=tf_utils.clone_initializer(self.kernel_initializer))
    self.w4 = self.add_weight(
        name='w4',
        shape=(input_shape[-1] // 128, 128, input_shape[-1]),
        trainable=True,
        initializer=tf_utils.clone_initializer(self.kernel_initializer))

    if self.use_bias:
      self.bias = self.add_weight(
          name='b',
          shape=(input_shape[-1] // 128, 1,
                 128 * (self.proj_size // input_shape[-1])),
          trainable=True,
          initializer=self.bias_initializer)
    else:
      self.bias = None

  def call(self, inputs: tf.Tensor, **kwargs):
    orig_shape = tf.shape(inputs)
    input_dim = inputs.shape[-1]
    tmp = tf.reshape(inputs, (-1, input_dim))
    # Shape is (BatchSeq, input_dim)

    # Expansion network
    tmp = tf.einsum('ab,Qb->aQ', self.w1, tmp)
    # Note: Letter Q will always represent the BatchSeq axis.
    tmp = tf.reshape(tmp, (input_dim // 128, 128, -1))
    tmp = tf.einsum('abQ,bd->aQd', tmp, self.w2)

    # Apply activation and then Condense
    tmp = self.activation(tmp + self.bias)
    tmp = tf.einsum('aQd,db->aQb', tmp, self.w3)
    tmp = tf.einsum('aQb,abd->Qd', tmp, self.w4)

    out = tf.reshape(tmp, orig_shape)
    return out

  def compute_output_shape(self, input_shape: List[int]) -> List[int]:
    return input_shape

  def get_config(self) -> Dict[Any, Any]:
    """Returns the config of the layer.

    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      Python dictionary containing the configuration of the layer.
    """
    config = {}

    # Include the layer-specific arguments
    args = ['proj_multiplier', 'use_bias']
    for arg in args:
      config[arg] = getattr(self, arg)

    # Serialize the activation
    config['activation'] = activations.serialize(getattr(self, 'activation'))

    # Serialize the initializers
    decomp_initializers = ['kernel_initializer', 'bias_initializer']
    for initializer_arg in decomp_initializers:
      config[initializer_arg] = initializers.serialize(
          getattr(self, initializer_arg))

    # Get base config
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
