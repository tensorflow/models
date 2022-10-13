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

"""Keras-based Scaffold TransformerEncoder block for vision models.

This implementation is subclassed from NLP TransformerScaffold to support
customized `attention_layer` and `feedforward_layer`. In addition, this
implementation has a few features to better support vision use cases:
1. `stochastic_depth_drop_rate` to supress model overfitting.
2. `return_attention_scores`, optionally returns the attention output.
3. `ffn_has_residual_connection`, clearly define whether feedforward network has
   residual connection or not to avoid ambiguity.
"""
from typing import List, Optional, Tuple, Union

import gin
import tensorflow as tf

from official.nlp import modeling
from official.vision.modeling.layers.nn_layers import StochasticDepth


@tf.keras.utils.register_keras_serializable(package="Vision")
@gin.configurable
class TransformerScaffold(modeling.layers.TransformerScaffold):
  """TransformerScaffold layer for vision applications.

  This layer is a subclass of NLP TransformerScaffold:

  Attributes:
    stochastic_depth_drop_rate: Drop rate for the residual connections.
    return_attention_scores: Optionally return the attention output.
    ffn_has_residual_connection: Whether the feedforward network has internal
      residual connection and layer norm. If False, the residual connection and
      the layer norm op are called inside TransformerScaffold.
  """

  def __init__(self,
               *args,
               stochastic_depth_drop_rate: float = 0.0,
               return_attention_scores: bool = False,
               ffn_has_residual_connection: bool = False,
               **kwargs):
    """Initializes TransformerEncoderBlock."""
    super().__init__(*args, **kwargs)
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._return_attention_scores = return_attention_scores
    self._ffn_has_residual_connection = ffn_has_residual_connection

  def build(self, input_shape: Union[tf.TensorShape, List[int]]):
    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = StochasticDepth(self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = lambda x, *args, **kwargs: tf.identity(x)

    super().build(input_shape)

  def get_config(self):
    config = {"stochastic_depth_drop_rate": self._stochastic_depth_drop_rate,
              "return_attention_scores": self._return_attention_scores,
              "ffn_has_residual_connection": self._ffn_has_residual_connection}
    base_config = super().get_config()
    base_config.update(config)
    return base_config

  def call(
      self,
      inputs: tf.Tensor,
      training: Optional[bool] = None
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Transformer self-attention encoder block call."""
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        key_value = None
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError("Unexpected inputs to %s with length at %d" %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = (inputs, None, None)

    if key_value is None:
      key_value = input_tensor

    if self._norm_first:
      source_tensor = input_tensor
      input_tensor = self._attention_layer_norm(input_tensor, training=training)

    attention_layer_output = self._attention_layer(
        query=input_tensor,
        value=key_value,
        attention_mask=attention_mask,
        training=training,
        return_attention_scores=self._return_attention_scores)
    if isinstance(attention_layer_output, tuple):
      # `attention_layer_output` contains two tensors when
      # `return_attention_scores` is True.
      attention_output, attention_scores = attention_layer_output
    else:
      attention_output = attention_layer_output
    attention_output = self._attention_dropout(attention_output,
                                               training=training)

    if self._norm_first:
      source_attention_output = source_tensor + self._stochastic_depth(
          attention_output, training=training)
      attention_output = self._output_layer_norm(source_attention_output,
                                                 training=training)
    else:
      attention_output = self._attention_layer_norm(
          input_tensor +
          self._stochastic_depth(attention_output, training=training),
          training=training)

    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      intermediate_output = self._intermediate_activation_layer(
          intermediate_output)
      layer_output = self._output_dense(intermediate_output, training=training)
      layer_output = self._output_dropout(layer_output, training=training)
    else:
      layer_output = self._feedforward_block(attention_output,
                                             training=training)

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)

    if self._norm_first:
      if self._ffn_has_residual_connection:
        raise ValueError(
            "In the case of `norm_first`, the residual connection should be"
            "done in the TransformerScaffold call function, not FFN's"
            "call function.")
      output = source_attention_output + self._stochastic_depth(
          layer_output, training=training)
    else:
      if self._ffn_has_residual_connection:
        output = self._stochastic_depth(layer_output, training=training)
      else:
        output = self._output_layer_norm(
            attention_output + self._stochastic_depth(
                layer_output, training=training))

    if self._return_attention_scores:
      return output, attention_scores
    else:
      return output
