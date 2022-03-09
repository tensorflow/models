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

"""Keras-based TransformerEncoder block layer."""
import tensorflow as tf

from official.nlp import modeling
from official.vision.modeling.layers.nn_layers import StochasticDepth


class TransformerEncoderBlock(modeling.layers.TransformerEncoderBlock):
  """TransformerEncoderBlock layer with stochastic depth."""

  def __init__(self,
               *args,
               stochastic_depth_drop_rate=0.0,
               return_attention=False,
               **kwargs):
    """Initializes TransformerEncoderBlock."""
    super().__init__(*args, **kwargs)
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._return_attention = return_attention

  def build(self, input_shape):
    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = StochasticDepth(self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = lambda x, *args, **kwargs: tf.identity(x)

    super().build(input_shape)

  def get_config(self):
    config = {"stochastic_depth_drop_rate": self._stochastic_depth_drop_rate}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
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

    if self._output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:self._output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor[:, 0:self._output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:self._output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor
    attention_output, attention_scores = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask,
        return_attention_scores=True)
    attention_output = self._attention_dropout(attention_output)

    if self._norm_first:
      attention_output = source_tensor + self._stochastic_depth(
          attention_output, training=training)
    else:
      attention_output = self._attention_layer_norm(
          target_tensor +
          self._stochastic_depth(attention_output, training=training))

    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      if self._return_attention:
        return source_attention_output + self._stochastic_depth(
            layer_output, training=training), attention_scores
      else:
        return source_attention_output + self._stochastic_depth(
            layer_output, training=training)

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    if self._return_attention:
      return self._output_layer_norm(layer_output + self._stochastic_depth(
          attention_output, training=training)), attention_scores
    else:
      return self._output_layer_norm(layer_output + self._stochastic_depth(
          attention_output, training=training))
