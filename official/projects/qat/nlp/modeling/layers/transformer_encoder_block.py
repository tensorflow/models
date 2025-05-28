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

"""Keras-based TransformerEncoder block layer."""
from typing import Optional

from absl import logging
import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official.projects.qat.nlp.modeling.layers.multi_head_attention import MultiHeadAttentionQuantized
from official.projects.qat.nlp.quantization import configs
from official.projects.qat.nlp.quantization import wrappers


def _quantized_multi_head_attention(*args, **kwargs):
  layer = MultiHeadAttentionQuantized(*args, **kwargs)
  return wrappers.MultiHeadAttentionQuantizeWrapper(
      layer, configs.DefaultMultiHeadAttentionQuantizeConfig())


def _quantized_einsum_dense(*args, **kwargs):
  layer = tf_keras.layers.EinsumDense(*args, **kwargs)
  return tfmot.quantization.keras.QuantizeWrapperV2(
      layer, configs.DefaultEinsumDenseQuantizeConfig())


def _output_quantize(layer):
  return tfmot.quantization.keras.QuantizeWrapperV2(
      layer, configs.Default8BitOutputQuantizeConfig())


class TransformerEncoderBlockQuantized(tf_keras.layers.Layer):
  """TransformerEncoderBlock layer.

  This layer implements the Transformer Encoder from
  "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
  which combines a `tf_keras.layers.MultiHeadAttention` layer with a
  two-layer feedforward network.

  References:
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [BERT: Pre-training of Deep Bidirectional Transformers for Language
     Understanding](https://arxiv.org/abs/1810.04805)
  """

  def __init__(self,
               num_attention_heads,
               inner_dim,
               inner_activation,
               output_range=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               inner_dropout=0.0,
               attention_initializer=None,
               attention_axes=None,
               **kwargs):
    """Initializes `TransformerEncoderBlock`.

    Args:
      num_attention_heads: Number of attention heads.
      inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network.
      inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network.
      output_range: the sequence output range, [0, output_range) for slicing the
        target sequence. `None` means the target sequence is not sliced.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: Dropout probability for within the attention layer.
      inner_dropout: Dropout probability for the first Dense layer in a
        two-layer feedforward network.
      attention_initializer: Initializer for kernels of attention layers. If set
        `None`, attention layers use kernel_initializer as initializer for
        kernel.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      **kwargs: keyword arguments/
    """
    super().__init__(**kwargs)

    if output_range is not None:
      logging.warning("`output_range` is available as an argument for `call()`."
                      "The `output_range` as __init__ argument is deprecated.")

    self._num_heads = num_attention_heads
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._attention_dropout = attention_dropout
    self._attention_dropout_rate = attention_dropout
    self._output_dropout = output_dropout
    self._output_dropout_rate = output_dropout
    self._output_range = output_range
    self._kernel_initializer = tf_keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf_keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf_keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf_keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf_keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf_keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf_keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._inner_dropout = inner_dropout
    if attention_initializer:
      self._attention_initializer = tf_keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = self._kernel_initializer
    self._attention_axes = attention_axes

  def build(self, input_shape):
    if isinstance(input_shape, tf.TensorShape):
      input_tensor_shape = input_shape
    elif isinstance(input_shape, (list, tuple)):
      input_tensor_shape = tf.TensorShape(input_shape[0])
    else:
      raise ValueError(
          "The type of input shape argument is not supported, got: %s" %
          type(input_shape))
    if len(input_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerEncoderBlock expects a three-dimensional "
                       "input of shape [batch, sequence, width].")
    hidden_size = input_tensor_shape[-1]
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_head_size = int(hidden_size // self._num_heads)
    common_kwargs = dict(
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    self._attention_layer = _quantized_multi_head_attention(
        num_heads=self._num_heads,
        key_dim=self._attention_head_size,
        dropout=self._attention_dropout,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        attention_axes=self._attention_axes,
        name="self_attention",
        **common_kwargs)
    self._attention_dropout = tf_keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = _output_quantize(
        tf_keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32))
    self._intermediate_dense = _quantized_einsum_dense(
        "abc,cd->abd",
        output_shape=(None, self._inner_dim),
        bias_axes="d",
        kernel_initializer=self._kernel_initializer,
        name="intermediate",
        **common_kwargs)
    policy = tf_keras.mixed_precision.global_policy()
    if policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      # TODO(b/154538392): Investigate this.
      policy = tf.float32
    self._intermediate_activation_layer = _output_quantize(
        tf_keras.layers.Activation(
            self._inner_activation, dtype=policy))
    self._inner_dropout_layer = tf_keras.layers.Dropout(
        rate=self._inner_dropout)
    self._output_dense = _quantized_einsum_dense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        name="output",
        kernel_initializer=self._kernel_initializer,
        **common_kwargs)
    self._output_dropout = tf_keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = _output_quantize(
        tf_keras.layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32))
    self._add = _output_quantize(tf_keras.layers.Add())
    self._output_add = tf_keras.layers.Add()

    super().build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self._num_heads,
        "inner_dim":
            self._inner_dim,
        "inner_activation":
            self._inner_activation,
        "output_dropout":
            self._output_dropout_rate,
        "attention_dropout":
            self._attention_dropout_rate,
        "output_range":
            self._output_range,
        "kernel_initializer":
            tf_keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf_keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf_keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf_keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf_keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf_keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf_keras.constraints.serialize(self._bias_constraint),
        "use_bias":
            self._use_bias,
        "norm_first":
            self._norm_first,
        "norm_epsilon":
            self._norm_epsilon,
        "inner_dropout":
            self._inner_dropout,
        "attention_initializer":
            tf_keras.initializers.serialize(self._attention_initializer),
        "attention_axes": self._attention_axes,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, output_range: Optional[tf.Tensor] = None):
    """Transformer self-attention encoder block call.

    Args:
      inputs: a single tensor or a list of tensors. `input tensor` as the single
        sequence of embeddings. [`input tensor`, `attention mask`] to have the
        additional attention mask. [`query tensor`, `key value tensor`,
        `attention mask`] to have separate input streams for the query, and
        key/value to the multi-head attention.
      output_range: the sequence output range, [0, output_range) for slicing the
        target sequence. `None` means the target sequence is not sliced. If you
        would like to have no change to the model training, it is better to only
        set the `output_range` for serving.

    Returns:
      An ouput tensor with the same dimensions as input/query tensor.
    """
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

    if output_range is None:
      output_range = self._output_range
    if output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor[:, 0:output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor
    attention_output = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)
    if self._norm_first:
      attention_output = self._add([source_tensor, attention_output])
    else:
      attention_output = self._attention_layer_norm(
          self._add([target_tensor, attention_output]))
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      return self._output_add([source_attention_output, layer_output])

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    return self._output_layer_norm(
        self._output_add([layer_output, attention_output]))
