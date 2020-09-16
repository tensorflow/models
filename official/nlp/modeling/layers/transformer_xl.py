# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Keras-based Transformer XL layer."""

from absl import logging

import tensorflow as tf

from official.nlp.modeling.layers import relative_attention


@tf.keras.utils.register_keras_serializable(package="Text")
class TransformerXLBlock(tf.keras.layers.Layer):
  """Transformer XL block.

  This implements a Transformer XL block from "Transformer-XL: Attentive
  Language Models Beyond a Fixed-Length Context"
  (https://arxiv.org/abs/1901.02860).

  This block is further extended to allow for the Transformer-XL
  re-parameterization in "XLNet: Generalized Autoregressive Pretraining for
  Language Understanding" (https://arxiv.org/abs/1906.08237).

  Given an input stream, this block computes attention, applies dropouts and
  layer norms and feeds into the FFN network.

  **Note: This layer is currently experimental.

  Attributes:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_attention_heads: The number of attention heads.
    head_size: The dimension size of each attention head.
    inner_size: The inner size for the transformer layers.
    dropout_rate: Dropout rate for the output of this layer.
    attention_dropout_rate: Dropout rate on attention probabilities.
    two_stream: Whether or not to use `TwoStreamRelativeAttention` used in the
      XLNet pretrainer. If `False`, then it will use
      `MultiHeadRelativeAttention` as in Transformer XL.
    norm_epsilon: Epsilon value to initialize normalization layers.
    inner_activation: The activation to use for the inner
      FFN layers.
    kernel_initializer: Initializer for dense layer kernels.
    inner_dropout: Dropout probability for the inner dropout
      layer.

  """

  def __init__(self,
               vocab_size,
               hidden_size,
               num_attention_heads,
               head_size,
               inner_size,
               dropout_rate,
               attention_dropout_rate,
               two_stream=False,
               norm_epsilon=1e-12,
               inner_activation="relu",
               kernel_initializer="variance_scaling",
               inner_dropout=0.0,
               **kwargs):
    """Initializes TransformerXLBlock layer."""

    super(TransformerXLBlock, self).__init__(**kwargs)
    self._vocab_size = vocab_size
    self._num_heads = num_attention_heads
    self._head_size = head_size
    self._hidden_size = hidden_size
    self._inner_size = inner_size
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._inner_activation = inner_activation
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._inner_dropout = inner_dropout
    self._two_stream = two_stream
    if two_stream:
      self._attention_layer_type = relative_attention.TwoStreamRelativeAttention
    else:
      self._attention_layer_type = relative_attention.MultiHeadRelativeAttention

  def build(self, input_shape):
    input_tensor = input_shape[0] if len(input_shape) == 2 else input_shape
    input_tensor_shape = tf.TensorShape(input_tensor)
    if len(input_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerLayer expects a three-dimensional input of "
                       "shape [batch, sequence, width].")
    batch_size, sequence_length, hidden_size = input_tensor_shape

    if len(input_shape) == 2:
      mask_tensor_shape = tf.TensorShape(input_shape[1])
      expected_mask_tensor_shape = tf.TensorShape(
          [batch_size, sequence_length, sequence_length])
      if not expected_mask_tensor_shape.is_compatible_with(mask_tensor_shape):
        raise ValueError("When passing a mask tensor to TransformerXLBlock, "
                         "the mask tensor must be of shape [batch, "
                         "sequence_length, sequence_length] (here %s). Got a "
                         "mask tensor of shape %s." %
                         (expected_mask_tensor_shape, mask_tensor_shape))
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_layer = self._attention_layer_type(
        num_heads=self._num_heads,
        key_dim=self._head_size,
        value_dim=self._head_size,
        dropout=self._attention_dropout_rate,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        name="rel_attn")
    self._attention_dropout = tf.keras.layers.Dropout(
        rate=self._attention_dropout_rate)
    self._attention_layer_norm = tf.keras.layers.LayerNormalization(
        name="self_attention_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)
    self._inner_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, self._inner_size),
        bias_axes="d",
        kernel_initializer=self._kernel_initializer,
        name="inner")

    self._inner_activation_layer = tf.keras.layers.Activation(
        self._inner_activation)
    self._inner_dropout_layer = tf.keras.layers.Dropout(
        rate=self._inner_dropout)
    self._output_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        name="output",
        kernel_initializer=self._kernel_initializer)
    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon)

    super(TransformerXLBlock, self).build(input_shape)

  def get_config(self):
    config = {
        "vocab_size":
            self._vocab_size,
        "hidden_size":
            self._hidden_size,
        "num_attention_heads":
            self._num_heads,
        "head_size":
            self._head_size,
        "inner_size":
            self._inner_size,
        "dropout_rate":
            self._dropout_rate,
        "attention_dropout_rate":
            self._attention_dropout_rate,
        "two_stream":
            self._two_stream,
        "norm_epsilon":
            self._norm_epsilon,
        "inner_activation":
            self._inner_activation,
        "kernel_initializer":
            self._kernel_initializer,
        "inner_dropout":
            self._inner_dropout,
    }
    base_config = super(TransformerXLBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           content_stream,
           content_attention_bias,
           positional_attention_bias,
           relative_position_encoding=None,
           segment_matrix=None,
           segment_encoding=None,
           segment_attention_bias=None,
           state=None,
           content_attention_mask=None,
           query_stream=None,
           query_attention_mask=None,
           target_mapping=None):
    """Implements `call` for the Layer.

    Arguments:
      content_stream: `Tensor`, the input content stream. This is the standard
        input to Transformer XL and is commonly referred to as `h` in XLNet.
      content_attention_bias: Bias `Tensor` for content based attention of shape
        `[num_heads, dim]`.
      positional_attention_bias: Bias `Tensor` for position based attention of
        shape `[num_heads, dim]`.
      relative_position_encoding: Relative positional encoding `Tensor` of shape
        `[B, L, dim]`.
      segment_matrix: Optional `Tensor` of shape `[B, S, S + M]`. Used in XLNet,
        but not in Transformer XL.
      segment_encoding: Optional `Tensor` of shape `[2, num_heads, dim]`. Used
        in XLNet, but not in Transformer XL.
      segment_attention_bias: Optional bias `Tensor` for segment based attention
        of shape `[num_heads, dim]`.
      state: Optional `Tensor` of shape `[B, M, E]`, where M is the length of
        the state or memory. If passed, this is also attended over as in
        Transformer XL.
      content_attention_mask: Optional `Tensor` representing the mask that is
        added to content attention logits. If state is not None, the mask source
        sequence dimension should extend M.
      query_stream: Optional `Tensor`, the query stream. This is introduced in
        `TwoStreamRelativeAttention`/XLNet pretrainer. This is ignored if
        `two_stream` is `False`.
      query_attention_mask: Optional `Tensor` representing the mask that is
        added to query attention logits. If state is not None, the mask source
        sequence dimension should extend M.
      target_mapping: Optional `Tensor` representing the target mapping when
        calculating query attention.

    Returns:
      A `dict` object, containing the key value pairs for `content_attention`
      and (if `two_stream` is `True`) `query_attention`.

    """
    if not self._two_stream and query_stream is not None:
      logging.warning("`query_stream` was provided but two stream attention is "
                      "disabled. `query_stream` will be ignored.")
    if self._two_stream:
      attention_kwargs = dict(
          content_stream=content_stream,
          query_stream=query_stream,
          query_attention_mask=query_attention_mask,
          target_mapping=target_mapping,
          content_attention_mask=content_attention_mask)
    else:
      attention_kwargs = dict(
          query=content_stream,
          value=content_stream,
          key=content_stream,
          attention_mask=content_attention_mask)

    common_attention_kwargs = dict(
        content_attention_bias=content_attention_bias,
        relative_position_encoding=relative_position_encoding,
        positional_attention_bias=positional_attention_bias,
        segment_matrix=segment_matrix,
        segment_encoding=segment_encoding,
        segment_attention_bias=segment_attention_bias,
        state=state)

    attention_kwargs.update(common_attention_kwargs)
    attention_output = self._attention_layer(**attention_kwargs)

    if self._two_stream:
      attention_streams = attention_output
      input_streams = [content_stream, query_stream]
    else:
      attention_streams = [attention_output]
      input_streams = [content_stream]

    attention_keys = ["content_attention", "query_attention"]
    attention_output = {}
    for attention_stream, input_stream, attention_key in zip(
        attention_streams, input_streams, attention_keys):
      attention_stream = self._attention_dropout(attention_stream)
      attention_stream = self._attention_layer_norm(
          attention_stream + input_stream)
      inner_output = self._inner_dense(attention_stream)
      inner_output = self._inner_activation_layer(
          inner_output)
      inner_output = self._inner_dropout_layer(
          inner_output)
      layer_output = self._output_dense(inner_output)
      layer_output = self._output_dropout(layer_output)
      layer_output = self._output_layer_norm(layer_output + attention_stream)
      attention_output[attention_key] = layer_output

    return attention_output

