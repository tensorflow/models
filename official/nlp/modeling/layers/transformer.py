# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras-based transformer block layer."""
# pylint: disable=g-classes-have-attributes

import gin
import tensorflow as tf

from official.nlp.modeling.layers import attention
from official.nlp.modeling.layers import multi_channel_attention
from official.nlp.modeling.layers.util import tf_function_if_eager


@tf.keras.utils.register_keras_serializable(package="Text")
class Transformer(tf.keras.layers.Layer):
  """Transformer layer.

  This layer implements the Transformer from "Attention Is All You Need".
  (https://arxiv.org/abs/1706.03762).

  Arguments:
    num_attention_heads: Number of attention heads.
    intermediate_size: Size of the intermediate layer.
    intermediate_activation: Activation for the intermediate layer.
    dropout_rate: Dropout probability for the post-attention and output dropout.
    attention_dropout_rate: Dropout probability for within the attention layer.
    output_range: the sequence output range, [0, output_range) by slicing the
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
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized.
    norm_epsilon: Epsilon value to initialize normalization layers.
    intermediate_dropout: Dropout probability for intermediate_dropout_layer.
    attention_initializer: Initializer for kernels of attention layers. If set
      `None`, attention layers use kernel_initializer as initializer for kernel.
  """

  def __init__(self,
               num_attention_heads,
               intermediate_size,
               intermediate_activation,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
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
               intermediate_dropout=0.0,
               attention_initializer=None,
               **kwargs):
    super(Transformer, self).__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._intermediate_activation = intermediate_activation
    self._attention_dropout_rate = attention_dropout_rate
    self._dropout_rate = dropout_rate
    self._output_range = output_range
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout
    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = self._kernel_initializer

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
        raise ValueError("When passing a mask tensor to TransformerLayer, the "
                         "mask tensor must be of shape [batch, "
                         "sequence_length, sequence_length] (here %s). Got a "
                         "mask tensor of shape %s." %
                         (expected_mask_tensor_shape, mask_tensor_shape))
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
    self._attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=self._num_heads,
        key_dim=self._attention_head_size,
        dropout=self._attention_dropout_rate,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        name="self_attention",
        **common_kwargs)
    self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32))
    self._intermediate_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, self._intermediate_size),
        bias_axes="d",
        kernel_initializer=self._kernel_initializer,
        name="intermediate",
        **common_kwargs)
    policy = tf.keras.mixed_precision.experimental.global_policy()
    if policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      # TODO(b/154538392): Investigate this.
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._intermediate_activation, dtype=policy)
    self._intermediate_dropout_layer = tf.keras.layers.Dropout(
        rate=self._intermediate_dropout)
    self._output_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        name="output",
        kernel_initializer=self._kernel_initializer,
        **common_kwargs)
    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)

    super(Transformer, self).build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self._num_heads,
        "intermediate_size":
            self._intermediate_size,
        "intermediate_activation":
            self._intermediate_activation,
        "dropout_rate":
            self._dropout_rate,
        "attention_dropout_rate":
            self._attention_dropout_rate,
        "output_range":
            self._output_range,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint),
        "use_bias":
            self._use_bias,
        "norm_first":
            self._norm_first,
        "norm_epsilon":
            self._norm_epsilon,
        "intermediate_dropout":
            self._intermediate_dropout,
        "attention_initializer":
            tf.keras.initializers.serialize(self._attention_initializer)
    }
    base_config = super(Transformer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
      input_tensor, attention_mask = inputs
    else:
      input_tensor, attention_mask = (inputs, None)

    if self._output_range:
      target_tensor = input_tensor[:, 0:self._output_range, :]
      attention_mask = attention_mask[:, 0:self._output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
      target_tensor = input_tensor

    attention_output = self._attention_layer(
        query=target_tensor, value=input_tensor, attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)
    if self._norm_first:
      attention_output = source_tensor + attention_output
    else:
      attention_output = self._attention_layer_norm(target_tensor +
                                                    attention_output)
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    intermediate_output = self._intermediate_dense(attention_output)
    intermediate_output = self._intermediate_activation_layer(
        intermediate_output)
    intermediate_output = self._intermediate_dropout_layer(intermediate_output)
    layer_output = self._output_dense(intermediate_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      return source_attention_output + layer_output

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    return self._output_layer_norm(layer_output + attention_output)


@tf.keras.utils.register_keras_serializable(package="Text")
@gin.configurable
class CompiledTransformer(Transformer):

  @tf_function_if_eager(experimental_compile=True)
  def call(self, inputs):
    return super(CompiledTransformer, self).call(inputs)


@tf.keras.utils.register_keras_serializable(package="Text")
class TransformerDecoderLayer(tf.keras.layers.Layer):
  """Single transformer layer for decoder.

  It has three sub-layers:
  (1) a multi-head self-attention mechanism.
  (2) a encoder-decoder attention.
  (3) a positionwise fully connected feed-forward network.

  Arguments:
    num_attention_heads: Number of attention heads.
    intermediate_size: Size of the intermediate layer.
    intermediate_activation: Activation for the intermediate layer.
    dropout_rate: Dropout probability for the post-attention and output dropout.
    attention_dropout_rate: Dropout probability for within the attention layer.
    multi_channel_cross_attention: Whether to use `MultiChannelAttention` for
      cross-attention between target sequences and source sequences.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
    use_bias: Whether to enable use_bias in attention layer. If set False,
      use_bias in attention layer is disabled.
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized.
    norm_epsilon: Epsilon value to initialize normalization layers.
    intermediate_dropout: Dropout probability for intermediate_dropout_layer.
    attention_initializer: Initializer for kernels of attention layers. If set
      `None`, attention layers use kernel_initializer as initializer for kernel.
  """

  def __init__(self,
               num_attention_heads,
               intermediate_size,
               intermediate_activation,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               multi_channel_cross_attention=False,
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
               intermediate_dropout=0.0,
               attention_initializer=None,
               **kwargs):
    super(TransformerDecoderLayer, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf.keras.activations.get(
        intermediate_activation)
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self.multi_channel_cross_attention = multi_channel_cross_attention
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout
    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = self._kernel_initializer
    if self.multi_channel_cross_attention:
      self._cross_attention_cls = multi_channel_attention.MultiChannelAttention
    else:
      self._cross_attention_cls = attention.MultiHeadAttention

  def build(self, input_shape):
    target_tensor_shape = tf.TensorShape(input_shape[0])
    if len(target_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerLayer expects a three-dimensional input of "
                       "shape [batch, sequence, width].")
    hidden_size = target_tensor_shape[2]
    if hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self.num_attention_heads))
    self.attention_head_size = int(hidden_size / self.num_attention_heads)
    common_kwargs = dict(
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    # Self attention.
    self.self_attention = attention.CachedAttention(
        num_heads=self.num_attention_heads,
        key_dim=self.attention_head_size,
        dropout=self.attention_dropout_rate,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        name="self_attention",
        **common_kwargs)
    self.self_attention_output_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        kernel_initializer=self._kernel_initializer,
        name="output",
        **common_kwargs)
    self.self_attention_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_rate)
    self.self_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon))
    # Encoder-decoder attention.
    self.encdec_attention = self._cross_attention_cls(
        num_heads=self.num_attention_heads,
        key_dim=self.attention_head_size,
        dropout=self.attention_dropout_rate,
        output_shape=hidden_size,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        name="attention/encdec",
        **common_kwargs)

    self.encdec_attention_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_rate)
    self.encdec_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="attention/encdec_output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon))

    # Feed-forward projection.
    self.intermediate_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, self.intermediate_size),
        bias_axes="d",
        kernel_initializer=self._kernel_initializer,
        name="intermediate",
        **common_kwargs)
    self.intermediate_activation_layer = tf.keras.layers.Activation(
        self.intermediate_activation)
    self._intermediate_dropout_layer = tf.keras.layers.Dropout(
        rate=self._intermediate_dropout)
    self.output_dense = tf.keras.layers.experimental.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        kernel_initializer=self._kernel_initializer,
        name="output",
        **common_kwargs)
    self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=self._norm_epsilon)
    super(TransformerDecoderLayer, self).build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self.num_attention_heads,
        "intermediate_size":
            self.intermediate_size,
        "intermediate_activation":
            self.intermediate_activation,
        "dropout_rate":
            self.dropout_rate,
        "attention_dropout_rate":
            self.attention_dropout_rate,
        "multi_channel_cross_attention":
            self.multi_channel_cross_attention,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint),
        "use_bias":
            self._use_bias,
        "norm_first":
            self._norm_first,
        "norm_epsilon":
            self._norm_epsilon,
        "intermediate_dropout":
            self._intermediate_dropout,
        "attention_initializer":
            tf.keras.initializers.serialize(self._attention_initializer)
    }
    base_config = super(TransformerDecoderLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def common_layers_with_encoder(self):
    """Gets layer objects that can make a Transformer encoder block."""
    return [
        self.self_attention, self.self_attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_layer_norm
    ]

  def call(self, inputs, cache=None, decode_loop_step=None):
    if self.multi_channel_cross_attention:
      if len(inputs) != 5:
        raise ValueError(
            "TransformerDecoderLayer must have 5 inputs, when it uses "
            "multi_channel_cross_attention. But it got: %d" % len(inputs))
    elif len(inputs) != 4:
      raise ValueError(
          "TransformerDecoderLayer must have 4 inputs, but it got: %d" %
          len(inputs))
    input_tensor, memory, attention_mask, self_attention_mask = inputs[:4]
    source_tensor = input_tensor
    if self._norm_first:
      input_tensor = self.self_attention_layer_norm(input_tensor)
    self_attention_output, cache = self.self_attention(
        query=input_tensor,
        value=input_tensor,
        attention_mask=self_attention_mask,
        cache=cache,
        decode_loop_step=decode_loop_step)
    self_attention_output = self.self_attention_dropout(self_attention_output)
    if self._norm_first:
      self_attention_output = source_tensor + self_attention_output
    else:
      self_attention_output = self.self_attention_layer_norm(
          input_tensor + self_attention_output)
    if self._norm_first:
      source_self_attention_output = self_attention_output
      self_attention_output = self.encdec_attention_layer_norm(
          self_attention_output)
    cross_attn_inputs = dict(
        query=self_attention_output,
        value=memory,
        attention_mask=attention_mask)
    if self.multi_channel_cross_attention:
      # Accesses the 5-th input tensor for the doc-attention probabilities.
      cross_attn_inputs["context_attention_weights"] = inputs[-1]
    attention_output = self.encdec_attention(**cross_attn_inputs)
    attention_output = self.encdec_attention_dropout(attention_output)
    if self._norm_first:
      attention_output = source_self_attention_output + attention_output
    else:
      attention_output = self.encdec_attention_layer_norm(
          self_attention_output + attention_output)
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self.output_layer_norm(attention_output)

    intermediate_output = self.intermediate_dense(attention_output)
    intermediate_output = self.intermediate_activation_layer(
        intermediate_output)
    intermediate_output = self._intermediate_dropout_layer(intermediate_output)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    if self._norm_first:
      layer_output = source_attention_output + layer_output
    else:
      layer_output = self.output_layer_norm(layer_output + attention_output)
    return layer_output, cache
