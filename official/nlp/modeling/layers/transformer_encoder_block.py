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

from absl import logging
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling.layers import util


@tf.keras.utils.register_keras_serializable(package="Text")
class TransformerEncoderBlock(tf.keras.layers.Layer):
  """TransformerEncoderBlock layer.

  This layer implements the Transformer Encoder from
  "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
  which combines a `tf.keras.layers.MultiHeadAttention` layer with a
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
               use_query_residual=True,
               key_dim=None,
               value_dim=None,
               output_last_dim=None,
               diff_q_kv_att_layer_norm=False,
               **kwargs):
    """Initializes `TransformerEncoderBlock`.

    Note: If `output_last_dim` is used and `use_query_residual` is `True`, the
    `output_last_dim`'s value must equal the first input's last dimension for
    the query residual connection to work. This is because the residual
    connection after the multi-head-attention requires their dimensions to
    match. If `use_query_residual` is `False`, the `output_last_dim` dictactes
    the last dimension of the output of this module and the
    multi-head-attention.

    E.g. let's say input dims are `[batch_size, seq_dim, input_last_dim]`.
    Scenario 1: If `output_last_dim` is not `None`, then the output dims of this
    module would be `[batch_size, seq_dim, output_last_dim]`. Note `key_dim` is
    overriden by `output_last_dim`.
    Scenario 2: If `output_last_dim` is `None` and `key_dim` is not `None`, then
    the output dims of this module would be `[batch_size, seq_dim, key_dim]`.
    Scenario 3: If the `output_last_dim` and `key_dim` are both `None`, the
    output dims would be `[batch_size, seq_dim, input_last_dim]`.

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
      use_query_residual: Toggle to execute residual connection after attention.
      key_dim: `key_dim` for the `tf.keras.layers.MultiHeadAttention`. If
        `None`, we use the first `input_shape`'s last dim.
      value_dim: `value_dim` for the `tf.keras.layers.MultiHeadAttention`.
      output_last_dim: Final dimension of the output of this module. This also
        dictates the value for the final dimension of the
        multi-head-attention. When it's `None`, we use, in order of decreasing
        precedence, `key_dim` * `num_heads` or the first `input_shape`'s last
        dim as the output's last dim.
      diff_q_kv_att_layer_norm: If `True`, create a separate attention layer
        norm layer for query and key-value if `norm_first` is `True`. Invalid
        to set to `True` if `norm_first` is `False`.
      **kwargs: keyword arguments.
    """
    util.filter_kwargs(kwargs)
    super().__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._attention_dropout = attention_dropout
    self._attention_dropout_rate = attention_dropout
    self._output_dropout = output_dropout
    self._output_dropout_rate = output_dropout
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
    self._inner_dropout = inner_dropout
    self._use_query_residual = use_query_residual
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._output_last_dim = output_last_dim
    self._diff_q_kv_att_layer_norm = diff_q_kv_att_layer_norm
    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = tf_utils.clone_initializer(
          self._kernel_initializer)
    self._attention_axes = attention_axes

    if self._diff_q_kv_att_layer_norm and not self._norm_first:
      raise ValueError("Setting `diff_q_and_kv_attention_layer_norm` to True"
                       "when `norm_first` is False is invalid.")

  def build(self, input_shape):
    if isinstance(input_shape, tf.TensorShape):
      input_tensor_shape = input_shape
    elif isinstance(input_shape, (list, tuple)):
      input_tensor_shape = tf.TensorShape(input_shape[0])
    else:
      raise ValueError(
          "The type of input shape argument is not supported, got: %s" %
          type(input_shape))
    einsum_equation = "abc,cd->abd"
    if len(input_tensor_shape.as_list()) > 3:
      einsum_equation = "...bc,cd->...bd"
    hidden_size = input_tensor_shape[-1]
    if hidden_size % self._num_heads != 0:
      logging.warning(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)", hidden_size, self._num_heads)
    if self._key_dim is None:
      self._key_dim = int(hidden_size // self._num_heads)
    if self._output_last_dim is None:
      last_output_shape = hidden_size
    else:
      last_output_shape = self._output_last_dim

    common_kwargs = dict(
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    self._attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=self._num_heads,
        key_dim=self._key_dim,
        value_dim=self._value_dim,
        dropout=self._attention_dropout,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        attention_axes=self._attention_axes,
        output_shape=self._output_last_dim,
        name="self_attention",
        **common_kwargs)
    self._attention_dropout = tf.keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32))
    self._attention_layer_norm_kv = self._attention_layer_norm
    if self._diff_q_kv_att_layer_norm:
      self._attention_layer_norm_kv = (
          tf.keras.layers.LayerNormalization(
              name="self_attention_layer_norm_kv",
              axis=-1,
              epsilon=self._norm_epsilon,
              dtype=tf.float32))

    self._intermediate_dense = tf.keras.layers.EinsumDense(
        einsum_equation,
        output_shape=(None, self._inner_dim),
        bias_axes="d",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        name="intermediate",
        **common_kwargs)
    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      # TODO(b/154538392): Investigate this.
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._inner_activation, dtype=policy)
    self._inner_dropout_layer = tf.keras.layers.Dropout(
        rate=self._inner_dropout)
    self._output_dense = tf.keras.layers.EinsumDense(
        einsum_equation,
        output_shape=(None, last_output_shape),
        bias_axes="d",
        name="output",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)
    self._output_dropout = tf.keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)

    super(TransformerEncoderBlock, self).build(input_shape)

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
        "inner_dropout":
            self._inner_dropout,
        "attention_initializer":
            tf.keras.initializers.serialize(self._attention_initializer),
        "attention_axes": self._attention_axes,
        "use_query_residual":
            self._use_query_residual,
        "key_dim":
            self._key_dim,
        "value_dim":
            self._value_dim,
        "output_last_dim":
            self._output_last_dim,
        "diff_q_kv_att_layer_norm":
            self._diff_q_kv_att_layer_norm,
    }
    base_config = super(TransformerEncoderBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Transformer self-attention encoder block call.

    Args:
      inputs: a single tensor or a list of tensors.
        `input tensor` as the single sequence of embeddings.
        [`input tensor`, `attention mask`] to have the additional attention
          mask.
        [`query tensor`, `key value tensor`, `attention mask`] to have separate
          input streams for the query, and key/value to the multi-head
          attention.

    Returns:
      An output tensor with the same dimensions as input/query tensor.
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

    if self._output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:self._output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm_kv(key_value)
      target_tensor = input_tensor[:, 0:self._output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:self._output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm_kv(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor
    attention_output = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)

    if self._norm_first:
      # Important to not combine `self._norm_first` and
      # `self._use_query_residual` into one if clause because else is only for
      # `_norm_first == False`.
      if self._use_query_residual:
        attention_output = source_tensor + attention_output
    else:
      if self._use_query_residual:
        attention_output = target_tensor + attention_output
      attention_output = self._attention_layer_norm(attention_output)

    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      return source_attention_output + layer_output

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    return self._output_layer_norm(layer_output + attention_output)
