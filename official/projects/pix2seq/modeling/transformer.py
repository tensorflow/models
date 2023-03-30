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

"""Specialized Transformers for DETR.

the position embeddings are added to the query and key for every self- and
cross-attention layer.
"""

import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import models
from official.nlp.modeling import layers
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers
from official.modeling import activations



class TransformerEncoder(tf.keras.layers.Layer):
  """Transformer Encoder."""

  def __init__(self,
               num_layers,
               mlp_dim,
               num_heads,
               dropout_rate=0.1,
               attention_dropout_rate=0.0,
               kernel_regularizer=None,
               inputs_positions=None,
               init_stochastic_depth_rate=0.1,
               kernel_initializer='glorot_uniform',
               **kwargs):
    super().__init__(**kwargs)
    self._num_layers = num_layers
    self._mlp_dim = mlp_dim
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._kernel_regularizer = kernel_regularizer
    self._inputs_positions = inputs_positions
    self._init_stochastic_depth_rate = init_stochastic_depth_rate
    self._kernel_initializer = kernel_initializer

  def build(self, input_shape):
    self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    self._encoder_layers = []
    # Set layer norm epsilons to 1e-6 to be consistent with JAX implementation.
    # https://flax.readthedocs.io/en/latest/_autosummary/flax.deprecated.nn.LayerNorm.html
    for i in range(self._num_layers):
      encoder_layer = nn_blocks.TransformerEncoderBlock(
          inner_activation=activations.gelu,
          num_attention_heads=self._num_heads,
          inner_dim=self._mlp_dim,
          output_dropout=self._dropout_rate,
          attention_dropout=self._attention_dropout_rate,
          kernel_regularizer=self._kernel_regularizer,
          kernel_initializer=self._kernel_initializer,
          use_bias=True,
          norm_first=True,
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 1, self._num_layers),
          norm_epsilon=1e-6)
      self._encoder_layers.append(encoder_layer)
    self._norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    super().build(input_shape)

  def call(self, inputs, training=None):
    x = inputs
    x = self._dropout(x, training=training)

    for encoder_layer in self._encoder_layers:
      x = encoder_layer(x, training=training)
    x = self._norm(x)
    return x

  def get_config(self):
    config = super().get_config()
    updates = {
        'num_layers': self._num_layers,
        'mlp_dim': self._mlp_dim,
        'num_heads': self._num_heads,
        'dropout_rate': self._dropout_rate,
        'attention_dropout_rate': self._attention_dropout_rate,
        'kernel_regularizer': self._kernel_regularizer,
        'inputs_positions': self._inputs_positions,
        'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
        'kernel_initializer': self._kernel_initializer,
        'add_pos_embed': self._add_pos_embed,
        'pos_embed_origin_shape': self._pos_embed_origin_shape,
        'pos_embed_target_shape': self._pos_embed_target_shape,
    }
    config.update(updates)
    return config


class TransformerDecoder(tf.keras.layers.Layer):
  """Transformer decoder.

  Like the encoder, the decoder is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self,
               num_layers=6,
               num_attention_heads=8,
               intermediate_size=2048,
               activation="relu",
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               use_bias=False,
               norm_first=True,
               norm_epsilon=1e-6,
               intermediate_dropout=0.0,
               **kwargs):
    """Initialize a Transformer decoder.

    Args:
      num_layers: Number of layers.
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate (Feedforward) layer.
      activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability.
      attention_dropout_rate: Dropout probability for attention layers.
      use_bias: Whether to enable use_bias in attention layer. If set `False`,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set `False`, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      **kwargs: key word arguemnts passed to tf.keras.layers.Layer.
    """
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_layers = num_layers
    self.num_attention_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._intermediate_dropout = intermediate_dropout

  def build(self, input_shape):
    """Implements build() for the layer."""
    self.decoder_layers = []
    for i in range(self.num_layers):
      self.decoder_layers.append(
          TransformerDecoderBlock(
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self._intermediate_size,
              intermediate_activation=self._activation,
              dropout_rate=self._dropout_rate,
              attention_dropout_rate=self._attention_dropout_rate,
              use_bias=self._use_bias,
              norm_first=self._norm_first,
              norm_epsilon=self._norm_epsilon,
              intermediate_dropout=self._intermediate_dropout,
              attention_initializer=tf_utils.clone_initializer(
                  models.seq2seq_transformer.attention_initializer(
                      input_shape[2])),
              name=("layer_%d" % i)))
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=self._norm_epsilon, dtype="float32")
    super(TransformerDecoder, self).build(input_shape)

  def get_config(self):
    config = {
        "num_layers": self.num_layers,
        "num_attention_heads": self.num_attention_heads,
        "intermediate_size": self._intermediate_size,
        "activation": self._activation,
        "dropout_rate": self._dropout_rate,
        "attention_dropout_rate": self._attention_dropout_rate,
        "use_bias": self._use_bias,
        "norm_first": self._norm_first,
        "norm_epsilon": self._norm_epsilon,
        "intermediate_dropout": self._intermediate_dropout
    }
    base_config = super(TransformerDecoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           target,
           memory,
           self_attention_mask=None,
           cross_attention_mask=None,
           cache=None,
           decode_loop_step=None,
           return_all_decoder_outputs=False,
           input_pos_embed=None,
           memory_pos_embed=None):
    """Return the output of the decoder layer stacks.

    Args:
      target: A tensor with shape `(batch_size, target_length, hidden_size)`.
      memory: A tensor with shape `(batch_size, input_length, hidden_size)`.
      self_attention_mask: A tensor with shape `(batch_size, target_len,
        target_length)`, the mask for decoder self-attention layer.
      cross_attention_mask: A tensor with shape `(batch_size, target_length,
        input_length)` which is the mask for encoder-decoder attention layer.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
        {layer_n: {"k": A tensor with shape `(batch_size, i, key_channels)`,
                   "v": A tensor with shape `(batch_size, i, value_channels)`},
                     ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.
      return_all_decoder_outputs: Return all decoder layer outputs. Note that
        the outputs are layer normed. This is useful when introducing per layer
        auxiliary loss.
      input_pos_embed: A tensor that is added to the query and key of the
        self-attention layer.
      memory_pos_embed: A tensor that is added to the query and key of the
        cross-attention layer.

    Returns:
      Output of decoder.
      float32 tensor with shape `(batch_size, target_length, hidden_size`).
    """

    output_tensor = target
    decoder_outputs = []
    for layer_idx in range(self.num_layers):
      transformer_inputs = [
          output_tensor, memory, cross_attention_mask, self_attention_mask,
          input_pos_embed, memory_pos_embed
      ]
      # Gets the cache for decoding.
      if cache is None:
        output_tensor, _ = self.decoder_layers[layer_idx](transformer_inputs)
      else:
        cache_layer_idx = str(layer_idx)
        output_tensor, cache[cache_layer_idx] = self.decoder_layers[layer_idx](
            transformer_inputs,
            cache=cache[cache_layer_idx],
            decode_loop_step=decode_loop_step)
      if return_all_decoder_outputs:
        decoder_outputs.append(self.output_normalization(output_tensor))

    if return_all_decoder_outputs:
      return decoder_outputs
    else:
      return self.output_normalization(output_tensor), cache


class TransformerDecoderBlock(tf.keras.layers.Layer):
  """Single transformer layer for decoder.

  It has three sub-layers:
  (1) a multi-head self-attention mechanism.
  (2) a encoder-decoder attention.
  (3) a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               num_attention_heads,
               intermediate_size,
               intermediate_activation,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
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
    """Initialize a Transformer decoder block.

    Args:
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate layer.
      intermediate_activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability for the post-attention and output
        dropout.
      attention_dropout_rate: Dropout probability for within the attention
        layer.
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
      intermediate_dropout: Dropout probability for intermediate_dropout_layer.
      attention_initializer: Initializer for kernels of attention layers. If set
        `None`, attention layers use kernel_initializer as initializer for
        kernel.
      **kwargs: key word arguemnts passed to tf.keras.layers.Layer.
    """
    super().__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf.keras.activations.get(
        intermediate_activation)
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
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
      self._attention_initializer = tf_utils.clone_initializer(
          self._kernel_initializer)
    self._cross_attention_cls = layers.attention.MultiHeadAttention

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
    self.attention_head_size = int(hidden_size) // self.num_attention_heads
    common_kwargs = dict(
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    # Self attention.
    self.self_attention = layers.attention.CachedAttention(
        num_heads=self.num_attention_heads,
        key_dim=self.attention_head_size,
        dropout=self.attention_dropout_rate,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        name="self_attention",
        **common_kwargs)
    self.self_attention_output_dense = tf.keras.layers.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        name="output",
        **common_kwargs)
    self.self_attention_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_rate)
    self.self_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype="float32"))
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
            epsilon=self._norm_epsilon,
            dtype="float32"))

    # Feed-forward projection.
    self.intermediate_dense = tf.keras.layers.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, self.intermediate_size),
        bias_axes="d",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        name="intermediate",
        **common_kwargs)
    self.intermediate_activation_layer = tf.keras.layers.Activation(
        self.intermediate_activation)
    self._intermediate_dropout_layer = tf.keras.layers.Dropout(
        rate=self._intermediate_dropout)
    self.output_dense = tf.keras.layers.EinsumDense(
        "abc,cd->abd",
        output_shape=(None, hidden_size),
        bias_axes="d",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        name="output",
        **common_kwargs)
    self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype="float32")
    super().build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self.num_attention_heads,
        "intermediate_size":
            self.intermediate_size,
        "intermediate_activation":
            tf.keras.activations.serialize(self.intermediate_activation),
        "dropout_rate":
            self.dropout_rate,
        "attention_dropout_rate":
            self.attention_dropout_rate,
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
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def common_layers_with_encoder(self):
    """Gets layer objects that can make a Transformer encoder block."""
    return [
        self.self_attention, self.self_attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_layer_norm
    ]

  def call(self, inputs, cache=None, decode_loop_step=None):
    input_tensor, memory, attention_mask, self_attention_mask, input_pos_embed, memory_pos_embed = inputs
    source_tensor = input_tensor
    if self._norm_first:
      input_tensor = self.self_attention_layer_norm(input_tensor)
    self_attention_output, cache = self.self_attention(
        query=input_tensor,
        key=input_tensor,
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
    
    # (gunho) transformer.py uses tf.float32 for numeric stability.
    
    cross_attn_inputs = dict(
        query=self_attention_output,
        key=memory,
        value=memory,
        attention_mask=attention_mask)
    attention_output = self.encdec_attention(**cross_attn_inputs)
    attention_output = self.encdec_attention_dropout(attention_output)
    
    # (gunho) transformer.py uses tf.float32 for numeric stability.
    attention_output = tf.cast(attention_output, tf.float32)

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
    
    # (gunho) transformer.py uses tf.float32 for numeric stability.
    layer_output = tf.cast(layer_output, tf.float32)

    if self._norm_first:
      layer_output = source_attention_output + layer_output
    else:
      layer_output = self.output_layer_norm(layer_output + attention_output)
    return layer_output, cache