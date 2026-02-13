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
"""Defines libraries for the Detection Transformer model in TF 2.0.

Model paper: https://arxiv.org/abs/2005.12872
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
import math

import tensorflow as tf

from object_detection.utils import shape_utils


class Transformer(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in the paper: End-to-End Object Detection
  with Transformers: https://arxiv.org/abs/2005.12872

  The Transformer model consists of an encoder and decoder. The input is
  a 3-D tensor of shape [batch_size, input_length, hidden_dimension].
  The encoder produces a continuous representation, and the decoder uses
  the encoder output to generate probabilities for the output sequence.
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0,
               layer_postprocess_dropout=0.1, relu_dropout=0, filter_size=256,
               num_hidden_layers=6, name="ODTransformer"):
    """Initialize layers to build Transformer model.

    Args:
      hidden_size: a number representing the length of the hidden dimension
      num_heads: the number of heads to use for multi-headed attention
      attention_dropout: dropout rate to apply in attention
      layer_postprocess_dropout: dropout rate to apply in postprocessing
      relu_dropout: dropout rate for the feed-forward networks
      filter_size: size of the middle layer in the feed-forward networks
      num_hidden_layers: number of encoder/decoder layers
      name: the name to give the transformer
    """
    super(Transformer, self).__init__(name=name)
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self._encoder_stack = EncoderStack(self._hidden_size,
                                       self._num_heads,
                                       self._attention_dropout,
                                       self._layer_postprocess_dropout,
                                       self._relu_dropout,
                                       self._filter_size,
                                       self._num_hidden_layers)
    self._decoder_stack = DecoderStack(self._hidden_size,
                                       self._num_heads,
                                       self._attention_dropout,
                                       self._layer_postprocess_dropout,
                                       self._relu_dropout,
                                       self._filter_size,
                                       self._num_hidden_layers)
    self._position_embedding = TwoDimensionalPositionEmbedding(
        hidden_size=self._hidden_size)

  def get_config(self):
    return {
        "_hidden_size": self._hidden_size,
        "_num_heads": self._num_heads,
        "_attention_dropout": self._attention_dropout,
        "_layer_postprocess_dropout": self._layer_postprocess_dropout,
        "_relu_dropout": self._relu_dropout,
        "_filter_size": self._filter_size,
        "_num_hidden_layers": self._num_hidden_layers,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 2.
        First item, inputs: float32 tensor with shape [batch_size,
            input_length, hidden_dimension].
        Second item, queries: float32 tensor with shape
          [batch_size, num_queries, hidden_dimension].
      training: boolean, whether in training mode or not.

    Returns:
      A tensor of length [batch_size, num_queries, hidden_dimension]
    """
    inputs, queries = inputs[0], inputs[1]

    with tf.name_scope("add_pos_encoding"):
      pos_encoding = self._position_embedding(inputs=inputs)

    with tf.name_scope("Transformer"):
      encoder_outputs = self._encoder_stack(
          inputs, training, pos_encoding)
      decoder_outputs = self._decoder_stack(
          queries, encoder_outputs, training, pos_encoding)
      return decoder_outputs

class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing.

  Note: this is the version described in the paper. However, a version
  with layer normalization before rather than after the layer may work better.
  """

  def __init__(self, layer, layer_postprocess_dropout):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self._postprocess_dropout = layer_postprocess_dropout

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "_postprocess_dropout": self._postprocess_dropout,
    }

  def call(self, to_add, *args, **kwargs):
    """Calls wrapped layer with same parameters."""

    training = kwargs["training"]

    y = self.layer(*args, **kwargs)

    if training:
      y = tf.nn.dropout(y, rate=self._postprocess_dropout)
    return self.layer_norm(to_add + y)

class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.0,
               layer_postprocess_dropout=0.1, relu_dropout=0.0,
               filter_size=256, num_hidden_layers=6):
    """ Initiallize the encoder stack.

    Args:
      hidden_size: a number representing the length of the hidden dimension
      num_heads: the number of heads to use for multi-headed attention
      attention_dropout: dropout rate to apply in attention
      layer_postprocess_dropout: dropout rate to apply in postprocessing
      relu_dropout: dropout rate for the feed-forward networks
      filter_size: size of the middle layer in the feed-forward networks
      num_hidden_layers: number of encoder/decoder layers
      name: the name to give the transformer
    """
    super(EncoderStack, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self.layers = []

  def build(self, input_shape):
    """Builds the encoder stack."""
    for _ in range(self._num_hidden_layers):
      # Create sublayers for each layer.
      self_attention_layer = SelfAttention(
          self._hidden_size, self._num_heads,
          self._attention_dropout)
      feed_forward_network = FeedForwardNetwork(
          self._hidden_size, self._filter_size, self._relu_dropout)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer,
                                   self._layer_postprocess_dropout),
          PrePostProcessingWrapper(feed_forward_network,
                                   self._layer_postprocess_dropout)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "_hidden_size": self._hidden_size,
        "_num_heads": self._num_heads,
        "_attention_dropout": self._attention_dropout,
        "_layer_postprocess_dropout": self._layer_postprocess_dropout,
        "_relu_dropout": self._relu_dropout,
        "_filter_size": self._filter_size,
        "_num_hidden_layers": self._num_hidden_layers
    }

  def call(self, encoder_inputs, training, encoding):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      training: boolean, whether in training mode or not.
      encoding: spatial encoding to add at every layer.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for _, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      encoder_inputs = self_attention_layer(
          encoder_inputs,
          encoder_inputs + encoding,
          encoder_inputs,
          training=training)
      encoder_inputs = feed_forward_network(
          encoder_inputs, encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.0,
               layer_postprocess_dropout=0.1, relu_dropout=0.0,
               filter_size=256, num_hidden_layers=6):
    """ Initiallize the decoder stack.

    Args:
      hidden_size: a number representing the length of the hidden dimension
      num_heads: the number of heads to use for multi-headed attention
      attention_dropout: dropout rate to apply in attention
      layer_postprocess_dropout: dropout rate to apply in postprocessing
      relu_dropout: dropout rate for the feed-forward networks
      filter_size: size of the middle layer in the feed-forward networks
      num_hidden_layers: number of encoder/decoder layers
      name: the name to give the transformer
    """
    super(DecoderStack, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self.layers = []

  def build(self, input_shape):
    """Builds the decoder stack."""
    for _ in range(self._num_hidden_layers):
      self_attention_layer = SelfAttention(
          self._hidden_size, self._num_heads,
          self._attention_dropout)
      enc_dec_attention_layer = Attention(
          self._hidden_size, self._num_heads,
          self._attention_dropout)
      feed_forward_network = FeedForwardNetwork(
          self._hidden_size, self._filter_size, self._relu_dropout)

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer,
                                   self._layer_postprocess_dropout),
          PrePostProcessingWrapper(enc_dec_attention_layer,
                                   self._layer_postprocess_dropout),
          PrePostProcessingWrapper(feed_forward_network,
                                   self._layer_postprocess_dropout)
      ])

    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)
    super(DecoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "_hidden_size": self._hidden_size,
        "_num_heads": self._num_heads,
        "_attention_dropout": self._attention_dropout,
        "_layer_postprocess_dropout": self._layer_postprocess_dropout,
        "_relu_dropout": self._relu_dropout,
        "_filter_size": self._filter_size,
        "_num_hidden_layers": self._num_hidden_layers,
    }

  def call(self,
           decoder_inputs,
           encoder_outputs,
           training,
           encoding):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape
        [batch_size, target_length, hidden_size].
      encoder_outputs: A tensor with shape
        [batch_size, input_length, hidden_size]
      training: A bool, whether in training mode or not.
      encoding: The spatial encoding to add at each layer.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    queries = decoder_inputs
    for _, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      decoder_inputs = self_attention_layer(
          decoder_inputs,
          decoder_inputs + queries,
          decoder_inputs,
          training=training)

      decoder_inputs = enc_dec_attention_layer(
          decoder_inputs,
          decoder_inputs + queries,
          encoder_outputs + encoding,
          encoder_outputs,
          training=training)

      decoder_inputs = feed_forward_network(
          decoder_inputs, decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)

@tf.keras.utils.register_keras_serializable(package="Text")
class TwoDimensionalPositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths. Defined and formulized
  in "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).

  Extended to 2D in the paper "Image Transformer".
  https://arxiv.org/abs/1802.05751

  Arguments:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.

  Raises:
    ValueError: if hidden_size is not a multiple of 2.
  """

  def __init__(self,
               hidden_size,
               min_timescale=1.0,
               max_timescale=1.0e4,
               **kwargs):
    super(TwoDimensionalPositionEmbedding, self).__init__(**kwargs)
    if hidden_size % 4 != 0:
      raise ValueError("Hidden size must be divisible by 4.")
    self._hidden_size = hidden_size / 2
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def get_config(self):
    config = {
        "hidden_size": self._hidden_size,
        "min_timescale": self._min_timescale,
        "max_timescale": self._max_timescale
    }
    base_config = super(TwoDimensionalPositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_1d_encoding(self, length):
    """Returns a 1-D positional encoding."""
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = self._hidden_size // 2
    min_timescale, max_timescale = self._min_timescale, self._max_timescale
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) *
        -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                                    axis=1)
    return position_embeddings

  def call(self, inputs):
    """Implements call() for the layer.

    Args:
      inputs: An tensor, the square root of whose second dimension
      will be used as the length of each 1-D embedding.

    Returns:
      A tensor in shape of [length, hidden_size].
    """
    input_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    per_axis_size = int(math.sqrt(input_shape[1]))
    one_d_encoding = self._get_1d_encoding(per_axis_size)
    encoding_x = tf.repeat(one_d_encoding, repeats=per_axis_size, axis=0)
    encoding_y = tf.tile(one_d_encoding, multiples=[per_axis_size, 1])
    return tf.concat([encoding_x, encoding_y], axis=1)


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer.

  TODO: switch to keras.layers.MultiHeadedAttention when available.
  """

  def __init__(self, hidden_size, num_heads, attention_dropout):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads

    def _glorot_initializer(fan_in, fan_out):
      limit = math.sqrt(6.0 / (fan_in + fan_out))
      return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

    attention_initializer = _glorot_initializer(input_shape.as_list()[-1],
                                                self.hidden_size)
    self.query_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abc,cde->abde",
        output_shape=(None, self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        name="query")
    self.key_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abc,cde->abde",
        output_shape=(None, self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        name="key")
    self.value_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abc,cde->abde",
        output_shape=(None, self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        name="value")

    output_initializer = _glorot_initializer(self.hidden_size,
                                             self.hidden_size)
    self.output_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abcd,cde->abe",
        output_shape=(None, self.hidden_size),
        kernel_initializer=output_initializer,
        name="output_transform")
    super(Attention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def call(self, query_input, key_input, value_input, training):
    """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      key_input: A tensor with shape [batch_size, length_key, hidden_size].
      value_input: A tensor with shape [batch_size, length_key, hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    query = self.query_dense_layer(query_input)
    key = self.key_dense_layer(key_input)
    value = self.value_dense_layer(value_input)

    # Scale query to prevent the dot product between query and key from growing
    # too large.
    depth = (self.hidden_size // self.num_heads)
    query *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    weights = tf.nn.softmax(logits, name="attention_weights")

    if training:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

    # Run the outputs through another linear projection layer. Recombining
    # heads is automatically done --> [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, query_input, value_input, training):
    return super(SelfAttention, self).call(
        query_input, query_input, value_input, training)


class FeedForwardNetwork(tf.keras.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout):
    """Initialize FeedForwardNetwork.

    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super(FeedForwardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

  def build(self, input_shape):
    self.filter_dense_layer = tf.keras.layers.Dense(
        self.filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name="filter_layer")
    self.output_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=True, name="output_layer")
    super(FeedForwardNetwork, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "filter_size": self.filter_size,
        "relu_dropout": self.relu_dropout,
    }

  def call(self, x, training):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    output = self.filter_dense_layer(x)

    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)

    return output
