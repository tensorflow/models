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

Layer normalization will come as a wrapper in a following PR.
"""
import tensorflow as tf
from object_detection.utils import shape_utils

import math

import tensorflow as tf


class Transformer(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in the paper: End-to-End Object Detection with Transformers

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0,
               layer_postprocess_dropout=0.1, relu_dropout=0, filter_size=256,
               num_hidden_layers=6, dtype=tf.float32, name="ODTransformer"):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(Transformer, self).__init__(name=name)
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._attention_dropout = attention_dropout
    self._layer_postprocess_dropout = layer_postprocess_dropout
    self._relu_dropout = relu_dropout
    self._filter_size = filter_size
    self._num_hidden_layers = num_hidden_layers
    self._dtype = tf.float32
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
        "_dtype": self._dtype,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: int tensor with shape [batch_size, input_length].
        Second item, queries: None or int tensor with shape
          [batch_size, num_queries, hidden_dimension].
      training: boolean, whether in training mode or not.

    Returns:
      A tensor of length [batch_size, num_queries, hidden_dimension]

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    #training = True
    inputs, targets = inputs[0], inputs[1]

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("add_pos_encoding"):
      pos_encoding = self._position_embedding(inputs=encoder_inputs)
      pos_encoding = tf.cast(pos_encoding, self._dtype)

    with tf.name_scope("Transformer"):
      encoder_outputs = self.encode(inputs, training, pos_encoding)
      logits = self.decode(targets, encoder_outputs, training, pos_encoding)
      return logits

  def encode(self, encoder_inputs, training, encoding):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self._layer_postprocess_dropout)

      return self._encoder_stack(
          encoder_inputs, training, encoding)

  def decode(self, targets, encoder_outputs, training, encoding=None):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      decoder_inputs = tf.cast(targets, self._dtype)
      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self._layer_postprocess_dropout)

      # Run values
      outputs = self._decoder_stack(
          decoder_inputs,
          encoder_outputs,
          training=training,
          encoding=encoding,
          queries=decoder_inputs)
      return outputs

class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, layer_postprocess_dropout):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.postprocess_dropout = layer_postprocess_dropout

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    self.layer_pre_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, add, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_pre_norm(x)
    newargs = [y]
    if len(args) == 1:
      newargs.append(self.layer_pre_norm(args[0]))
    else:
      newargs.extend(args)
    
    # Get layer output
    y = self.layer(*newargs, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    return add + y

class PrePostProcessingWrapperOld(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, layer_postprocess_dropout):
    super(PrePostProcessingWrapperOld, self).__init__()
    self.layer = layer
    self._postprocess_dropout = layer_postprocess_dropout

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapperOld, self).build(input_shape)

  def get_config(self):
    return {
        "_postprocess_dropout": self._postprocess_dropout,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer(*args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    #y = tf.keras.layers.Dropout(self._postprocess_dropout)(y, training=training)
    if training:
      y = tf.nn.dropout(y, rate=self._postprocess_dropout)
    return self.layer_norm(x + y)

class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.0,
               layer_postprocess_dropout=0.0, relu_dropout=0.0, filter_size=256,
               num_hidden_layers=6, dtype=tf.float32):
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
          PrePostProcessingWrapperOld(self_attention_layer, self._layer_postprocess_dropout),
          PrePostProcessingWrapperOld(feed_forward_network, self._layer_postprocess_dropout)
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
        "_num_hidden_layers": self._num_hidden_layers,
        "_dtype": self._dtype,
    }

  def call(self, encoder_inputs, training, encoding=None):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    print(self.layers)
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs,
              encoder_inputs + encoding,
              encoder_inputs,
              training=training)
        with tf.name_scope("ffn"):
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

  def __init__(self, hidden_size=256, num_heads=8, attention_dropout=0.1,
               layer_postprocess_dropout=0.1, relu_dropout=0.1, filter_size=256,
               num_hidden_layers=6, dtype=tf.float32):
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
          PrePostProcessingWrapperOld(self_attention_layer, self._layer_postprocess_dropout),
          PrePostProcessingWrapperOld(enc_dec_attention_layer, self._layer_postprocess_dropout),
          PrePostProcessingWrapperOld(feed_forward_network, self._layer_postprocess_dropout)
      ])
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
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
        "_dtype": self._dtype,
    }

  def call(self,
           decoder_inputs,
           encoder_outputs,
           training,
           cache=None,
           decode_loop_step=None,
           encoding=None,
           queries=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape
        [batch_size, target_length, hidden_size].
      encoder_outputs: A tensor with shape
        [batch_size, input_length, hidden_size]
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.name_scope(layer_name):
        with tf.name_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs,
              decoder_inputs + queries,
              decoder_inputs,
              training=training,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
        with tf.name_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs,
              decoder_inputs + queries,
              encoder_outputs + encoding,
              encoder_outputs,
              training=training)
        with tf.name_scope("ffn"):
          decoder_inputs = feed_forward_network(
              decoder_inputs, decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)

@tf.keras.utils.register_keras_serializable(package="Text")
class TwoDimensionalPositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths. Defined and formulized in
   "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).

  Arguments:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
  """

  def __init__(self,
               hidden_size,
               min_timescale=1.0,
               max_timescale=1.0e4,
               **kwargs):
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically
    # unstable in float16.
    if "dtype" not in kwargs:
      kwargs["dtype"] = "float32"

    super(TwoDimensionalPositionEmbedding, self).__init__(**kwargs)
    self._hidden_size = hidden_size / 2
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def get_config(self):
    config = {
        "hidden_size": self._hidden_size,
        "min_timescale": self._min_timescale,
        "max_timescale": self._max_timescale,
        "length": self._length,
    }
    base_config = super(TwoDimensionalPositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_1d_encoding(self, length):
    print(length)
    print(self._hidden_size)
    print(self._min_timescale)
    print(self._max_timescale)
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = self._hidden_size // 2
    min_timescale, max_timescale = self._min_timescale, self._max_timescale
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) *
        -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales,
                                                                0)
    position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                                    axis=1)
    return position_embeddings


  def call(self, inputs, length=None):
    """Implements call() for the layer.

    Args:
      inputs: An tensor whose second dimension will be used as `length`. If
        `None`, the other `length` argument must be specified.
      length: An optional integer specifying the number of positions. If both
        `inputs` and `length` are spcified, `length` must be equal to the
        second dimension of `inputs`.

    Returns:
      A tensor in shape of [length, hidden_size].
    """
    input_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    print(input_shape)
    per_axis_size = int(math.sqrt(input_shape[1]))
    one_d_encoding = self._get_1d_encoding(per_axis_size)
    print(one_d_encoding)
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

    output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
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

  def call(self, query_input, key_input, value_input, training, cache=None,
           decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      key_input: A tensor with shape [batch_size, length_key, hidden_size].
      value_input: A tensor with shape [batch_size, length_key, hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]}
        where i is the current decoded length for non-padded decode, or max
        sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.

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

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, query_input, value_input, training, cache=None,
           decode_loop_step=None):
    return super(SelfAttention, self).call(
        query_input, query_input, value_input, training, cache, decode_loop_step)


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
