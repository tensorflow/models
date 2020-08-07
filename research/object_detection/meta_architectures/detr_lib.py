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
import tensorflow as tf
from object_detection.utils import shape_utils
from official.nlp.modeling import layers
from official.modeling import tf_utils

import math

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Text")
class TwoDimensionalPositionEmbedding(tf.keras.layers.Layer):
  """Creates a 2D positional embedding.

  This layer calculates the position encoding as a mix of sine and cosine
  functions with geometrically increasing wavelengths, independently in the x
  and y directions. It assumes the input is square in shape. Originally defined and 
  formulized in  "Attention is All You Need", section 3.5.
  (https://arxiv.org/abs/1706.03762).

  Generalized to two dimensions in DETR, based on "Image Transformer"
  (https://arxiv.org/abs/1802.05751).

  Arguments:
    hidden_size: Size of the hidden layer. Must be a multiple of 2.
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

    if self._hidden_size % 2 != 0:
      raise ValueError("Hidden size must be even.")

    super(TwoDimensionalPositionEmbedding, self).__init__(**kwargs)
    self._hidden_size = hidden_size / 2
    self._min_timescale = min_timescale
    self._max_timescale = max_timescale

  def get_config(self):
    config = {
        "hidden_size": self._hidden_size,
        "min_timescale": self._min_timescale,
        "max_timescale": self._max_timescale,
    }
    base_config = super(TwoDimensionalPositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_1d_encoding(self, length):
    """
    Generates a 1D encoding, implementing the functionality of the relative
    positional encoding, which this was based on.

    Args:
      length: the length of the encoding to generate.

    Returns:
      a 1D spatial encoding.
    """
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
    per_axis_size = int(math.sqrt(input_shape[1]))
    one_d_encoding = self._get_1d_encoding(per_axis_size)
    encoding_x = tf.repeat(one_d_encoding, repeats=per_axis_size, axis=0)
    encoding_y = tf.tile(one_d_encoding, multiples=[per_axis_size, 1])
    return tf.concat([encoding_x, encoding_y], axis=1)


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

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
        output_shape=(input_shape[1], self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        name="query")
    self.key_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abc,cde->abde",
        output_shape=(input_shape[1], self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        name="key")
    self.value_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abc,cde->abde",
        output_shape=(input_shape[1], self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        name="value")

    output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
    self.output_dense_layer = tf.keras.layers.experimental.EinsumDense(
        equation="abcd,cde->abe",
        output_shape=self.hidden_size,
        #num_summed_dimensions=2,
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
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
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
    print("QUERY")
    query = self.query_dense_layer(query_input)
    print("KEY")
    key = self.key_dense_layer(key_input)
    print("VALUE")
    value = self.value_dense_layer(value_input)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

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

    #weights = tf.keras.layers.Dropout(self.attention_dropout)(weights, training=training)
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
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    output = self.filter_dense_layer(x)
    
    #output = tf.keras.layers.Dropout(self.relu_dropout)(output, training=training)
    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)

    return output
