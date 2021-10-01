"""Keras-based rotary embedding layer."""
# pylint: disable=g-classes-have-attributes
import math
import tensorflow as tf
from official.modeling import tf_utils
from official.nlp.modeling import layers

@tf.keras.utils.register_keras_serializable(package="Text")
class RotaryPositionEmbedding(tf.keras.layers.Layer):
    """Creates a rotary positional embedding

    Example:
    ```python
    rotary_position_embedding = RotaryPositionEmbedding(hidden_size=100,)
    inputs = tf.keras.Input((x,x), dtype=tf.float32)
    outputs = position_embedding(inputs)
    ```


    Args:
        hidden_size: Size of the hidden layer.

    Reference: This layer creates a positional embedding as described in
  [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self,
                 hidden_size: int,
                 min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4,
                 **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically
        # unstable in float16.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super().__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def get_config(self):
        config = {
            "hidden_size": self._hidden_size,
            "min_timescale": self._min_timescale,
            "max_timescale": self._max_timescale,
        }
        base_config = super(RotaryPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Implements call() for the layer.

        Args:
          inputs: An tensor whose second dimension will be used as `length`. If
            `None`, the other `length` argument must be specified.
          length: An optional integer specifying the number of positions. If both
            `inputs` and `length` are spcified, `length` must be equal to the second
            dimension of `inputs`.

        Returns:
          A tensor in shape of `(length, hidden_size)`.
        """
        if inputs is None:
            raise ValueError("If inputs is None, `length` must be set in "
                             "RelativePositionEmbedding().")
        q, k, v = inputs  # q = (batch_size, seq_len, num_heads, head_size)
        input_shape = tf_utils.get_shape_list(q)
        batch_size = input_shape[0]
        length = input_shape[1]
        num_heads1 = input_shape[2]
        head_size = input_shape[3]

        input_shape2 = tf_utils.get_shape_list(k)
        length2 = input_shape2[1]
        num_heads2 = input_shape2[2]
        head_size2 = input_shape2[3]

        position_ids = tf.cast(tf.range(length), tf.float32)[None]  # (1, length)
        num_timescales = self._hidden_size // 2
        indices = tf.cast(tf.range(num_timescales), tf.float32)  # (d/2)
        indices = tf.pow(10000.0, -2 * indices / num_timescales)  # (d/2,)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)  # (1, length, d/2)
        sin_emb = tf.repeat(tf.sin(embeddings), repeats=2, axis=-1)
        sin_emb = tf.expand_dims(sin_emb, 2)  # (1, length, 1, d/2)
        cos_emb = tf.repeat(tf.cos(embeddings), repeats=2, axis=-1)
        cos_emb = tf.expand_dims(cos_emb, 2)  # (1, length, 1, d/2)
        q2 = tf.stack([-q[..., 1::2], q[..., ::2]], axis=4)
        q2 = tf.reshape(q2, (batch_size, length, num_heads1, head_size))
        k2 = tf.stack([-k[..., 1::2], k[..., ::2]], axis=4)
        k2 = tf.reshape(k2, (batch_size, length2, num_heads2, head_size2))
        ret_q = q * cos_emb + q2 * sin_emb
        ret_w = k * cos_emb + k2 * sin_emb
        return ret_q, ret_w, v

class Roformer_Attention(tf.keras.layers.Layer):
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

    super(Roformer_Attention, self).__init__()
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
    self.query_dense_layer = layers.DenseEinsum(
        output_shape=(self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        use_bias=False,
        name="query")
    self.key_dense_layer = layers.DenseEinsum(
        output_shape=(self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        use_bias=False,
        name="key")
    self.value_dense_layer = layers.DenseEinsum(
        output_shape=(self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        use_bias=False,
        name="value")

    self.position_embedding = RotaryPositionEmbedding(hidden_size=size_per_head)

    output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
    self.output_dense_layer = layers.DenseEinsum(
        output_shape=self.hidden_size,
        num_summed_dimensions=2,
        kernel_initializer=output_initializer,
        use_bias=False,
        name="output_transform")
    super(Roformer_Attention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def call(self,
           query_input,
           source_input,
           bias,
           training,
           cache=None,
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
             "v": tensor with shape [batch_size, i, heads, dim_per_head]} where
               i is the current decoded length for non-padded decode, or max
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
    key = self.key_dense_layer(source_input)
    value = self.value_dense_layer(source_input)

    query, key, value = self.position_embedding([query, key, value])

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
    query *= depth**-0.5

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
    logits += bias
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

class Roformer_SelfAttention(Roformer_Attention):
  """Multiheaded self-attention layer."""

  def call(self,
           query_input,
           bias,
           training,
           cache=None,
           decode_loop_step=None):
    return super(Roformer_SelfAttention, self).call(query_input, query_input, bias,
                                           training, cache, decode_loop_step)