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

"""Implement T5 Transformer model by TF official NLP library.

Model paper: https://arxiv.org/pdf/1910.10683.pdf

T5TransformerParams and T5Transformer are public interfaces.
Other modules are implementation details, so users should never build libraries
depending on them.

To use with Keras, users can wrap them within Keras customized layers.
"""
import dataclasses
import functools
import math
from typing import Callable, Dict, Optional, Sequence, Text, Union

import numpy as np
import tensorflow as tf

from official.modeling import tf_utils

ShapeLike = Union[int, Sequence[int], tf.TensorShape]
Initializer = Callable[..., tf.Tensor]


class Module(tf.Module):
  """The nn Module extends from the tf.Module."""

  def __init__(self, dtype: tf.DType = tf.float32, name: Optional[Text] = None):
    """Initializes the nn Module.

    Args:
      dtype: the variable allocation dtype.
      name: a string for the module name.
    """
    super().__init__(name=name)
    self.dtype = dtype

  def create_variable(self,
                      name: Text,
                      shape: ShapeLike,
                      initializer: Initializer,
                      dtype: tf.DType = tf.float32,
                      **kwargs):
    initializer = tf_utils.clone_initializer(initializer)
    return tf.Variable(initializer(shape, dtype=dtype, **kwargs), name=name)

  def read_variable(self,
                    variable: tf.Variable,
                    as_dtype: Optional[tf.DType] = None):
    if as_dtype is not None:
      variable = tf.cast(variable, dtype=as_dtype)
    return variable


@tf.custom_gradient
def dense_gradient(x: tf.Tensor):
  """Identity operation whose gradient is converted to a ``tf.Tensor``.

  >>> embedding = tf.Variable(tf.random.normal([3, 3]))
  >>> with tf.GradientTape() as tape:
  ...   y = tf.nn.embedding_lookup(dense_gradient(embedding), [1])
  >>> tape.gradient(y, embedding).numpy()
  array([[ 0.,  0.,  0.],
         [ 1.,  1.,  1.],
         [ 0.,  0.,  0.]], dtype=float32)

  Args:
    x: A ``tf.Tensor``.

  Returns:
    The input ``tf.Tensor`` and a dense identity gradient function.
  """

  def grad(dy):
    if isinstance(dy, tf.IndexedSlices):
      return tf.convert_to_tensor(dy)
    else:
      return dy

  return x, grad


def make_attention_mask(query_input,
                        key_input,
                        pairwise_fn=tf.multiply,
                        dtype=tf.float32):
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
  attention weights will be `[batch..., heads, len_q, len_kv]` and this
  function will produce `[batch..., 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  mask = pairwise_fn(
      tf.expand_dims(query_input, axis=-1), tf.expand_dims(key_input, axis=-2))
  mask = tf.expand_dims(mask, axis=-3)
  return tf.cast(mask, dtype=dtype)


def make_causal_mask(x, dtype=tf.float32):
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
  will be `[batch..., heads, len, len]` and this function will produce a
  causal mask of shape `[batch..., 1, len, len]`.

  Args:
    x: input array of shape `[batch..., len]`
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  x_shape = tf.shape(x)
  idxs = tf.broadcast_to(tf.range(x_shape[-1], dtype=tf.int32), x_shape)
  return make_attention_mask(idxs, idxs, tf.greater_equal, dtype=dtype)


class Embed(Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def __init__(self,
               vocab_size: int,
               features: int,
               embeddings_initializer: Optional[Initializer] = None,
               compute_dtype: tf.DType = tf.float32,
               **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.features = features
    self.compute_dtype = compute_dtype
    if embeddings_initializer:
      self.embed_init = embeddings_initializer
    else:
      self.embed_init = tf.keras.initializers.TruncatedNormal(stddev=1.0)
    with self.name_scope:
      self.embeddings = self.create_variable(
          "embedding", [self.vocab_size, self.features],
          self.embed_init,
          dtype=self.dtype)

  @tf.Module.with_name_scope
  def __call__(self, inputs: tf.Tensor, one_hot: bool = True):
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, the last dimension is to embed.
      one_hot: whether to use one-hot matmul to gather embeddings.

    Returns:
      The output shape follows the input, with an additional `features`
      dimension appended.
    """
    if one_hot:
      flat_inputs = tf.reshape(inputs, [-1])
      one_hot_data = tf.one_hot(
          flat_inputs, depth=self.vocab_size, dtype=self.compute_dtype)
      embeddings = tf.matmul(
          one_hot_data,
          self.read_variable(self.embeddings, as_dtype=self.compute_dtype))
      input_shape = tf_utils.get_shape_list(inputs)
      embeddings = tf.reshape(embeddings, input_shape + [self.features])
      return embeddings
    else:
      return tf.nn.embedding_lookup(
          dense_gradient(
              self.read_variable(self.embeddings, as_dtype=self.compute_dtype)),
          inputs)

  def attend(self, query):
    """Attends over the embedding using a query tensor.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An tensor with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    return tf.matmul(
        query,
        self.read_variable(self.embeddings, as_dtype=query.dtype),
        transpose_b=True)


class RMSNorm(Module):
  """A layernorm module in the T5 style.

  No bias and no subtraction of mean.
  """

  def __init__(self, hidden_size: int, epsilon: float = 1e-6, **kwargs):
    super().__init__(**kwargs)
    self.variance_epsilon = epsilon
    with self.name_scope:
      self.weight = self.create_variable(
          "scale", [hidden_size],
          dtype=self.dtype,
          initializer=tf.keras.initializers.Ones())

  @tf.Module.with_name_scope
  def __call__(self, x):
    # Keeps the computation inside the layer norm to be float32.
    compute_dtype = x.dtype
    x = tf.cast(x, dtype=tf.float32)
    variance = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
    x = x * tf.math.rsqrt(variance + self.variance_epsilon)
    x = tf.cast(x, dtype=compute_dtype)
    return self.read_variable(self.weight, as_dtype=compute_dtype) * x


class Linear(Module):
  """Linear module, optionally including bias."""

  def __init__(self,
               in_features: int,
               out_features: int,
               use_bias: bool = True,
               w_init: Optional[Initializer] = None,
               b_init: Optional[Initializer] = None,
               **kwargs):
    """Constructs a `Linear` module."""
    super().__init__(**kwargs)
    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.w_init = w_init
    if self.use_bias:
      self.b_init = b_init if b_init else tf.keras.initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

    with self.name_scope:
      if self.w_init is None:
        stddev = 1 / math.sqrt(self.in_features)
        self.w_init = tf.keras.initializers.HeNormal()

      self.w = self.create_variable(
          "kernel", [self.in_features, self.out_features],
          initializer=self.w_init,
          dtype=self.dtype)

      if self.use_bias:
        self.b = self.create_variable(
            "bias", [self.out_features],
            initializer=self.b_init,
            dtype=self.dtype)

  @tf.Module.with_name_scope
  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    outputs = tf.matmul(inputs,
                        self.read_variable(self.w, as_dtype=inputs.dtype))
    if self.use_bias:
      outputs = tf.add(outputs,
                       self.read_variable(self.b, as_dtype=inputs.dtype))
    return outputs


class Linear3D(Module):
  """Linear3D module, optionally including bias.

  Kernel stored as 2d parameter for compatibility with Adafactor optimizer.
  """

  def __init__(self,
               in_features: int,
               out_features: int,
               num_heads: int,
               use_bias: bool = True,
               to_3d: bool = True,
               w_init: Optional[Initializer] = None,
               b_init: Optional[Initializer] = None,
               **kwargs):
    """Constructs a `Linear3D` module."""
    super().__init__(**kwargs)
    self.in_features = in_features
    self.out_features = out_features
    self.num_heads = num_heads
    self.use_bias = use_bias
    self.to_3d = to_3d
    self.w_init = w_init
    if self.to_3d:
      self.kernel_2d_shape = (self.in_features,
                              self.num_heads * self.out_features)
      self.kernel_3d_shape = (self.in_features, self.num_heads,
                              self.out_features)
      self.bias_shape = (self.num_heads, self.out_features)
      bias_rank = 2
    else:
      self.kernel_2d_shape = (self.in_features * self.num_heads,
                              self.out_features)
      self.kernel_3d_shape = (self.num_heads, self.in_features,
                              self.out_features)
      self.bias_shape = (self.out_features,)
      bias_rank = 1
    if self.use_bias:
      self.b_init = b_init or tf.keras.initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

    with self.name_scope:
      if self.w_init is None:
        self.w_init = tf.keras.initializers.HeNormal()

      self.w = self.create_variable(
          "kernel",
          self.kernel_2d_shape,
          initializer=self.w_init,
          dtype=self.dtype)

      if self.use_bias:
        self.b = self.create_variable(
            "bias", self.bias_shape, initializer=self.b_init, dtype=self.dtype)

  @tf.Module.with_name_scope
  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    # B: batch size
    # S: From Sequence length
    # D: dimension
    # N: Number of heads
    # H: head size
    compute_dtype = inputs.dtype
    w = self.read_variable(self.w, as_dtype=compute_dtype)
    w = tf.reshape(w, self.kernel_3d_shape)
    if self.to_3d:
      outputs = tf.einsum("BSD,DNH->BSNH", inputs, w)
    else:
      outputs = tf.einsum("BSNH,NHD->BSD", inputs, w)
    if self.use_bias:
      outputs = tf.add(outputs,
                       self.read_variable(self.b, as_dtype=compute_dtype))
    return outputs


class Dropout(Module):
  """Randomly drop units in the input at a given rate."""

  def __init__(self, rate: float, **kwargs):
    """Constructs a Dropout module.

    Args:
      rate: Probability that each element of x is discarded. Must be a scalar in
        the range `[0, 1)`.
      **kwargs: other keyword args.
    """
    super().__init__(**kwargs)
    self._rate = rate

  @tf.Module.with_name_scope
  def __call__(self,
               x: tf.Tensor,
               training: bool,
               noise_shape: Optional[ShapeLike] = None) -> tf.Tensor:
    """call method for the Dropout module.

    Args:
      x: the input tensor.
      training: whether it is performing training pass.
      noise_shape: (Optional) Shape vector controlling the shape of the random
        noise used to apply dropout. If not set this will be the shape of the
        input. If set it should be broadcastable to the input shape.

    Returns:
      A tensor after applying dropout.
    """
    if not training:
      return x
    return tf.nn.dropout(x, rate=self._rate, noise_shape=noise_shape)


class FFN(Module):
  """Feed-forward Network. No layer norm, output dropout, or skip connection."""

  activation_map = {
      "relu": tf.nn.relu,
      "gelu": functools.partial(tf.nn.gelu, approximate=True),
      "swish": tf.nn.silu,
      "silu": tf.nn.silu,
  }

  def __init__(self,
               d_model: int,
               d_ff: int,
               activations: Sequence[str],
               use_bias: bool = False,
               dropout_rate: Optional[float] = 0.0,
               layer_norm_epsilon: Optional[float] = 1e-6,
               weight_initializer: Optional[Initializer] = None,
               bias_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    self.use_bias = use_bias
    with self.name_scope:
      self.wi = []
      self.activations = activations
      for idx, act_fn in enumerate(activations):
        if (act_fn is not None and act_fn != "linear" and
            act_fn not in self.activation_map):
          raise ValueError("Invalid activation function string is passed: %s" %
                           act_fn)
        dense_name = "wi" if len(activations) == 1 else f"wi_{idx}"
        self.wi.append(
            Linear(
                d_model,
                d_ff,
                use_bias=self.use_bias,
                w_init=weight_initializer,
                b_init=bias_initializer,
                dtype=self.dtype,
                name=dense_name))

      self.wo = Linear(
          d_ff,
          d_model,
          use_bias=self.use_bias,
          w_init=weight_initializer,
          b_init=bias_initializer,
          dtype=self.dtype,
          name="wo")
      self.dropout = Dropout(rate=dropout_rate)

  @tf.Module.with_name_scope
  def __call__(self,
               hidden_states: tf.Tensor,
               training: bool = False) -> tf.Tensor:
    h = hidden_states
    factors = []
    for wi, act_fn in zip(self.wi, self.activations):
      if act_fn is None or act_fn == "linear":
        factors.append(wi(h))
      else:
        factors.append(self.activation_map[act_fn](wi(h)))
    h = functools.reduce(tf.math.multiply, factors)
    h_shape = tf_utils.get_shape_list(h)
    h_shape[-2] = 1
    h = self.dropout(h, noise_shape=h_shape, training=training)
    h = self.wo(h)
    return h


class RelativePositionEmbedding(Module):
  """Relative position embeddings of T5 style."""

  def __init__(self,
               num_heads: int,
               relative_attention_num_buckets: int = 32,
               relative_attention_max_distance: int = 128,
               bidirectional: bool = True,
               embeddings_initializer: Optional[Initializer] = None,
               compute_dtype: tf.DType = tf.float32,
               **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.bidirectional = bidirectional
    self.relative_attention_max_distance = relative_attention_max_distance
    with self.name_scope:
      self.relative_attention_bias = Embed(
          vocab_size=self.relative_attention_num_buckets,
          features=self.num_heads,
          embeddings_initializer=embeddings_initializer,
          dtype=self.dtype,
          compute_dtype=compute_dtype,
          name="rel_embedding")

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.

    If bidirectional=False, then positive relative positions are invalid.

    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.

    All relative positions >=max_distance map to the same bucket.

    All relative positions <=-max_distance map to the same bucket.

    This should allow for more graceful generalization to longer sequences
    than the model has been trained on.

    Args:
      relative_position: an int32 Tensor
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
      values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += tf.cast(tf.math.less(n, 0), tf.int32) * num_buckets
      n = tf.math.abs(n)
    else:
      n = tf.math.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = tf.math.less(n, max_exact)
    val_if_large = max_exact + tf.dtypes.cast(
        tf.math.log(
            tf.cast(n, tf.float32) / max_exact + np.finfo(np.float32).eps) /
        math.log(max_distance / max_exact) * (num_buckets - max_exact),
        tf.int32,
    )
    val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
    ret += tf.where(is_small, n, val_if_large)
    return ret

  @tf.Module.with_name_scope
  def __call__(self, qlen, klen):
    context_position = tf.range(qlen)[:, None]
    memory_position = tf.range(klen)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=self.bidirectional,
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance)
    values = self.relative_attention_bias(rp_bucket)
    values = tf.expand_dims(
        tf.transpose(values, [2, 0, 1]),
        axis=0)  # shape (1, num_heads, qlen, klen)
    return values


class MultiHeadAttention(Module):
  """T5 Attention from Mesh TensorFlow."""

  def __init__(self,
               d_model: int,
               d_kv: int,
               num_heads: int,
               use_bias: bool = False,
               dropout_rate: Optional[float] = 0.0,
               rescale_query: bool = False,
               weight_initializer: Optional[Initializer] = None,
               bias_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    with self.name_scope:
      self.d_model = d_model
      self.d_kv = d_kv
      self.num_heads = num_heads
      self.rescale_query = rescale_query
      self.use_bias = use_bias

      if rescale_query or weight_initializer is None:
        query_w_init = weight_initializer
      else:
        init_std_rescaling = tf.math.sqrt(tf.cast(self.d_kv, dtype=self.dtype))
        query_w_init = (
            lambda *args, **kwargs: (  # pylint: disable=g-long-lambda
                tf_utils.clone_initializer(weight_initializer)(
                    *args, **kwargs) / init_std_rescaling))
      self.q = Linear3D(
          self.d_model,
          self.d_kv,
          num_heads=self.num_heads,
          use_bias=self.use_bias,
          w_init=query_w_init,
          b_init=bias_initializer,
          dtype=self.dtype,
          name="q")
      self.k = Linear3D(
          self.d_model,
          self.d_kv,
          num_heads=self.num_heads,
          use_bias=self.use_bias,
          w_init=weight_initializer,
          b_init=bias_initializer,
          dtype=self.dtype,
          name="k")
      self.v = Linear3D(
          self.d_model,
          self.d_kv,
          num_heads=self.num_heads,
          use_bias=self.use_bias,
          w_init=weight_initializer,
          b_init=bias_initializer,
          dtype=self.dtype,
          name="v")
      self.o = Linear3D(
          self.d_kv,
          self.d_model,
          num_heads=self.num_heads,
          use_bias=self.use_bias,
          to_3d=False,
          w_init=weight_initializer,
          b_init=bias_initializer,
          dtype=self.dtype,
          name="o")
      self.dropout = Dropout(dropout_rate)

  def _update_cache(self, key, value, cache, decode_position):
    """Updates cache states and gets full-length key/value tensors."""
    # Combines cached keys and values with new keys and values.
    # TPU one-hot handling.
    key_seq_dim = cache["key"].shape.as_list()[1]
    indices = tf.reshape(
        tf.one_hot(decode_position, key_seq_dim, dtype=key.dtype),
        [1, key_seq_dim, 1, 1])
    key = cache["key"] + key * indices
    value_seq_dim = cache["value"].shape.as_list()[1]
    indices = tf.reshape(
        tf.one_hot(decode_position, value_seq_dim, dtype=value.dtype),
        [1, value_seq_dim, 1, 1])
    value = cache["value"] + value * indices

    # Update cache
    cache["key"] = key
    cache["value"] = value

    return key, value

  @tf.Module.with_name_scope
  def __call__(self,
               query,
               mask=None,
               kv=None,
               position_bias=None,
               cache: Optional[Dict[str, tf.Tensor]] = None,
               decode_position=None,
               training=False):
    """MultiHeadAttention at work.

    Args:
      query: Tensor of shape (bs, qlen, d_model).
      mask: None or Tensor of shape (bs, n_heads, qlen, klen).
      kv: None or Tensor of shape (bs, klen, d_model).
      position_bias: None or Tensor of shape (bs, n_heads, qlen, klen).
      cache: If not None, cache["key"] and cache["value"] are Tensors of shape
        (bs, klen, n_heads, d_kv).
      decode_position: If not None, which position of the sequence we are
        decoding for. Ranges from 0 to klen - 1.
      training: Effects the behavior of dropout.

    Returns:
      A dictionary, output["context"] is the output after attention,
        output["cache"] contains updated cache for the next round of
        autoregressive decoding.
    """
    # Input is (bs, qlen, d_model)
    use_cache = cache is not None
    if kv is None:
      kv = query
    q = self.q(query)
    if self.rescale_query:
      q /= tf.math.sqrt(tf.cast(self.d_kv, dtype=q.dtype))
    k = self.k(kv)
    v = self.v(kv)
    if use_cache:
      k, v = self._update_cache(k, v, cache, decode_position)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(q_dim)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    scores = tf.einsum("bqnd,bknd->bnqk", q, k)  # (bs, n_heads, qlen, klen)
    if position_bias is not None:
      # If position_bias is None, the input embedings should already include
      # position embeddings.
      if use_cache:
        bias_shape = position_bias.shape.as_list()
        position_bias = tf.slice(
            position_bias, [0, 0, decode_position, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      scores += position_bias

    if mask is not None:
      scores += mask  # (bs, n_heads, qlen, klen)
    weights = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
    # weights shape = (bs, n_heads, qlen, klen)
    weights = tf.cast(weights, scores.dtype)
    weight_shape = tf_utils.get_shape_list(weights)
    # NOTE: T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to. We assume it is the query dimension.
    # (bs, n_heads, qlen, klen)
    weight_shape[-2] = 1
    weights = self.dropout(weights, training=training, noise_shape=weight_shape)

    c = tf.einsum("bnqk,bknd->bqnd", weights, v)
    c = self.o(c)

    outputs = dict(context=c)
    if cache:
      outputs["cache"] = cache
    return outputs


class SelfAttention(Module):
  """Self attention block including residual connection."""

  def __init__(self,
               d_model: int,
               d_kv: int,
               num_heads: int,
               dropout_rate: Optional[float] = 0.0,
               layer_norm_epsilon: Optional[float] = 1e-6,
               rescale_query: bool = False,
               weight_initializer: Optional[Initializer] = None,
               bias_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    with self.name_scope:
      self.self_attention = MultiHeadAttention(
          d_model=d_model,
          d_kv=d_kv,
          num_heads=num_heads,
          dropout_rate=dropout_rate,
          rescale_query=rescale_query,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="attention")
      self.layer_norm = RMSNorm(
          hidden_size=d_model,
          epsilon=layer_norm_epsilon,
          dtype=self.dtype,
          name="layer_norm")
      self.dropout = Dropout(dropout_rate)

  @tf.Module.with_name_scope
  def __call__(self,
               hidden_states,
               attention_mask=None,
               position_bias=None,
               cache=None,
               decode_position=None,
               training=False):
    norm_x = self.layer_norm(hidden_states)
    attention_outputs = self.self_attention(
        query=norm_x,
        mask=attention_mask,
        position_bias=position_bias,
        cache=cache,
        decode_position=decode_position,
        training=training)
    y = attention_outputs.pop("context")
    tensor_shape = tf_utils.get_shape_list(y)
    tensor_shape[-2] = 1
    y = self.dropout(y, noise_shape=tensor_shape, training=training)
    layer_output = hidden_states + y
    attention_outputs["layer_output"] = layer_output
    return attention_outputs


class CrossAttention(Module):
  """Cross attention block including residual connection."""

  def __init__(self,
               d_model: int,
               d_kv: int,
               num_heads: int,
               dropout_rate: Optional[float] = 0.0,
               layer_norm_epsilon: Optional[float] = 1e-6,
               rescale_query: bool = False,
               weight_initializer: Optional[Initializer] = None,
               bias_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    with self.name_scope:
      self.cross_attention = MultiHeadAttention(
          d_model=d_model,
          d_kv=d_kv,
          num_heads=num_heads,
          dropout_rate=dropout_rate,
          rescale_query=rescale_query,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="attention")
      self.layer_norm = RMSNorm(
          hidden_size=d_model,
          epsilon=layer_norm_epsilon,
          dtype=self.dtype,
          name="layer_norm")
      self.dropout = Dropout(dropout_rate)

  @tf.Module.with_name_scope
  def __call__(self,
               hidden_states,
               kv,
               attention_mask=None,
               position_bias=None,
               cache=None,
               training=False):
    norm_x = self.layer_norm(hidden_states)
    attention_outputs = self.cross_attention(
        query=norm_x,
        kv=kv,
        mask=attention_mask,
        position_bias=position_bias,
        cache=cache,
        training=training)
    y = attention_outputs.pop("context")
    tensor_shape = tf_utils.get_shape_list(y)
    tensor_shape[-2] = 1
    y = self.dropout(y, noise_shape=tensor_shape, training=training)
    layer_output = hidden_states + y
    attention_outputs["layer_output"] = layer_output
    return attention_outputs


class EncoderBlock(Module):
  """Transformer Encoder Block with only self attention."""

  def __init__(self,
               d_model: int,
               d_kv: int,
               num_heads: int,
               d_ff: int,
               ffn_activations: Sequence[str] = ("relu",),
               dropout_rate: Optional[float] = 0.0,
               layer_norm_epsilon: Optional[float] = 1e-6,
               rescale_query: bool = False,
               weight_initializer: Optional[Initializer] = None,
               bias_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    with self.name_scope:
      self.self_attention = SelfAttention(
          d_model=d_model,
          d_kv=d_kv,
          num_heads=num_heads,
          dropout_rate=dropout_rate,
          rescale_query=rescale_query,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="self_attention")
      self.ffn_layer_norm = RMSNorm(
          hidden_size=d_model,
          epsilon=layer_norm_epsilon,
          dtype=self.dtype,
          name="ffn_layer_norm")
      self.ffn = FFN(
          d_model=d_model,
          d_ff=d_ff,
          dropout_rate=dropout_rate,
          activations=ffn_activations,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="ffn")
      self.ffn_output_dropout = Dropout(dropout_rate)

  @tf.Module.with_name_scope
  def __call__(self,
               hidden_states,
               attention_mask=None,
               position_bias=None,
               training=False):
    attention_outputs = self.self_attention(
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        training=training)
    attn_output = attention_outputs["layer_output"]

    ffn_output = self.ffn_layer_norm(attn_output)
    ffn_output = self.ffn(ffn_output, training=training)
    tensor_shape = tf_utils.get_shape_list(ffn_output)
    tensor_shape[-2] = 1
    ffn_output = self.ffn_output_dropout(
        ffn_output, noise_shape=tensor_shape, training=training)
    ffn_output = attn_output + ffn_output

    return ffn_output


class EncDecoderBlock(Module):
  """Transformer Decoder Block with enc-decoder cross attention."""

  def __init__(self,
               d_model: int,
               d_kv: int,
               num_heads: int,
               d_ff: int,
               ffn_activations: Sequence[str] = ("relu",),
               dropout_rate: Optional[float] = 0.0,
               layer_norm_epsilon: Optional[float] = 1e-6,
               rescale_query: bool = False,
               weight_initializer: Optional[Initializer] = None,
               bias_initializer: Optional[Initializer] = None,
               **kwargs):
    super().__init__(**kwargs)
    with self.name_scope:
      self.self_attention = SelfAttention(
          d_model=d_model,
          d_kv=d_kv,
          num_heads=num_heads,
          dropout_rate=dropout_rate,
          rescale_query=rescale_query,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="self_attention")
      self.cross_attention = CrossAttention(
          d_model=d_model,
          d_kv=d_kv,
          num_heads=num_heads,
          dropout_rate=dropout_rate,
          rescale_query=rescale_query,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="cross_attention")
      self.ffn_layer_norm = RMSNorm(
          hidden_size=d_model,
          epsilon=layer_norm_epsilon,
          dtype=self.dtype,
          name="ffn_layer_norm")
      self.ffn = FFN(
          d_model=d_model,
          d_ff=d_ff,
          dropout_rate=dropout_rate,
          activations=ffn_activations,
          weight_initializer=weight_initializer,
          bias_initializer=bias_initializer,
          dtype=self.dtype,
          name="ffn")
      self.ffn_output_dropout = Dropout(dropout_rate,)

  @tf.Module.with_name_scope
  def __call__(self,
               hidden_states,
               encoder_hidden_states,
               attention_mask=None,
               encoder_decoder_mask=None,
               position_bias=None,
               cache=None,
               decode_position=None,
               training=False):
    self_attention_outputs = self.self_attention(
        hidden_states,
        attention_mask=attention_mask,
        decode_position=decode_position,
        position_bias=position_bias,
        cache=cache,
        training=training)
    if "cache" in self_attention_outputs:
      cache = self_attention_outputs["cache"]
    # No relative position bias is used for encoder-decoder cross attention.
    cross_attention_outputs = self.cross_attention(
        self_attention_outputs["layer_output"],
        kv=encoder_hidden_states,
        attention_mask=encoder_decoder_mask,
        training=training)
    attn_output = cross_attention_outputs["layer_output"]

    ffn_output = self.ffn_layer_norm(attn_output)
    ffn_output = self.ffn(ffn_output, training=training)
    tensor_shape = tf_utils.get_shape_list(ffn_output)
    tensor_shape[-2] = 1
    ffn_output = self.ffn_output_dropout(
        ffn_output, noise_shape=tensor_shape, training=training)
    ffn_output = attn_output + ffn_output

    return ffn_output, cache


@dataclasses.dataclass
class T5TransformerParams:
  """Transformer parameters."""
  num_layers: int
  d_model: int
  d_kv: int
  num_heads: int
  d_ff: int
  vocab_size: int
  target_vocab_size: Optional[int] = None
  dropout_rate: float = 0.0
  layer_norm_epsilon: float = 1e-6
  shared_embedding: bool = False
  vocab_embeddings_initializer: Optional[Initializer] = None
  relative_attention_num_buckets: int = 32
  relative_attention_max_distance: int = 128
  relative_embeddings_initializer: Optional[Initializer] = None
  weight_initializer: Optional[Initializer] = (tf.keras.initializers.HeNormal())
  bias_initializer: Optional[Initializer] = None
  rescale_query: bool = False
  bidirectional: bool = True
  ffn_activations: Sequence[str] = ("relu",)
  logits_via_embedding: bool = True
  num_decoder_layers: Optional[int] = None
  one_hot_embedding: bool = True
  layer_sharing: bool = False


class Encoder(Module):
  """Transformer Model Encoder for sequence to sequence."""

  def __init__(self,
               config: T5TransformerParams,
               shared_embedding: Optional[tf.Variable] = None,
               compute_dtype: tf.DType = tf.float32,
               **kwargs):
    super().__init__(**kwargs)
    self.config = config
    self.compute_dtype = compute_dtype
    self.embed_dim = config.d_model
    with self.name_scope:
      # Input Embedding.
      if shared_embedding is None:
        self.input_embed = Embed(
            vocab_size=self.config.vocab_size,
            features=self.config.d_model,
            embeddings_initializer=self.config.vocab_embeddings_initializer,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            name="input_embedding")
      else:
        self.input_embed = shared_embedding
      # Creates an alias to the input embed for encoder-only models.
      self.word_embed = self.input_embed
      self.relative_embedding = RelativePositionEmbedding(
          num_heads=self.config.num_heads,
          relative_attention_num_buckets=self.config
          .relative_attention_num_buckets,
          relative_attention_max_distance=self.config
          .relative_attention_max_distance,
          bidirectional=self.config.bidirectional,
          embeddings_initializer=self.config.relative_embeddings_initializer,
          dtype=self.dtype,
          compute_dtype=self.compute_dtype,
          name="relative_posemb")
      self.input_dropout = Dropout(self.config.dropout_rate,)
      self.encoder_layers = []
      for layer_idx in range(self.config.num_layers):
        if self.config.layer_sharing and layer_idx > 0:
          self.encoder_layers.append(self.encoder_layers[0])
        else:
          self.encoder_layers.append(
              EncoderBlock(
                  d_model=self.config.d_model,
                  d_kv=self.config.d_kv,
                  num_heads=self.config.num_heads,
                  d_ff=self.config.d_ff,
                  dropout_rate=self.config.dropout_rate,
                  ffn_activations=self.config.ffn_activations,
                  rescale_query=self.config.rescale_query,
                  weight_initializer=self.config.weight_initializer,
                  bias_initializer=self.config.bias_initializer,
                  dtype=self.dtype,
                  name="encoder_block_%d" % layer_idx))
      self.output_norm = RMSNorm(
          hidden_size=self.config.d_model,
          epsilon=self.config.layer_norm_epsilon,
          dtype=self.dtype,
          name="final_layer_norm")
      self.output_dropout = Dropout(self.config.dropout_rate,)

  @tf.Module.with_name_scope
  def __call__(self,
               inputs=None,
               encoder_mask=None,
               dense_inputs=None,
               training=False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input word ids. Optional if dense data are provided.
      encoder_mask: the encoder self-attention mask.
      dense_inputs: dense input data. Concat after the embedding if word ids
        are provided.
      training: whether it is training pass, affecting dropouts.

    Returns:
      output of a transformer encoder.
    """
    # Casts inputs to the dtype.
    if encoder_mask is not None:
      encoder_mask = tf.cast(encoder_mask, self.compute_dtype)
    cfg = self.config
    inputs_array = []
    if inputs is not None:
      inputs_array.append(
          self.input_embed(inputs, one_hot=cfg.one_hot_embedding))
    if dense_inputs is not None:
      inputs_array.append(dense_inputs)
    if not inputs_array:
      raise ValueError("At least one of inputs and dense_inputs must not be "
                       "None.")
    x = tf.concat(inputs_array, axis=1)
    tensor_shape = tf_utils.get_shape_list(x)
    tensor_shape[-2] = 1
    x = self.input_dropout(x, noise_shape=tensor_shape, training=training)
    if inputs is not None:
      input_length = tf_utils.get_shape_list(inputs)[1]
    else:
      input_length = 0
    position_bias = self.relative_embedding(input_length, input_length)
    if dense_inputs is not None:
      # Here we ignore relative position bias for dense embeddings.
      # TODO(yejiayu): If we proceed to video use cases, rework this part.
      dense_input_length = tf_utils.get_shape_list(dense_inputs)[1]
      # Position bias shape: [batch, 1, len, len]
      paddings = tf.constant([[0, 0], [0, 0], [0, dense_input_length],
                              [0, dense_input_length]])
      position_bias = tf.pad(position_bias, paddings, "CONSTANT")

    for i in range(cfg.num_layers):
      x = self.encoder_layers[i](
          x,
          attention_mask=encoder_mask,
          position_bias=position_bias,
          training=training)

    encoded = self.output_norm(x)
    encoded = self.output_dropout(encoded, training=training)
    return encoded


class Decoder(Module):
  """Transformer Model Decoder for sequence to sequence."""

  def __init__(self,
               config: T5TransformerParams,
               shared_embedding: Optional[tf.Variable] = None,
               compute_dtype: tf.DType = tf.float32,
               **kwargs):
    super().__init__(**kwargs)
    self.config = config
    self.compute_dtype = compute_dtype
    if self.config.num_decoder_layers is None:
      self.config.num_decoder_layers = self.config.num_layers
    if not hasattr(
        self.config,
        "target_vocab_size") or self.config.target_vocab_size is None:
      self.config.target_vocab_size = self.config.vocab_size
    with self.name_scope:
      # Target Embedding.
      if shared_embedding is None:
        self.target_embed = Embed(
            vocab_size=self.config.target_vocab_size,
            features=self.config.d_model,
            embeddings_initializer=self.config.vocab_embeddings_initializer,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            name="target_embedding")
      else:
        self.target_embed = shared_embedding
      self.target_dropout = Dropout(self.config.dropout_rate,)
      # Position bias for the target self attention.
      self.relative_embedding = RelativePositionEmbedding(
          num_heads=self.config.num_heads,
          relative_attention_num_buckets=self.config
          .relative_attention_num_buckets,
          relative_attention_max_distance=self.config
          .relative_attention_max_distance,
          bidirectional=self.config.bidirectional,
          embeddings_initializer=self.config.relative_embeddings_initializer,
          dtype=self.dtype,
          compute_dtype=self.compute_dtype,
          name="relative_posemb")
      self.decoder_layers = []
      for layer_idx in range(self.config.num_decoder_layers):
        if self.config.layer_sharing and layer_idx > 0:
          self.decoder_layers.append(self.decoder_layers[0])
        else:
          self.decoder_layers.append(
              EncDecoderBlock(
                  d_model=self.config.d_model,
                  d_kv=self.config.d_kv,
                  num_heads=self.config.num_heads,
                  d_ff=self.config.d_ff,
                  dropout_rate=self.config.dropout_rate,
                  ffn_activations=self.config.ffn_activations,
                  rescale_query=self.config.rescale_query,
                  weight_initializer=self.config.weight_initializer,
                  bias_initializer=self.config.bias_initializer,
                  dtype=self.dtype,
                  name="decoder_block_%d" % layer_idx))
      self.output_norm = RMSNorm(
          hidden_size=self.config.d_model,
          epsilon=self.config.layer_norm_epsilon,
          dtype=self.dtype,
          name="final_layer_norm")
      self.output_dropout = Dropout(self.config.dropout_rate,)
      if not self.config.logits_via_embedding:
        self.logits_dense = Linear(
            in_features=self.config.d_model,
            out_features=self.config.target_vocab_size,
            use_bias=False,
            dtype=self.dtype,
            name="logits")

  @tf.Module.with_name_scope
  def __call__(self,
               decoder_input_tokens,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               decode=False,
               decode_position=None,
               cache=None,
               max_decode_len=None,
               training=False):
    """Applies Transformer model on the inputs.

    Args:
      decoder_input_tokens: the decoder input tokens.
      encoded: the encoder outputs.
      decoder_mask: the decoder self-attention mask.
      encoder_decoder_mask: the cross-attention mask.
      decode: Whether to perform autoregressive decoding.
      decode_position: integer, the position to decode.
      cache: The cache dictionary of key, value tensors.
      max_decode_len: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      training: Whether it is training pass, affecting dropouts.

    Returns:
      output of a transformer encoder.
    """
    cfg = self.config
    # Casts inputs to the dtype.
    encoded = tf.cast(encoded, self.compute_dtype)
    if decoder_mask is not None:
      decoder_mask = tf.cast(decoder_mask, self.compute_dtype)
    if encoder_decoder_mask is not None:
      encoder_decoder_mask = tf.cast(encoder_decoder_mask, self.compute_dtype)
    x = self.target_embed(decoder_input_tokens, one_hot=cfg.one_hot_embedding)
    tensor_shape = tf_utils.get_shape_list(x)
    tensor_shape[-2] = 1
    x = self.target_dropout(x, noise_shape=tensor_shape, training=training)
    if cache is not None:
      position_bias = self.relative_embedding(max_decode_len, max_decode_len)
    else:
      input_length = tf_utils.get_shape_list(decoder_input_tokens)[1]
      position_bias = self.relative_embedding(input_length, input_length)
    for i in range(cfg.num_decoder_layers):
      if cache is None:
        x, _ = self.decoder_layers[i](
            x,
            encoder_hidden_states=encoded,
            attention_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            position_bias=position_bias,
            training=training)
      else:
        x, cache[i] = self.decoder_layers[i](
            x,
            encoder_hidden_states=encoded,
            attention_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            position_bias=position_bias,
            decode_position=decode_position,
            cache=cache[i],
            training=training)

    output = self.output_norm(x)
    tensor_shape = tf_utils.get_shape_list(output)
    tensor_shape[-2] = 1
    output = self.target_dropout(
        output, noise_shape=tensor_shape, training=training)
    if self.config.logits_via_embedding:
      logits = self.target_embed.attend(output)
      logits = logits / math.sqrt(cfg.d_model)
    else:
      logits = self.logits_dense(output)
    return logits, cache


class T5Transformer(Module):
  """Transformer Encoder+Decoder for sequence to sequence."""

  def __init__(self,
               config: T5TransformerParams,
               compute_dtype: tf.DType = tf.float32,
               **kwargs):
    super().__init__(**kwargs)
    # Builds the model components.
    shared_embedding = config.shared_embedding
    self.compute_dtype = compute_dtype
    self.decoder_cfg = dataclasses.replace(config, bidirectional=False)
    if self.decoder_cfg.num_decoder_layers is None:
      self.decoder_cfg.num_decoder_layers = self.decoder_cfg.num_layers
    self.encoder_cfg = dataclasses.replace(config, bidirectional=True)
    with self.name_scope:
      if shared_embedding:
        self.shared_embedding = Embed(
            vocab_size=config.vocab_size,
            features=config.d_model,
            embeddings_initializer=config.vocab_embeddings_initializer,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            name="shared")
      else:
        self.shared_embedding = None
      self.encoder = Encoder(
          self.encoder_cfg,
          self.shared_embedding,
          dtype=self.dtype,
          compute_dtype=self.compute_dtype)
      self.decoder = Decoder(
          self.decoder_cfg,
          self.shared_embedding,
          dtype=self.dtype,
          compute_dtype=self.compute_dtype)

  def encode(self,
             encoder_input_tokens=None,
             encoder_segment_ids=None,
             encoder_dense_inputs=None,
             encoder_dense_segment_ids=None,
             training=False):
    eligible_position_array = []
    if encoder_input_tokens is not None:
      eligible_position_array.append(
          tf.cast(tf.not_equal(encoder_input_tokens, 0), self.compute_dtype))
    if encoder_dense_inputs is not None:
      eligible_dense_positions = tf.cast(
          tf.reduce_any(tf.not_equal(encoder_dense_inputs, 0), axis=-1),
          self.compute_dtype)
      eligible_position_array.append(eligible_dense_positions)
    if not eligible_position_array:
      raise ValueError("At least one of encoder_input_tokens and"
                       " encoder_dense_inputs must be provided.")

    eligible_positions = tf.concat(eligible_position_array, axis=1)
    encoder_mask = make_attention_mask(
        eligible_positions, eligible_positions, dtype=tf.bool)

    encoder_segment_id_array = []
    if encoder_segment_ids is not None:
      encoder_segment_id_array.append(encoder_segment_ids)
    if encoder_dense_segment_ids is not None:
      encoder_segment_id_array.append(encoder_dense_segment_ids)
    if encoder_segment_id_array:
      encoder_segment_ids = tf.concat(encoder_segment_id_array, axis=1)
      segment_mask = make_attention_mask(
          encoder_segment_ids, encoder_segment_ids, tf.equal, dtype=tf.bool)
      encoder_mask = tf.math.logical_and(encoder_mask, segment_mask)
    encoder_mask = (1.0 - tf.cast(encoder_mask, self.compute_dtype)) * -1e9
    return self.encoder(
        encoder_input_tokens,
        encoder_mask,
        encoder_dense_inputs,
        training=training)

  def decode(
      self,
      encoded,
      decoder_target_tokens,
      encoder_input_tokens=None,  # only used for masks
      encoder_dense_inputs=None,
      decoder_input_tokens=None,
      encoder_segment_ids=None,
      encoder_dense_segment_ids=None,
      decoder_segment_ids=None,
      decode_position=None,
      cache=None,
      max_decode_len=None,
      decode=False,
      training=False):
    eligible_inputs_array = []
    if encoder_input_tokens is not None:
      eligible_inputs = tf.cast(
          tf.not_equal(encoder_input_tokens, 0), self.compute_dtype)
      eligible_inputs_array.append(eligible_inputs)
    if encoder_dense_inputs is not None:
      eligible_dense_inputs = tf.cast(
          tf.reduce_any(tf.not_equal(encoder_dense_inputs, 0), axis=-1),
          self.compute_dtype)
      eligible_inputs_array.append(eligible_dense_inputs)
    eligible_inputs = tf.concat(eligible_inputs_array, axis=1)

    if decode:
      # For decoding, the decoder_input_tokens is the decoder_target_tokens.
      decoder_input_tokens = decoder_target_tokens
      # fast autoregressive decoding uses only a special encoder-decoder mask
      decoder_mask = None
      encoder_decoder_mask = make_attention_mask(
          tf.cast(
              tf.not_equal(tf.ones_like(decoder_target_tokens), 0),
              self.compute_dtype),
          eligible_inputs,
          dtype=tf.bool)
    else:
      # Note that, masks should be created using decoder_target_tokens.
      eligible_targets = tf.cast(
          tf.not_equal(decoder_target_tokens, 0), self.compute_dtype)
      decoder_mask = tf.math.logical_and(
          make_attention_mask(
              eligible_targets, eligible_targets, dtype=tf.bool),
          make_causal_mask(decoder_target_tokens, dtype=tf.bool))
      encoder_decoder_mask = make_attention_mask(
          eligible_targets, eligible_inputs, dtype=tf.bool)
      if encoder_segment_ids is not None:
        if decoder_mask is not None:
          decoder_mask = tf.math.logical_and(
              decoder_mask,
              make_attention_mask(
                  decoder_segment_ids,
                  decoder_segment_ids,
                  tf.equal,
                  dtype=tf.bool))
        if encoder_dense_segment_ids is not None:
          encoder_segment_ids = tf.concat(
              [encoder_segment_ids, encoder_dense_segment_ids], axis=1)
        encoder_decoder_mask = tf.math.logical_and(
            encoder_decoder_mask,
            make_attention_mask(
                decoder_segment_ids,
                encoder_segment_ids,
                tf.equal,
                dtype=tf.bool))
    if decoder_mask is not None:
      decoder_mask = (1.0 - tf.cast(decoder_mask, self.compute_dtype)) * -1e9
    encoder_decoder_mask = (
        1.0 - tf.cast(encoder_decoder_mask, self.compute_dtype)) * -1e9
    logits, cache = self.decoder(
        decoder_input_tokens,
        encoded,
        decode_position=decode_position,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        cache=cache,
        max_decode_len=max_decode_len,
        decode=decode,
        training=training)
    return dict(logits=logits, encoded=encoded, cache=cache)

  @tf.Module.with_name_scope
  def __call__(self,
               encoder_input_tokens=None,
               decoder_target_tokens=None,
               encoder_dense_inputs=None,
               encoder_dense_segment_ids=None,
               decoder_input_tokens=None,
               encoder_segment_ids=None,
               decoder_segment_ids=None,
               training=False):
    """Applies Transformer model on the inputs.

    Args:
      encoder_input_tokens: input tokens to the encoder.
      decoder_target_tokens: target tokens to the decoder.
      encoder_dense_inputs: input dense vectors to the encoder.
      encoder_dense_segment_ids: dense input segmentation info for packed
      decoder_input_tokens: input tokens to the decoder, only required for
        training.
      encoder_segment_ids: input segmentation info for packed examples.
        examples.
      decoder_segment_ids: target segmentation info for packed examples.
      training: whether it is training pass, affecting dropouts.

    Returns:
      a dictionary of logits/cache.
    """
    encoded = self.encode(
        encoder_input_tokens=encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        encoder_dense_inputs=encoder_dense_inputs,
        encoder_dense_segment_ids=encoder_dense_segment_ids,
        training=training)
    outputs = self.decode(
        encoded=encoded,
        decoder_target_tokens=decoder_target_tokens,
        encoder_input_tokens=encoder_input_tokens,  # only used for masks.
        encoder_dense_inputs=encoder_dense_inputs,  # only used for masks.
        decoder_input_tokens=decoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        encoder_dense_segment_ids=encoder_dense_segment_ids,
        decoder_segment_ids=decoder_segment_ids,
        training=training)
    outputs["encoded"] = encoded
    return outputs

  @property
  def checkpoint_items(self):
    return dict(encoder=self.encoder, decoder=self.decoder)
