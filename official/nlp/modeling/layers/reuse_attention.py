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

"""Keras-based attention layer."""
# pylint: disable=g-classes-have-attributes

import collections
import math
import string

import numpy as np
import tensorflow as tf

from official.modeling import tf_utils


_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(rank, attn_axes):
  """Builds einsum equations for the attention computation.

  Query, key, value inputs after projection are expected to have the shape as:
  `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
  `bs` and `<non-attention dims>` are treated as `<batch dims>`.

  The attention operations can be generalized:
  (1) Query-key dot product:
  `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
  <key attention dims>, num_heads, channels) -> (<batch dims>,
  num_heads, <query attention dims>, <key attention dims>)`
  (2) Combination:
  `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
  (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
  <query attention dims>, num_heads, channels)`

  Args:
    rank: Rank of query, key, value tensors.
    attn_axes: List/tuple of axes, `[-1, rank)`,
      that attention will be applied to.

  Returns:
    Einsum equations.
  """
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ""
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                        product_notation)
  attn_scores_rank = len(product_notation)
  combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                    target_notation)
  return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
  """Builds an einsum equation for projections inside multi-head attention."""
  input_str = ""
  kernel_str = ""
  output_str = ""
  bias_axes = ""
  letter_offset = 0
  for i in range(free_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    output_str += char

  letter_offset += free_dims
  for i in range(bound_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    kernel_str += char

  letter_offset += bound_dims
  for i in range(output_dims):
    char = _CHR_IDX[i + letter_offset]
    kernel_str += char
    output_str += char
    bias_axes += char
  equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

  return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


class ReuseMultiHeadAttention(tf.keras.layers.Layer):
  """MultiHeadAttention layer.

  This is an implementation of multi-headed attention as described in the paper
  "Attention is all you Need" (Vaswani et al., 2017).
  If `query`, `key,` `value` are the same, then
  this is self-attention. Each timestep in `query` attends to the
  corresponding sequence in `key`, and returns a fixed-width vector.

  This layer first projects `query`, `key` and `value`. These are
  (effectively) a list of tensors of length `num_attention_heads`, where the
  corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
  `(batch_size, <key/value dimensions>, key_dim)`,
  `(batch_size, <key/value dimensions>, value_dim)`.

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor.

  Finally, the result tensor with the last dimension as value_dim can take an
  linear projection and return.

  Examples:

  Performs 1D cross-attention over two sequence inputs with an attention mask.
  Returns the additional attention weights over heads.

  >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
  >>> target = tf.keras.Input(shape=[8, 16])
  >>> source = tf.keras.Input(shape=[4, 16])
  >>> output_tensor, weights = layer(target, source,
  ...                                return_attention_scores=True)
  >>> print(output_tensor.shape)
  (None, 8, 16)
  >>> print(weights.shape)
  (None, 2, 8, 4)

  Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

  >>> layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
  >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
  >>> output_tensor = layer(input_tensor, input_tensor)
  >>> print(output_tensor.shape)
  (None, 5, 3, 4, 16)

  Args:
    num_heads: Number of attention heads.
    key_dim: Size of each attention head for query and key.
    value_dim: Size of each attention head for value.
    dropout: Dropout probability.
    reuse_attention: An integer specifying number of heads to reuse.
      -1 for all heads.
    use_relative_pe: Whether to use relative position bias.
    max_sequence_length: Used to set the size of the relative positin encodings.
    use_bias: Boolean, whether the dense layers use bias vectors/matrices.
    output_shape: The expected shape of an output tensor, besides the batch and
      sequence dims. If not specified, projects back to the key feature dim.
    attention_axes: axes over which the attention is applied. `None` means
      attention over all axes, but batch, heads, and features.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.

  Call arguments:
    query: Query `Tensor` of shape `(B, T, dim)`.
    value: Value `Tensor` of shape `(B, S, dim)`.
    key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
      `value` for both `key` and `value`, which is the most common case.
    attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
      attention to certain positions. The boolean mask specifies which query
      elements can attend to which key elements, 1 indicates attention and 0
      indicates no attention. Broadcasting can happen for the missing batch
      dimensions and the head dimension.
    return_attention_scores: A boolean to indicate whether the output should
      be attention output if True, or (attention_output, attention_scores) if
      False. Defaults to False.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (no dropout).
      Defaults to either using the training mode of the parent layer/model,
      or False (inference) if there is no parent layer.

  Returns:
    attention_output: The result of the computation, of shape `(B, T, E)`,
      where `T` is for target sequence shapes and `E` is the query input last
      dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
      are project to the shape specified by `output_shape`.
    attention_scores: [Optional] multi-head attention coeffients over
      attention axes.
  """

  def __init__(self,
               num_heads,
               key_dim,
               value_dim=None,
               dropout=0.0,
               reuse_attention=0,
               use_relative_pe=False,
               pe_max_seq_length=512,
               use_bias=True,
               output_shape=None,
               attention_axes=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(ReuseMultiHeadAttention, self).__init__(**kwargs)
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim if value_dim else key_dim
    self._dropout = dropout
    if reuse_attention > self._num_heads or reuse_attention < -1:
      raise ValueError("reuse_attention should be between -1 "
                       "and %d in call to %s." % (self.__class__,
                                                  self._num_heads))
    if reuse_attention == -1:
      reuse_attention = self._num_heads
    self._reuse_heads = reuse_attention
    self._use_relative_pe = use_relative_pe
    self._pe_max_seq_length = pe_max_seq_length
    self._use_bias = use_bias
    self._output_shape = output_shape
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    if attention_axes is not None and not isinstance(attention_axes,
                                                     collections.abc.Sized):
      self._attention_axes = (attention_axes,)
    else:
      self._attention_axes = attention_axes
    self._built_from_signature = False
    self._query_shape, self._key_shape, self._value_shape = None, None, None
    # Use relative PE only if reuse_heads < num_heads.
    if self._use_relative_pe and self._reuse_heads < self._num_heads:
      # Determine the dtype from global policy.
      policy = tf.keras.mixed_precision.global_policy()
      if policy.name == "mixed_bfloat16":
        policy = tf.bfloat16
      elif policy.name == "mixed_float16":
        policy = tf.float16
      else:
        policy = tf.float32
      self._position_embeddings = tf.Variable(
          name="relative_position_embeddings",
          initial_value=lambda: tf.random.truncated_normal(  # pylint: disable=g-long-lambda
              [
                  1, self._num_heads - self._reuse_heads, 2 * self.
                  _pe_max_seq_length - 1
              ], mean=0.0, stddev=0.2, dtype=policy),
          trainable=True, dtype=policy)

  def get_config(self):
    config = {
        "num_heads": self._num_heads,
        "key_dim": self._key_dim,
        "value_dim": self._value_dim,
        "dropout": self._dropout,
        "use_bias": self._use_bias,
        "output_shape": self._output_shape,
        "attention_axes": self._attention_axes,
        "reuse_attention": self._reuse_heads,
        "use_relative_pe": self._use_relative_pe,
        "pe_max_seq_length": self._pe_max_seq_length,
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
        "query_shape": self._query_shape,
        "key_shape": self._key_shape,
        "value_shape": self._value_shape,
    }
    base_config = super(ReuseMultiHeadAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # If the layer has a different build() function from the Keras default,
    # we need to trigger the customized build to create weights.
    query_shape = config.pop("query_shape")
    key_shape = config.pop("key_shape")
    value_shape = config.pop("value_shape")
    layer = cls(**config)
    if None in [query_shape, key_shape, value_shape]:
      tf.get_logger().warning(
          "One of dimensions of the input shape is missing. It should have been"
          " memorized when the layer was serialized. "
          "%s is created without weights.",
          str(cls))
    else:
      layer._build_from_signature(query_shape, value_shape, key_shape)  # pylint: disable=protected-access
    return layer

  def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.

    Once the method is called, self._built_from_signature will be set to True.

    Args:
      query: Query tensor or TensorShape.
      value: Value tensor or TensorShape.
      key: Key tensor or TensorShape.
    """
    self._built_from_signature = True
    if hasattr(query, "shape"):
      self._query_shape = tf.TensorShape(query.shape)
    else:
      self._query_shape = tf.TensorShape(query)
    if hasattr(value, "shape"):
      self._value_shape = tf.TensorShape(value.shape)
    else:
      self._value_shape = tf.TensorShape(value)
    if key is None:
      self._key_shape = self._value_shape
    elif hasattr(key, "shape"):
      self._key_shape = tf.TensorShape(key.shape)
    else:
      self._key_shape = tf.TensorShape(key)

    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    # Any setup work performed only once should happen in an `init_scope`
    # to avoid creating symbolic Tensors that will later pollute any eager
    # operations.
    with tf.init_scope():
      free_dims = self._query_shape.rank - 1
      if self._reuse_heads < self._num_heads:
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=1, output_dims=2)
        self._query_dense = tf.keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads - self._reuse_heads, self._key_dim]),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
            **common_kwargs)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            self._key_shape.rank - 1, bound_dims=1, output_dims=2)
        self._key_dense = tf.keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads - self._reuse_heads, self._key_dim]),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
            **common_kwargs)
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          self._value_shape.rank - 1, bound_dims=1, output_dims=2)

      self._value_dense = []
      if self._reuse_heads > 0:
        self._value_dense.append(
            tf.keras.layers.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1, [self._reuse_heads, self._value_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="value_reuse",
                kernel_initializer=tf_utils.clone_initializer(
                    self._kernel_initializer),
                bias_initializer=tf_utils.clone_initializer(
                    self._bias_initializer),
                **common_kwargs))
      if self._reuse_heads < self._num_heads:
        self._value_dense.append(
            tf.keras.layers.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1,
                    [self._num_heads - self._reuse_heads, self._value_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="value_new",
                kernel_initializer=tf_utils.clone_initializer(
                    self._kernel_initializer),
                bias_initializer=tf_utils.clone_initializer(
                    self._bias_initializer),
                **common_kwargs))

      # Builds the attention computations for multi-head dot product attention.
      # These computations could be wrapped into the keras attention layer once
      # it support mult-head einsum computations.
      self._build_attention(output_rank)
      self._output_dense = []
      if self._reuse_heads > 0:
        self._output_dense.append(self._make_output_dense(
            free_dims, common_kwargs, "attention_output_reuse"))
      if self._reuse_heads < self._num_heads:
        self._output_dense.append(self._make_output_dense(
            free_dims, common_kwargs, "attention_output_new",
            self._reuse_heads == 0))

  def _make_output_dense(self, free_dims, common_kwargs, name=None,
                         use_bias=True):
    """Builds the output projection matrix.

    Args:
      free_dims: Number of free dimensions for einsum equation building.
      common_kwargs: Common keyword arguments for einsum layer.
      name: Name for the projection layer.
      use_bias: Use bias if self._use_bias is true

    Returns:
      Projection layer.
    """
    if self._output_shape:
      if not isinstance(self._output_shape, collections.abc.Sized):
        output_shape = [self._output_shape]
      else:
        output_shape = self._output_shape
    else:
      output_shape = [self._query_shape[-1]]
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=2, output_dims=len(output_shape))
    return tf.keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1, output_shape),
        bias_axes=bias_axes if (use_bias and self._use_bias) else None,
        name=name,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)

  def _build_attention(self, rank):
    """Builds multi-head dot-product attention computations.

    This function builds attributes necessary for `_compute_attention` to
    customize attention computation to replace the default dot-product
    attention.

    Args:
      rank: the rank of query, key, value tensors.
    """
    if self._attention_axes is None:
      self._attention_axes = tuple(range(1, rank - 2))
    else:
      self._attention_axes = tuple(self._attention_axes)
    self._dot_product_equation, self._combine_equation, attn_scores_rank = (
        _build_attention_equation(rank, attn_axes=self._attention_axes))
    norm_axes = tuple(
        range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = tf.keras.layers.Softmax(axis=norm_axes)
    self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)

  def _masked_softmax(self, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, T, S]
    if attention_mask is not None:
      # The expand dim happens starting from the `num_heads` dimension,
      # (<batch_dims>, num_heads, <query_attention_dims, key_attention_dims>)
      mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
      for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
        attention_mask = tf.expand_dims(
            attention_mask, axis=mask_expansion_axes)
    return self._softmax(attention_scores, attention_mask)

  def _compute_relative_position(self, query_seq_length, key_seq_length):
    position_zero = self._pe_max_seq_length - 1
    # We take the vector position variable and concatenate to form a matrix of
    # relative position encodings. i=0 indicates reltaive position is 0.
    indices = tf.expand_dims(tf.range(0, -query_seq_length, -1),
                             -1) + tf.range(key_seq_length) + position_zero
    indices = tf.maximum(indices, 0)
    indices = tf.minimum(indices, 2*self._pe_max_seq_length-2)
    attention_biases = tf.gather(self._position_embeddings, indices, axis=2)
    return attention_biases

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         reuse_scores=None,
                         attention_mask=None,
                         training=None):
    """Applies Dot-product attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for customized
    attention implementation.

    Args:
      query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
      key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
      value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
      reuse_scores: Attention scores from a previous layer if needed.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    # Partial or no reuse
    if self._reuse_heads < self._num_heads:
      query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))
      new_scores = tf.einsum(self._dot_product_equation, key, query)
      # Add relative position embeddings if required.
      if self._use_relative_pe:
        new_scores = new_scores + self._compute_relative_position(
            tf.shape(query)[1], tf.shape(key)[1])
      new_scores = self._masked_softmax(new_scores, attention_mask)
      if self._reuse_heads > 0:  # Partial reuse
        reuse_scores = reuse_scores[:, :self._reuse_heads, :, :]
        attention_scores = tf.concat([new_scores, reuse_scores], 1)
      else:  # No reuse
        attention_scores = new_scores
    else:  # Full reuse
      attention_scores = reuse_scores
      new_scores = None

    # `context_layer` = [B, T, N, H]
    attention_output = []
    # Partial or full reuse
    if self._reuse_heads > 0:
      attention_output.append(
          tf.einsum(self._combine_equation, self._dropout_layer(
              reuse_scores, training=training), value[0]))
    # Partial or no reuse
    if self._reuse_heads < self._num_heads:
      attention_output.append(
          tf.einsum(self._combine_equation, self._dropout_layer(
              new_scores, training=training), value[-1]))
    return attention_output, attention_scores

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None,
           reuse_attention_scores=None):
    if self._reuse_heads > 0 and reuse_attention_scores is None:
      raise ValueError("reuse_attention_scores cannot be None when "
                       "reuse_attention is True or > 0.")
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `value` = [B, S, N, H]
    value = [vd(value) for vd in self._value_dense]
    if self._reuse_heads < self._num_heads:
      # `query` = [B, T, N ,H]
      query = self._query_dense(query)

      # `key` = [B, S, N, H]
      key = self._key_dense(key)
    else:
      query, key = None, None

    attention_output, attention_scores = self._compute_attention(
        query, key, value, reuse_attention_scores, attention_mask, training)
    attention_output = [od(attention_output[i]) for i, od in enumerate(
        self._output_dense)]
    if len(attention_output) == 1:
      attention_output = attention_output[0]
    else:
      attention_output = attention_output[0] + attention_output[1]

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output
