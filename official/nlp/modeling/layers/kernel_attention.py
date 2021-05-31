# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Keras-based kernel attention layer."""

import functools
import math
import tensorflow as tf

_NUMERIC_STABLER = 1e-6


class KernelMask(tf.keras.layers.Layer):
  """Creates kernel attention mask.

    inputs: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    mask: a Tensor of shape [batch_size, from_seq_length] which indicates
      which part of the inputs we should not attend.

    Returns:
      float Tensor of shape [batch_size, from_seq_length] that KernelAttention
      takes as mask.
  """

  def call(self, inputs, mask):
    mask = tf.cast(mask, inputs.dtype)
    return mask


def create_projection_matrix(m, d, seed=None):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random length taken from the
  \chi(d) distribution.).

  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections. If not, we use the stateful
      api.

  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = math.ceil(m / d)
  block_list = tf.TensorArray(tf.float32,
                              size=tf.cast(nb_full_blocks, dtype=tf.int32))
  stateful = False
  if seed is None:
    stateful = True
    # dummy seed to make sure the graph compiles though the path is not taken.
    seed = tf.constant([0, 1])
  current_seed = seed
  for i in range(nb_full_blocks):
    if stateful:
      unstructured_block = tf.random.normal((d, d))
    else:
      unstructured_block = tf.random.stateless_normal((d, d), seed=current_seed)
      current_seed = tf.random.stateless_uniform([2],
                                                 seed=current_seed,
                                                 minval=None,
                                                 dtype=tf.int32)
    q, _ = tf.linalg.qr(unstructured_block)
    q = tf.transpose(q)
    block_list = block_list.write(i, q)
  final_matrix = block_list.concat()[:m]
  if stateful is None:
    multiplier = tf.norm(tf.random.normal((m, d)), axis=1)
  else:
    multiplier = tf.norm(
        tf.random.stateless_normal((m, d), seed=current_seed), axis=1)
  return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def _generalized_kernel(x, projection_matrix, is_query, f, h,
                        data_normalizer_fn=None):
  """Generalized kernel in RETHINKING ATTENTION WITH PERFORMERS.

  Args:
    x: The feature being transformed with shape [B, T, N ,H].
    projection_matrix: The matrix with shape [M, H] that we projecct x to, where
      M is the number of projections.
    is_query: Whether the transform is a query or key. This transform is
      symmetric is the argument is not used.
    f: A non-linear function applied on x or projected x.
    h: A muliplier which is a function of x applied after projected and
      transformed. Only applied if projection_matrix is not None.
    data_normalizer_fn: A function which takes x and returns a scalar that
      normalize data.

  Returns:
    Transformed feature.
  """
  # No asymmetric operations.
  del is_query

  if data_normalizer_fn is not None:
    x = data_normalizer_fn(x)

  if projection_matrix is None:
    return h(x) * f(x)
  else:
    x_projected = tf.einsum("BTNH,MH->BTNM", x, projection_matrix)
    return h(x) * f(x_projected) / tf.math.sqrt(
        tf.cast(tf.shape(projection_matrix)[0], tf.float32))


# pylint: disable=g-long-lambda
_TRANSFORM_MAP = {
    "elu":
        functools.partial(
            _generalized_kernel,
            f=lambda x: tf.keras.activations.elu(x) + 1,
            h=lambda x: 1),
    "relu":
        functools.partial(
            _generalized_kernel, f=tf.keras.activations.relu, h=lambda x: 1),
    "square":
        functools.partial(
            _generalized_kernel, f=tf.math.square, h=lambda x: 1),
    "exp":
        functools.partial(
            _generalized_kernel,
            # Avoid exp explosion by shifting.
            f=lambda x: tf.math.exp(
                x - tf.math.reduce_max(x, axis=[1, 2, 3], keepdims=True)),
            h=lambda x: tf.math.exp(
                -0.5 * tf.math.reduce_sum(
                    tf.math.square(x), axis=-1, keepdims=True)),
            data_normalizer_fn=lambda x: x /
            (tf.math.sqrt(tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))))),
    "expmod":
        functools.partial(
            _generalized_kernel,
            # Avoid exp explosion by shifting.
            f=lambda x: tf.math.exp(
                x - tf.math.reduce_max(x, axis=[1, 2, 3], keepdims=True)),
            h=lambda x: tf.math.exp(
                -0.5 * tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))),
            data_normalizer_fn=lambda x: x /
            (tf.math.sqrt(tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))))),
    "l2":
        functools.partial(
            _generalized_kernel,
            f=lambda x: x,
            h=lambda x: tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32)),
            data_normalizer_fn=lambda x: x),
    "identity": lambda x, projection_matrix, is_query: x
}
# pylint: enable=g-long-lambda


class KernelAttention(tf.keras.layers.MultiHeadAttention):
  """A variant of efficient transformers which replaces softmax with kernels.

  This module combines ideas from the two following papers:

  Rethinking Attention with Performers
  (https://arxiv.org/abs/2009.14794)
  - exp (Lemma 1, positive), relu, l2
  - random/deterministic projection

  Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
  (https://arxiv.org/abs/2006.16236)
  - elu

  with the theory of approximating angular Performer kernels from go/performer.

  The module enables computing efficient attention in both: long sequence and
  shorter sequence regimes. In the former setting, the attention matrix is never
  explicitly computed and instead its low-rank decomposition obtained with given
  kernel feature maps is leveraged to conduct attention module calculations
  (see: https://arxiv.org/abs/2006.16236). In the latter setting, attention
  matrix is constructed, but kernel features providing dimensionality reduction
  are applied, resulting in more efficient computation of the attention matrix.
  """

  def __init__(self,
               feature_transform="exp",
               num_random_features=256,
               seed=0,
               redraw=False,
               is_short_seq=False,
               begin_kernel=0,
               **kwargs):
    r"""Constructor of KernelAttention.

    Args:
      feature_transform: A non-linear transform of the keys and quries.
      Possible transforms are "elu", "relu", "square", "exp", "expmod",
      "l2", "identity". If <is_short_seq> = True, it is recommended to choose
      feature_transform as "l2".
      num_random_features: Number of random features to be used for projection.
        if num_random_features <= 0, no production is used before transform.
      seed: The seed to begin drawing random features. Once the seed is set, the
        psedo number generation is determinisitc. Users should pass different
        seed for different layers. For multi-worker, each layer will use the
        same projection at each step.
      redraw: Whether to redraw projection every forward pass during training.
        The argument is only effective when num_random_features > 0.
      is_short_seq: boolean predicate indicating whether input data consists of
        very short sequences or not; in most cases this should be False
        (default option).
      begin_kernel: Apply kernel_attention after this sequence id and apply
        softmax attention before this.
      **kwargs: The same arguments `MultiHeadAttention` layer.
    """
    if feature_transform not in _TRANSFORM_MAP:
      raise ValueError("Unsupported feature_transform. The supported "
                       "feature_transform are %s. "
                       "Got '%s'." % (_TRANSFORM_MAP.keys(), feature_transform))
    if num_random_features <= 0 and redraw:
      raise ValueError(
          "There is nothing to redraw when num_random_features <= 0.")
    self._feature_transform = feature_transform
    self._num_random_features = num_random_features
    self._redraw = redraw
    self._is_short_seq = is_short_seq
    self._begin_kernel = begin_kernel
    # We use the seed for two scenarios:
    # 1. inference
    # 2. no redraw
    self._seed = seed

    super().__init__(**kwargs)
    self._projection_matrix = None
    if num_random_features > 0:
      self._projection_matrix = create_projection_matrix(
          self._num_random_features, self._key_dim,
          tf.constant([self._seed, self._seed + 1]))

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         feature_transform,
                         is_short_seq,
                         attention_mask=None,
                         training=False,
                         numeric_stabler=_NUMERIC_STABLER):
    """Applies kernel attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for customized
    attention implementation.

    Args:
      query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
      key: Projected key `Tensor` of shape `[B, S, N, key_dim]`.
      value: Projected value `Tensor` of shape `[B, S, N, value_dim]`.
      feature_transform: A non-linear transform of the keys and quries.
      is_short_seq: boolean predicate indicating whether input data consists of
        short or long sequences; usually short sequence is defined as having
        length L <= 1024.
      attention_mask: a boolean mask of shape `[B, S]`, that prevents
        attenting to masked positions. Note that the mask is only appied to
        the keys. User may want to mask the output if query contains pads.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
      numeric_stabler: A scalar value added to avoid divide by 0.

    Returns:
      attention_output: Multi-headed outputs of attention computation.
    """

    projection_matrix = None
    if self._num_random_features > 0:
      if self._redraw and training:
        projection_matrix = create_projection_matrix(self._num_random_features,
                                                     self._key_dim)
      else:
        projection_matrix = self._projection_matrix

    key = _TRANSFORM_MAP[feature_transform](key, projection_matrix, False)
    query = _TRANSFORM_MAP[feature_transform](query, projection_matrix, True)

    if attention_mask is not None:
      key = tf.einsum("BSNH,BS->BSNH", key, attention_mask)

    if is_short_seq:
      attention_scores = tf.einsum("BTNH,BSNH->BTSN", query, key)
      attention_scores = tf.nn.softmax(attention_scores, axis=2)
      attention_output = tf.einsum("BTSN,BSNH->BTNH", attention_scores, value)
      return attention_output
    else:
      kv = tf.einsum("BSNH,BSND->BNDH", key, value)
      denominator = 1.0 / (
          tf.einsum("BTNH,BNH->BTN", query, tf.reduce_sum(key, axis=1)) +
          _NUMERIC_STABLER)
      return tf.einsum("BTNH,BNDH,BTN->BTND", query, kv, denominator)

  def _build_from_signature(self, query, value, key=None):
    super()._build_from_signature(query=query, value=value, key=key)
    if self._begin_kernel > 0:
      common_kwargs = dict(
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activity_regularizer=self._activity_regularizer,
          kernel_constraint=self._kernel_constraint,
          bias_constraint=self._bias_constraint)
      self._output_dense_softmax = self._make_output_dense(
          self._query_shape.rank - 1, common_kwargs,
          name="attention_output_softmax")
      self._dropout_softmax = tf.keras.layers.Dropout(rate=self._dropout)

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           training=False):
    """Compute attention with kernel mechanism.

    Args:
      query: Query `Tensor` of shape `[B, T, dim]`.
      value: Value `Tensor` of shape `[B, S, dim]`.
      key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `[B, S]`, that prevents
        attenting to masked positions. Note that the mask is only appied to
        the keys. User may want to mask the output if query contains pads.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      Multi-headed outputs of attention computation.
    """
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, D]
    value = self._value_dense(value)

    if self._begin_kernel > 0:
      attention_output_softmax = self._compute_attention(
          query[:, :self._begin_kernel],
          key, value, "identity", True, attention_mask, training)
      attention_output_softmax = self._dropout_softmax(attention_output_softmax)
      attention_output_softmax = self._output_dense_softmax(
          attention_output_softmax)

      attention_output_kernel = self._compute_attention(
          query[:, self._begin_kernel:],
          key, value, self._feature_transform, self._is_short_seq,
          attention_mask, training)
      attention_output_kernel = self._dropout_layer(attention_output_kernel)
      attention_output_kernel = self._output_dense(
          attention_output_kernel)
      attention_output = tf.concat(
          [attention_output_softmax, attention_output_kernel], axis=1)
    else:
      attention_output = self._compute_attention(
          query, key, value, self._feature_transform,
          self._is_short_seq, attention_mask, training)
      # This is actually dropping out entire tokens to attend to, which might
      # seem a bit unusual, but is taken from the original Transformer paper.
      attention_output = self._dropout_layer(attention_output)
      attention_output = self._output_dense(attention_output)
    return attention_output

  def get_config(self):
    config = {
        "feature_transform": self._feature_transform,
        "num_random_features": self._num_random_features,
        "seed": self._seed,
        "redraw": self._redraw,
        "is_short_seq": self._is_short_seq,
        "begin_kernel": self._begin_kernel,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
