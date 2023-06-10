# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Keras-based MegaEncoder block layer."""

from typing import Any

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.lra.exponential_moving_average import MultiHeadEMA


def get_activation_fn(activation):
  ## Helper Function for Activation
  if activation == "silu":
    return tf.nn.silu
  elif activation == "softmax":
    return tf.nn.softmax
  else:
    raise NotImplementedError
  return


class RelativePositionBias(tf.keras.layers.Layer):
  """Relative position embedding layer with bias."""

  def __init__(self, max_positions):
    super().__init__()
    self.max_positions = max_positions

  def build(self, input_shape):
    gauss_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    self.rel_pos_bias = tf.Variable(
        gauss_init(shape=[2 * self.max_positions - 1], dtype=tf.float32),
        trainable=True,
    )

  def call(self, seq_len):
    if seq_len is None:
      seq_len = self.max_positions
    seq_len = tf.get_static_value(seq_len)
    # seq_len * 2 -1
    b = self.rel_pos_bias[
        (self.max_positions - seq_len) : (self.max_positions + seq_len - 1)
    ]
    # seq_len * 3 - 1
    t = tf.pad(b, paddings=tf.constant([[0, seq_len]]))
    # (seq_len * 3 - 1) * seq_len
    t = tf.tile(t, (seq_len,))
    t = t[:-seq_len]
    # seq_len x (3 * seq_len - 2)
    t = tf.reshape(t, shape=(seq_len, 3 * seq_len - 2))
    r = (2 * seq_len - 1) // 2
    start = r
    end = t.shape[1] - r
    t = t[:, start:end]
    return t


class MovingAverageGatedAttention(tf.keras.layers.Layer):
  """MegaEncoderBlock layer.

  This layer implements the Mega Encoder from
  "Mega: Moving Average Equipped Gated Attention".
  (https://arxiv.org/abs/2209.10655)
  """

  def __init__(
      self,
      embed_dim,
      zdim,
      hdim,
      ndim,
      intermediate_size,
      inner_activation=None,
      dropout=0.0,
      attention_dropout=0.0,
      hidden_dropout=0.0,
      activation="silu",
      bidirectional=False,
      truncation=None,
      prenorm=True,
      max_positions=1024,
      use_bias=True,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
      attention_initializer=None,
      attention_axes=None,
      return_attention_scores=False,
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
  ):
    self.embed_dim = embed_dim
    self.hdim = hdim
    self.zdim = zdim
    self.ndim = ndim
    self.inner_dim = intermediate_size
    self.activation = get_activation_fn(activation=activation)
    self.inner_activation = inner_activation
    self.scaling = self.zdim**-0.5

    self.dropout = tf.keras.layers.Dropout(rate=dropout)
    self.hidden_dropout = tf.keras.layers.Dropout(rate=hidden_dropout)
    self.attention_dropout_rate = attention_dropout
    self.attention_dropout = tf.keras.layers.Dropout(rate=attention_dropout)

    self.ffn_intermediate_dropout = tf.keras.layers.Dropout(rate=hidden_dropout)
    self.output_dropout = tf.keras.layers.Dropout(rate=hidden_dropout)

    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
          attention_initializer
      )
    else:
      self._attention_initializer = tf_utils.clone_initializer(
          self._kernel_initializer
      )
    self._attention_axes = attention_axes
    self._use_bias = use_bias
    self.return_attention_scores = return_attention_scores

    self.prenorm = prenorm
    self.norm = tf.keras.layers.LayerNormalization(axis=-1)
    self.ffn_norm = tf.keras.layers.LayerNormalization(axis=-1)

    self.move = MultiHeadEMA(
        embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation
    )

    self.max_positions = max_positions
    super().__init__()

  def build(self, input_shape):
    gauss_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    zero_init = tf.keras.initializers.Zeros()

    self.v_proj = tf.keras.layers.Dense(
        self.hdim,
        activation=None,
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(gauss_init),
        bias_initializer=tf_utils.clone_initializer(zero_init),
        name="v_proj",
    )

    self.mx_proj = tf.keras.layers.Dense(
        self.zdim + self.hdim + 2 * self.embed_dim,
        activation=None,
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(gauss_init),
        bias_initializer=tf_utils.clone_initializer(zero_init),
        name="mx_proj",
    )

    self.h_proj = tf.keras.layers.Dense(
        self.embed_dim,
        activation=None,
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(gauss_init),
        bias_initializer=tf_utils.clone_initializer(zero_init),
        name="h_proj",
    )

    self._intermediate_dense = tf.keras.layers.Dense(
        self.inner_dim, use_bias=True
    )

    self._output_dense = tf.keras.layers.Dense(self.embed_dim, use_bias=True)

    policy = tf.keras.mixed_precision.global_policy()
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self.inner_activation, dtype=policy
    )

    self.gamma = tf.Variable(
        gauss_init(shape=[2, self.zdim], dtype=tf.float32), trainable=True
    )
    self.beta = tf.Variable(
        zero_init(shape=[2, self.zdim], dtype=tf.float32), trainable=True
    )

    self.rel_pos_bias = RelativePositionBias(max_positions=self.max_positions)

    super().build(input_shape)

  def get_config(self):
    base_config = super().get_config()
    base_config.update({
        "embed_dim": self.embed_dim,
        "zdim": self.zdim,
        "hdim": self.hdim,
        "dropout": self.dropout,
        "attention_dropout": self.attention_dropout_rate,
        "kernel_initializer": tf.keras.initializers.serialize(
            self._kernel_initializer
        ),
        "bias_initializer": tf.keras.initializers.serialize(
            self._bias_initializer
        ),
        "use_bias": self._use_bias,
        "prenorm": self.prenorm,
        "max_positions": self.max_positions,
        "attention_initializer": tf.keras.initializers.serialize(
            self._attention_initializer
        ),
        "attention_axes": self._attention_axes,
        "return_attention_scores": self.return_attention_scores,
    })
    return base_config

  def _softmax_attention(self, q, k):
    slen = k.shape[1]
    # C x C
    if slen is None:
      slen = 2
    bias = self.rel_pos_bias(slen)

    # scaled attention
    q = q * self.scaling
    # B x K x C x C
    qk = tf.matmul(q, tf.transpose(k, perm=(0, 2, 1))) + bias

    attn_weights = tf.nn.softmax(qk, axis=-1)
    return attn_weights

  def call(self, inputs: Any) -> Any:
    """MEGA encoder block call.

    Args:
      inputs: a single tensor or a list of tensors. `input tensor`
        as the single sequence of embeddings. [`input tensor`,
        `attention mask`] to have the
        additional attention mask. [`query tensor`, `key value tensor`,
        `attention mask`] to have separate input streams for the query, and
        key/value to the multi-head attention.
    Returns:
      An output tensor with the same dimensions as input/query tensor.
    """
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        (input_tensor, attention_mask) = inputs
        key_value = None
      elif len(inputs) == 3:
        (input_tensor, key_value, attention_mask) = inputs
      else:
        raise ValueError(
            "Unexpected inputs to %s with length at %d"
            % (self.__class__, len(inputs))
        )
    else:
      (input_tensor, key_value, attention_mask) = (inputs, None, None)

    if self.prenorm:
      input_tensor = self.norm(input_tensor)
      if key_value is not None:
        key_value = self.norm(key_value)

    ## B*L*D -> L*B*D
    ## Multi-Dimensional Damped EMA
    x = tf.transpose(input_tensor, perm=[1, 0, 2])
    residual = x

    seq_len, bsz, _ = x.shape

    # L x B x E
    v = self.activation(self.v_proj(x))

    # L x B x D
    mx = self.move(x, attention_mask)
    mx = self.dropout(mx)

    # L x B x D -> L x B x (2*D+S+E)
    base = self.mx_proj(mx)

    u, zr, hx = tf.split(
        base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], axis=-1
    )
    # L x B x D
    u = tf.math.sigmoid(u)
    # L x B x (E+S)
    z, r = tf.split(tf.nn.silu(zr), [self.zdim, self.hdim], axis=-1)
    # L x B x S -> L x B x 1 x S -> L x B x 2 x S
    z = tf.expand_dims(z, axis=2) * self.gamma + self.beta
    # L x B x 2 x S -> L x B x S
    q, k = tf.unstack(z, axis=2)

    # L x B x D -> B x L x D
    q = tf.transpose(q, perm=(1, 0, 2))
    k = tf.transpose(k, perm=(1, 0, 2))
    # L x B x E -> B x L x E
    v = tf.transpose(v, perm=(1, 0, 2))

    attn_weights = self._softmax_attention(q, k)
    v = self.hidden_dropout(v)
    kernel = tf.squeeze(self.attention_dropout(attn_weights))
    # B x K x C x E -> B x L x E -> L x B x E
    h = tf.transpose(
        tf.reshape(
            tf.linalg.matmul(kernel, v), shape=(bsz, seq_len, self.hdim)
        ),
        perm=(1, 0, 2),
    )

    # L x B x E -> L x B x D
    h = self.activation(hx + self.h_proj(h * r))
    h = self.dropout(h)
    # L x B x D
    out = residual + tf.math.multiply(u, h - residual)

    if not self.prenorm:
      out = self.norm(out)

    out = tf.transpose(out, perm=(1, 0, 2))

    if self.prenorm:
      out = self.ffn_norm(out)

    inner_output = self._intermediate_dense(out)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self.ffn_intermediate_dropout(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self.output_dropout(layer_output) + out

    if not self.prenorm:
      layer_output = self.ffn_norm(layer_output)

    return layer_output
