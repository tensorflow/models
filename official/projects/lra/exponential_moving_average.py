# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from typing import Optional
import tensorflow as tf, tf_keras


class MultiHeadEMA(tf_keras.layers.Layer):
  """Exponential Moving Average Layer.

  See "https://arxiv.org/abs/2209.10655" for more details.
  """

  def __init__(
      self, embed_dim, ndim=2, bidirectional=False, truncation=None, **kwargs
  ):
    super().__init__(**kwargs)

    self.embed_dim = embed_dim
    self.ndim = ndim
    self.bidirectional = bidirectional
    self.truncation = truncation
    self.scale = tf.math.sqrt(1.0 / self.ndim)

    self.kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim

    self._kernel = None
    self._coeffs = None

  def build(self, input_shape):
    self.damping_factor = self.add_weight(
        shape=(self.kernel_dim, self.ndim, 1),
        initializer="random_normal",
        trainable=True,
        name="damping_factor",
        dtype=tf.float32,
    )
    self.decay_factor = self.add_weight(
        shape=(self.kernel_dim, self.ndim, 1),
        initializer="random_normal",
        trainable=True,
        name="decay_factor",
        dtype=tf.float32,
    )
    self.ema_expansion_matrix = self.add_weight(
        shape=(self.kernel_dim, self.ndim, 1),
        initializer="random_normal",
        trainable=True,
        name="ema_expansion_matrix",
        dtype=tf.float32,
    )
    self.kernel_projection_matrix = self.add_weight(
        shape=(self.kernel_dim, self.ndim),
        initializer="random_normal",
        trainable=True,
        name="kernel_projection_matrix",
        dtype=tf.float32,
    )
    self.residual_weight = self.add_weight(
        shape=(self.embed_dim,),
        initializer="ones",
        trainable=True,
        name="residual_weight",
        dtype=tf.float32,
    )

    super().build(input_shape)

  def _calc_coeffs(self):
    self._coeffs = None
    # D x N x 1
    damping_factor = tf.math.sigmoid(self.damping_factor)
    decay_factor = tf.math.sigmoid(self.decay_factor)
    previous_timestep_weight = 1.0 - damping_factor * decay_factor
    return damping_factor, previous_timestep_weight

  def _compute_kernel(self, length: int):
    self._kernel = None
    # D x N x 1
    damping_factor, previous_timestep_weight = self._calc_coeffs()
    # D x N x L
    vander = tf.cast(
        tf.reshape(tf.range(length), shape=(1, 1, length)),
        dtype=damping_factor.dtype,
    ) * tf.math.log(previous_timestep_weight)
    kernel = (damping_factor * self.ema_expansion_matrix) * tf.math.exp(vander)
    # D x L
    return tf.einsum(
        "dnl,dn->dl", kernel, self.kernel_projection_matrix * self.scale
    )

  def coeffs(self):
    if self.training:
      return self._calc_coeffs()
    else:
      if self._coeffs is None:
        self._coeffs = self._calc_coeffs()
      return self._coeffs

  def kernel(self, length: int):
    assert self.truncation is None, "WEIRD!"
    kernel_size = (
        length if self.truncation is None else min(self.truncation, length)
    )
    return self._compute_kernel(kernel_size)

  def call(self, x, padding_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Input shape: Time x Batch x Channel.

    Args:
      x: Tensor input.
      padding_mask (ByteTensor, optional): mask to exclude keys that are pads,
        of shape `(batch, src_len)`, where padding elements are indicated by
        1s.
    Returns:
      transformed: transformed Tensor.
    """

    seq_len, _, embed_dim = x.shape
    assert embed_dim == self.embed_dim
    if seq_len is None:
      seq_len = 1

    # L x B x D
    residual = x * self.residual_weight

    # L x B x D -> B x D x L
    x = tf.transpose(x, perm=(1, 2, 0))

    # Masking of the tensor
    if padding_mask is not None:
      x = x * tf.cast(tf.expand_dims(padding_mask, axis=1), x.dtype)

    k = self.kernel(seq_len)

    kernel_size = k.shape[1]
    fft_len = seq_len
    s = 0

    if self.bidirectional:
      k1, k2 = tf.split(k, [self.embed_dim, self.embed_dim], axis=0)
      # D x 2*L-1
      padding_l = tf.constant([[0, 0], [kernel_size - 1, 0]])
      padding_r = tf.constant([[0, 0], [0, kernel_size - 1]])
      padding_x = tf.constant([[0, 0], [0, 0], [kernel_size - 1, 0]])
      k = tf.pad(k1, padding_l) + tf.pad(tf.reverse(k2, axis=[-1]), padding_r)
      x = tf.pad(x, padding_x)
      fft_len = fft_len + kernel_size - 1
      s = 2 * kernel_size - 2

    k_f = tf.signal.rfft(
        k, fft_length=tf.constant([2 * fft_len], dtype=tf.int32)
    )
    x_f = tf.signal.rfft(
        x, fft_length=tf.constant([2 * fft_len], dtype=tf.int32)
    )
    # B x D x L
    out = tf.signal.irfft(
        x_f * k_f, fft_length=tf.constant([2 * fft_len], dtype=tf.int32)
    )[..., s : s + seq_len]

    # B x D x L -> L x B x D
    out = tf.nn.silu(tf.transpose(out, perm=(2, 0, 1)) + residual)
    return out
