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

"""Roformer attention layer."""
# pylint: disable=g-classes-have-attributes
import tensorflow as tf, tf_keras

EinsumDense = tf_keras.layers.EinsumDense
MultiHeadAttention = tf_keras.layers.MultiHeadAttention


def _build_trig_vector(length, key_dim):
  """Builds the trig vector."""
  tf_dtype = tf_keras.mixed_precision.global_policy().compute_dtype
  position_ids = tf.cast(tf.range(length), dtype=tf_dtype)
  position_ids = tf.expand_dims(position_ids, axis=0)
  steps = key_dim // 2
  # 2 (i - 1) / key_dim = (i - 1) / steps: (-1 achieved with zero-indexing)
  wavenumber_exponent = -tf.cast(tf.range(steps), dtype=tf_dtype) / steps
  wavenumbers = tf.pow(
      tf.constant(10000.0, dtype=tf_dtype), wavenumber_exponent
  )
  vec = tf.einsum('bl,d->bld', position_ids, wavenumbers)
  sin_vec = tf.repeat(tf.sin(vec), repeats=2, axis=-1)
  cos_vec = tf.repeat(tf.cos(vec), repeats=2, axis=-1)
  sin_vec, cos_vec = tf.expand_dims(sin_vec, 2), tf.expand_dims(cos_vec, 2)
  return sin_vec, cos_vec


@tf_keras.utils.register_keras_serializable(package='Text')
class RoformerAttention(tf_keras.layers.MultiHeadAttention):
  """Roformer Attention."""

  def __init__(self,
               q_max_sequence_length,
               kv_max_sequence_length,
               output_range=None,
               **kwargs):
    """Instantiates a roformer attention layer.

    Roformer paper: https://arxiv.org/abs/2104.09864

    Args:
      q_max_sequence_length: maximum length in input for the query
      kv_max_sequence_length: maximum length in input for key and value, can be
        different from q_max_sequence_length
      output_range: length of the query tensor to consider.
      **kwargs: other keyword arguments.
    """
    super().__init__(**kwargs)
    self._q_max_sequence_length = q_max_sequence_length
    self._kv_max_sequence_length = kv_max_sequence_length
    assert self._key_dim % 2 == 0
    q_sin_vec, q_cos_vec = _build_trig_vector(self._q_max_sequence_length,
                                              self._key_dim)
    k_sin_vec, k_cos_vec = _build_trig_vector(self._kv_max_sequence_length,
                                              self._key_dim)
    # pylint:disable=g-long-ternary
    self.q_sin_vec, self.q_cos_vec = (q_sin_vec,
                                      q_cos_vec) if output_range is None else (
                                          q_sin_vec[:, 0:output_range, ...],
                                          q_cos_vec[:, 0:output_range, ...])
    # pylint:enable=g-long-ternary
    self.k_sin_vec, self.k_cos_vec = (k_sin_vec, k_cos_vec)

  def roformer_recompute_qkv(self, q, k, v):
    q_shape = tf.shape(q)
    q_len = q_shape[1]
    k_shape = tf.shape(k)
    k_len = k_shape[1]

    q2 = tf.stack([-q[..., 1::2], q[..., ::2]], axis=4)
    q2 = tf.reshape(q2, q_shape)
    k2 = tf.stack([-k[..., 1::2], k[..., ::2]], axis=4)
    k2 = tf.reshape(k2, k_shape)
    ret_q = q * self.q_cos_vec[:, 0:q_len,
                               ...] + q2 * self.q_sin_vec[:, 0:q_len, ...]
    ret_w = k * self.k_cos_vec[:, 0:k_len,
                               ...] + k2 * self.k_sin_vec[:, 0:k_len, ...]
    return ret_q, ret_w, v

  def call(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None):
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    query = self._query_dense(query)
    key = self._key_dense(key)
    value = self._value_dense(value)

    query, key, value = self.roformer_recompute_qkv(query, key, value)

    attention_output, attention_scores = self._compute_attention(
        query, key, value, attention_mask, training)
    attention_output = self._output_dense(attention_output)

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output
