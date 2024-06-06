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

"""Block sparse attention converts query/key/value into blocks and performs diagonal block sparse attention."""
import collections

import tensorflow as tf, tf_keras


class MultiHeadAttention(tf_keras.layers.MultiHeadAttention):
  """Multi-head block sparse attention layer."""

  def __init__(self, src_block_size=None, tgt_block_size=None, **kwargs):
    """Initializes the block sparse attention layer.

    Args:
      src_block_size: The block size of the query. An integer that divides the
        sequence length into blocks.
      tgt_block_size: The block size of the key/value. An integer that divides
        the sequence length into blocks. The number of blocks in the source and
        target must be the same.
      **kwargs: Args passed to the base class.
    """
    super().__init__(**kwargs)
    if src_block_size is None or src_block_size <= 0:
      raise ValueError("src_block_size must be specified.")
    self._src_block_size = src_block_size
    self._tgt_block_size = tgt_block_size or self._src_block_size

  def _build_from_signature(self, query, value, key=None):
    # pytype: disable=attribute-error
    super()._build_from_signature(query, value, key)
    # pytype: enable=attribute-error
    # The following capital letters are used to denote the tensor dimension
    # parameters:
    # B = batch size
    # S = length of the key/value (target)
    # D = model dimension.
    # T = length of the query (source)
    # t = block size of the source.
    # s = block size of the target.
    # L = number of blocks in the source/target.
    # N = number of attention heads
    # H = dimensions of each attention head.
    with tf.init_scope():
      proj_einsum_eqn = "BTD,DNH->BNTH"
      bias_axes = "NH"
      qk_output_shape = [
          self._num_heads,
          None,
          self._key_dim,
      ]
      v_output_shape = [
          self._num_heads,
          None,
          self._value_dim,
      ]
      self._query_dense = tf_keras.layers.EinsumDense(
          proj_einsum_eqn,
          output_shape=qk_output_shape,
          bias_axes=bias_axes if self._use_bias else None,
          name="query",
          **self._get_common_kwargs_for_sublayer(),
      )
      self._key_dense = tf_keras.layers.EinsumDense(
          proj_einsum_eqn,
          output_shape=qk_output_shape,
          bias_axes=bias_axes if self._use_bias else None,
          name="key",
          **self._get_common_kwargs_for_sublayer(),
      )
      self._value_dense = tf_keras.layers.EinsumDense(
          proj_einsum_eqn,
          output_shape=v_output_shape,
          bias_axes=bias_axes if self._use_bias else None,
          name="value",
          **self._get_common_kwargs_for_sublayer(),
      )
      self._dot_product_equation = "BNLsH,BNLtH->BNLts"
      self._combine_equation = "BNLts,BNLsH->BNLtH"
      if self._output_shape:
        if not isinstance(self._output_shape, collections.abc.Sized):
          output_shape = [self._output_shape]
        else:
          output_shape = self._output_shape
      else:
        output_shape = [self._query_shape[-1]]
      output_shape = [None] + output_shape
      self._output_dense = tf_keras.layers.EinsumDense(
          "BNTH,DNH->BTD",
          output_shape=output_shape,
          bias_axes="D" if self._use_bias else None,
          name="attention_output",
          **self._get_common_kwargs_for_sublayer(),
      )

  def _block_diagonal_mask(self, attention_mask, dtype=None):
    """Converts the attention mask to block diagonal."""
    # Uses the same key mask for the entire query sequence since softmax
    # is applied only on the key axis.
    attention_mask = tf.cast(attention_mask[:, 0, :], dtype=dtype)
    tgt_num_blocks = self._key_shape[-2] // self._tgt_block_size
    attention_mask = tf.reshape(
        attention_mask,
        [
            -1,
            tgt_num_blocks,
            self._tgt_block_size,
        ],
    )
    return tf.einsum("BLQ,BLK->BLQK", attention_mask, attention_mask)

  def _masked_softmax(self, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, L, T, S]
    if attention_mask is not None:
      # `attention_mask` = [B, 1, L, T, S]
      attention_mask = tf.expand_dims(attention_mask, axis=1)
    return self._softmax(attention_scores, attention_mask)

  def _compute_attention(
      self, query, key, value, attention_mask=None, training=None
  ):
    # src_num_blocks and tgt_num_blocks are the number of blocks in the source
    # and target. Care should be taken to ensure that the number of blocks in
    # the source and target are the same.
    if self._query_shape[-2] % self._src_block_size != 0:
      raise ValueError(
          "query_shape[-2] must be divisible by src_block_size."
      )
    if self._key_shape[-2] % self._tgt_block_size != 0:
      raise ValueError(
          "key_shape[-2] must be divisible by tgt_block_size."
      )
    src_num_blocks = self._query_shape[-2] // self._src_block_size
    tgt_num_blocks = self._key_shape[-2] // self._tgt_block_size

    if src_num_blocks != tgt_num_blocks:
      raise ValueError(
          "src_num_blocks must be equal to tgt_num_blocks."
      )
    # Convert the query/key/value into blocks to perform block diagonal
    # attention.
    query_blocks = tf.reshape(query, [
        -1,
        self._num_heads,
        src_num_blocks,
        self._src_block_size,
        self._key_dim,
    ])
    key_blocks = tf.reshape(key, [
        -1,
        self._num_heads,
        tgt_num_blocks,
        self._tgt_block_size,
        self._key_dim,
    ])
    value_blocks = tf.reshape(value, [
        -1,
        self._num_heads,
        tgt_num_blocks,
        self._tgt_block_size,
        self._value_dim,
    ])
    if attention_mask is not None:
      attention_mask = self._block_diagonal_mask(attention_mask, key.dtype)
    # pytype: disable=attribute-error
    attention_output, attention_scores = super()._compute_attention(
        query_blocks,
        key_blocks,
        value_blocks,
        attention_mask=attention_mask,
        training=training,
    )
    # pytype: enable=attribute-error
    # Reshape the attention output to the original shape.
    attention_output = tf.reshape(attention_output, [
        -1,
        self._num_heads,
        self._query_shape[1],
        self._value_dim,
    ])
    return attention_output, attention_scores

  def call(
      self,
      query,
      value,
      key=None,
      attention_mask=None,
      return_attention_scores=False,
      training=None,
      use_causal_mask=False,
  ):
    if use_causal_mask:
      raise ValueError("use_causal_mask is not supported.")
    return super().call(
        query,
        value,
        key=key,
        attention_mask=attention_mask,
        return_attention_scores=return_attention_scores,
        training=training,
        use_causal_mask=use_causal_mask,
    )
