# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
import logging

import tensorflow as tf, tf_keras


def _large_compatible_negative(tensor_type):
  """Large negative number as Tensor.

  This function is necessary because the standard value for epsilon
  in this module (-1e9) cannot be represented using tf.float16

  Args:
      tensor_type: a dtype to determine the type.

  Returns:
      a large negative number.
  """
  # In case of dtype=float16 (e.g., for mixed-precision), the largest
  # negative number (dtypes.float16.min) is divided by 2, in order to
  # avoid overflows when summing negative inputs.
  if tensor_type == tf.float16:
    return tf.float16.min / 2.0
  return -1e9


class MultiHeadAttention(tf_keras.layers.MultiHeadAttention):
  """Multi-head block sparse attention layer."""

  def __init__(
      self,
      src_block_size=None,
      tgt_block_size=None,
      use_sigmoid_attn=False,
      sigmoid_attn_bias=None,
      num_kv_heads=None,
      **kwargs
  ):
    """Initializes the block sparse attention layer.

    Args:
      src_block_size: The block size of the query. An integer that divides the
        sequence length into blocks.
      tgt_block_size: The block size of the key/value. An integer that divides
        the sequence length into blocks. The number of blocks in the source and
        target must be the same.
      use_sigmoid_attn: If enabled, uses sigmoid instead of softmax to compute
        attn probs. https://arxiv.org/pdf/2409.04431
      sigmoid_attn_bias: Bias for sigmoid attn. Suggested value -ln(seq_len).
      num_kv_heads: Number of key/value heads in the multi-head self attention.
        Refer to multi_query_attention.py for more details.
      **kwargs: Args passed to the base class.
    """
    super().__init__(**kwargs)
    if src_block_size is None or src_block_size <= 0:
      raise ValueError("src_block_size must be specified.")
    self._src_block_size = src_block_size
    self._tgt_block_size = tgt_block_size or self._src_block_size
    self._num_kv_heads = num_kv_heads
    if num_kv_heads is not None and num_kv_heads != 1:
      raise ValueError(
          "num_kv_heads must be 1. Grouped-query attention is not supported."
      )
    self._use_sigmoid_attn = use_sigmoid_attn
    self._sigmoid_attn_bias = sigmoid_attn_bias
    if self._use_sigmoid_attn:
      if self._sigmoid_attn_bias is None:
        raise ValueError(
            "sigmoid_attn_bias must be specified for sigmoid attn."
        )

  def get_config(self):
    config = super().get_config()
    config.update({
        "src_block_size": self._src_block_size,
        "tgt_block_size": self._tgt_block_size,
        "use_sigmoid_attn": self._use_sigmoid_attn,
        "sigmoid_attn_bias": self._sigmoid_attn_bias,
        "num_kv_heads": self._num_kv_heads,
    })
    return config

  def _build_from_signature(self, query, value, key=None):
    # pytype: disable=attribute-error
    super()._build_from_signature(query, value, key)
    # pytype: enable=attribute-error
    # If block sizes are same as sequence lengths, we defer to default attn.
    if (
        self._query_shape[-2] == self._src_block_size
        and self._key_shape[-2] == self._tgt_block_size
    ):
      return
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
      if self._num_kv_heads == 1:
        self._key_dense = tf_keras.layers.EinsumDense(
            "BTD,DH->BTH",
            output_shape=[None, self._key_dim],
            bias_axes="H" if self._use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense = tf_keras.layers.EinsumDense(
            "BTD,DH->BTH",
            output_shape=[None, self._value_dim],
            bias_axes="H" if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
      else:
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
      if self._key_shape[-2] == self._tgt_block_size:
        if self._num_kv_heads == 1:
          self._dot_product_equation = "BsH,BNLtH->BNLts"
          self._combine_equation = "BNLts,BsH->BNLtH"
        else:
          self._dot_product_equation = "BNsH,BNLtH->BNLts"
          self._combine_equation = "BNLts,BNsH->BNLtH"
      else:
        if self._num_kv_heads == 1:
          self._dot_product_equation = "BLsH,BNLtH->BNLts"
          self._combine_equation = "BNLts,BLsH->BNLtH"
        else:
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
    tgt_num_blocks = self._key_shape[-2] // self._tgt_block_size
    if tgt_num_blocks == 1:
      src_num_blocks = self._query_shape[-2] // self._src_block_size
      result = tf.reshape(
          attention_mask,
          [-1, src_num_blocks, self._src_block_size, self._tgt_block_size],
      )
    else:
      attention_mask = tf.cast(attention_mask[:, 0, :], dtype=dtype)
      attention_mask = tf.reshape(
          attention_mask,
          [
              -1,
              tgt_num_blocks,
              self._tgt_block_size,
          ],
      )
      result = tf.einsum("BLQ,BLK->BLQK", attention_mask, attention_mask)
    return result

  def _masked_softmax(self, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, L, T, S]
    if attention_mask is not None:
      # `attention_mask` = [B, 1, L, T, S]
      attention_mask = tf.expand_dims(attention_mask, axis=1)
    if self._use_sigmoid_attn:
      if attention_mask is not None:
        adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * (
            _large_compatible_negative(attention_scores.dtype)
        )
        attention_scores += adder
      attention_scores += self._sigmoid_attn_bias
      return tf_keras.activations.sigmoid(attention_scores)
    else:
      return self._softmax(attention_scores, attention_mask)

  def _compute_attention(
      self, query, key, value, attention_mask=None, training=None
  ):
    # If block sizes are same as sequence lengths, we defer to default attn.
    if (
        self._query_shape[-2] == self._src_block_size
        and self._key_shape[-2] == self._tgt_block_size
    ):
      logging.info(
          "Computing default attention as block sizes are equal to sequence"
          " lengths."
      )
      # pytype: disable=attribute-error
      return super()._compute_attention(
          query,
          key,
          value,
          attention_mask=attention_mask,
          training=training,
      )
      # pytype: enable=attribute-error
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

    if src_num_blocks != tgt_num_blocks and tgt_num_blocks != 1:
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
    if tgt_num_blocks != 1 and self._num_kv_heads != 1:
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
    elif tgt_num_blocks != 1 and self._num_kv_heads == 1:
      key_blocks = tf.reshape(key, [
          -1,
          tgt_num_blocks,
          self._tgt_block_size,
          self._key_dim,
      ])
      value_blocks = tf.reshape(value, [
          -1,
          tgt_num_blocks,
          self._tgt_block_size,
          self._value_dim,
      ])
    else:
      key_blocks = key
      value_blocks = value
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
