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

"""Longformer attention block. Modified From huggingface/transformers."""

# pylint: disable=g-classes-have-attributes

import math
import string

import numpy as np
import tensorflow as tf, tf_keras

from official.modeling.tf_utils import get_shape_list

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
    attn_axes: List/tuple of axes, `[-1, rank)`, that attention will be applied
      to.

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
  dot_product_equation = f"{source_notation},{target_notation}->{product_notation}"
  attn_scores_rank = len(product_notation)
  combine_equation = f"{product_notation},{source_notation}->{target_notation}"
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
  equation = f"{input_str},{kernel_str}->{output_str}"

  return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


@tf_keras.utils.register_keras_serializable(package="Text")
class LongformerAttention(tf_keras.layers.MultiHeadAttention):
  """LongformerAttention.

    Args:
      attention_window: int representing the window size for attention.
      layer_id: int of the id of the layer.
      global_attention_size: the size of global attention used for each token.
  """

  def __init__(self, attention_window, layer_id, global_attention_size,
               **kwargs):
    super().__init__(**kwargs)
    self._layer_id = layer_id
    self._attention_window = attention_window
    assert (self._attention_window % 2 == 0), (
        f"`attention_window` for layer {self._layer_id} has to be an even "
        f"value. Given {self.attention_window}")
    assert (self._attention_window > 0), (
        f"`attention_window` for layer {self._layer_id} has to be positive. "
        f"Given {self.attention_window}")
    self._one_sided_attn_window_size = self._attention_window // 2
    self.global_attention_size = global_attention_size

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
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    # Any setup work performed only once should happen in an `init_scope`
    # to avoid creating symbolic Tensors that will later pollute any eager
    # operations.
    # with tf_utils.maybe_init_scope(self):
    # TODO(crickwu): check whether tf_utils.maybe_init_scope(self) (keras)
    # is needed.
    free_dims = self._query_shape.rank - 1
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=1, output_dims=2)
    self._query_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        name="query",
        **common_kwargs)
    self._global_query_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        name="global_query",
        **common_kwargs)
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        self._key_shape.rank - 1, bound_dims=1, output_dims=2)
    self._key_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        name="key",
        **common_kwargs)
    self._global_key_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._key_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        name="global_key",
        **common_kwargs)
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        self._value_shape.rank - 1, bound_dims=1, output_dims=2)
    self._value_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._value_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        name="value",
        **common_kwargs)
    self._global_value_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1,
                                       [self._num_heads, self._value_dim]),
        bias_axes=bias_axes if self._use_bias else None,
        name="global_value",
        **common_kwargs)

    # Builds the attention computations for multi-head dot product attention.
    # These computations could be wrapped into the keras attention layer once
    # it support mult-head einsum computations.
    self._build_attention(output_rank)
    self._global_dropout_layer = tf_keras.layers.Dropout(rate=self._dropout)
    # self._output_dense = self._make_output_dense(
    #   free_dims, common_kwargs, "attention_output")
    self._output_dense = tf_keras.layers.Dense(
        units=self._num_heads * self._key_dim, name="dense", **common_kwargs)

  def call(self,
           hidden_states,
           attention_mask=None,
           is_index_masked=None,
           is_index_global_attn=None,
           training=None):
    """Applies Dot-product attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for customized
    attention implementation.
    Args:
      hidden_states: inputs for generating query, key and value tensors.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions.
      is_index_masked: boolean indicating whether the index is masked.
      is_index_global_attn: boolean indicating whether the index is global
        attention.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
    """
    if not self._built_from_signature:
      self._build_from_signature(
          query=hidden_states, value=hidden_states, key=hidden_states)

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(hidden_states)

    # `key` = [B, S, N, H]
    key = self._key_dense(hidden_states)

    # `value` = [B, S, N, H]
    value = self._value_dense(hidden_states)

    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))
    batch_size, seq_len, num_heads, head_dim = get_shape_list(query)

    # attn_probs = (batch_size, seq_len, num_heads, window*2+1)
    attn_scores = self._sliding_chunks_query_key_matmul(
        query, key, self._one_sided_attn_window_size)

    # diagonal mask with zeros everywhere and -inf inplace of padding
    diagonal_mask = self._sliding_chunks_query_key_matmul(
        tf.ones(get_shape_list(attention_mask)),
        attention_mask,
        self._one_sided_attn_window_size,
    )

    # pad local attention probs
    attn_scores += diagonal_mask

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          get_shape_list(attn_scores),
          [
              batch_size, seq_len, self._num_heads,
              self._one_sided_attn_window_size * 2 + 1
          ],
          message=f"attn_probs should be of size "
          f"({batch_size}, {seq_len}, {num_heads}, "
          f"{self._one_sided_attn_window_size * 2 + 1}),"
          f" but is of size {get_shape_list(attn_scores)}",
      )

    # compute global attn indices required through out forward fn
    (
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ) = self._get_global_attn_indices(is_index_global_attn,
                                      self.global_attention_size)
    # this function is only relevant for global attention
    if self.global_attention_size > 0:
      attn_scores = self._concat_with_global_key_attn_probs(
          attn_scores=attn_scores,
          query_vectors=query,
          key_vectors=key,
          max_num_global_attn_indices=max_num_global_attn_indices,
          is_index_global_attn_nonzero=is_index_global_attn_nonzero,
          is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
          is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      )
    else:
      pass

    attn_probs = tf.nn.softmax(attn_scores, axis=-1)

    # softmax sometimes inserts NaN if all positions are masked,
    # replace them with 0
    # Make sure to create a mask with the proper shape:
    # if is_global_attn==True => [batch_size, seq_len, self.num_heads,
    # self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
    # if is_global_attn==False => [batch_size, seq_len, self.num_heads,
    # self.one_sided_attn_window_size * 2 + 1]
    if self.global_attention_size > 0:
      masked_index = tf.tile(
          is_index_masked[:, :, None, None],
          (1, 1, self._num_heads, self._one_sided_attn_window_size * 2 +
           max_num_global_attn_indices + 1),
      )
    else:
      masked_index = tf.tile(
          is_index_masked[:, :, None, None],
          (1, 1, self._num_heads, self._one_sided_attn_window_size * 2 + 1),
      )

    attn_probs = tf.where(
        masked_index,
        tf.zeros(get_shape_list(masked_index), dtype=attn_probs.dtype),
        attn_probs,
    )

    layer_head_mask = None
    if layer_head_mask is not None:
      if tf.executing_eagerly():
        tf.debugging.assert_equal(
            get_shape_list(layer_head_mask),
            [self._num_heads],
            message=f"Head mask for a single layer should be of size "
            f"{(self._num_heads)}, but is "
            f"{get_shape_list(layer_head_mask)}",
        )

      attn_probs = tf.reshape(layer_head_mask, (1, 1, -1, 1)) * attn_probs

    # apply dropout
    attn_probs = self._dropout_layer(attn_probs, training=training)
    value_vectors = tf.reshape(
        value, (batch_size, seq_len, self._num_heads, self._key_dim))

    # if global attention, compute sum of global and local attn
    if self.global_attention_size > 0:
      attn_output = self._compute_attn_output_with_global_indices(
          value_vectors=value_vectors,
          attn_probs=attn_probs,
          max_num_global_attn_indices=max_num_global_attn_indices,
          is_index_global_attn_nonzero=is_index_global_attn_nonzero,
          is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      )
    else:
      attn_output = self._sliding_chunks_matmul_attn_probs_value(
          attn_probs, value_vectors, self._one_sided_attn_window_size)

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          get_shape_list(attn_output),
          [batch_size, seq_len, self._num_heads, head_dim],
          message="Unexpected size",
      )

    attn_output = tf.reshape(
        attn_output,
        (batch_size, seq_len, self._num_heads * self._key_dim))  # FIXME

    # compute value for global attention and overwrite to attention output
    # TODO(crickwu): remove the redundant computation
    if self.global_attention_size > 0:
      attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(  # pylint: disable=unused-variable
          attn_output=attn_output,
          hidden_states=hidden_states,
          max_num_global_attn_indices=max_num_global_attn_indices,
          layer_head_mask=layer_head_mask,
          is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
          is_index_global_attn_nonzero=is_index_global_attn_nonzero,
          is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
          is_index_masked=is_index_masked,
          training=training,
      )
    else:
      global_attn_probs = tf.zeros(
          (batch_size, self._num_heads, max_num_global_attn_indices, seq_len))

    # make sure that local attention probabilities are set to 0 for indices of
    # global attn
    if self.global_attention_size > 0:
      masked_global_attn_index = tf.tile(
          is_index_global_attn[:, :, None, None],
          (1, 1, self._num_heads, self._one_sided_attn_window_size * 2 +
           max_num_global_attn_indices + 1),
      )
    else:
      masked_global_attn_index = tf.tile(
          is_index_global_attn[:, :, None, None],
          (1, 1, self._num_heads, self._one_sided_attn_window_size * 2 + 1),
      )

    attn_probs = tf.where(
        masked_global_attn_index,
        tf.zeros(
            get_shape_list(masked_global_attn_index), dtype=attn_probs.dtype),
        attn_probs,
    )

    # we can return extra information here
    # (attn_output, attn_probs, global_attn_probs)

    return attn_output

  def get_config(self):
    config = {
        "layer_id": self._layer_id,
        "attention_window": self._one_sided_attn_window_size,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
    """Matrix multiplication of query and key tensors.

    This multiplication uses a sliding window attention pattern.

    This implementation splits the input into overlapping chunks of size
    2w (e.g. 512 for pretrained Longformer) with an overlap of size
    window_overlap.
    Args:
      query: query tensor.
      key: key tensor.
      window_overlap: int.
    Returns:
      diagonal_attention_scores: tensor.
    """
    batch_size, seq_len, num_heads, head_dim = get_shape_list(query)

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          seq_len % (window_overlap * 2),
          0,
          message=f"Sequence length should be multiple of {window_overlap * 2}. "
          f"Given {seq_len}",
      )
      tf.debugging.assert_equal(
          get_shape_list(query),
          get_shape_list(key),
          message=f"Shape of query and key should be equal, but got query: "
          f"{get_shape_list(query)} and key: {get_shape_list(key)}",
      )

    chunks_count = seq_len // window_overlap - 1

    # group batch_size and num_heads dimensions into one,
    # then chunk seq_len into chunks of size window_overlap * 2
    query = tf.reshape(
        tf.transpose(query, (0, 2, 1, 3)),
        (batch_size * num_heads, seq_len, head_dim),
    )
    key = tf.reshape(
        tf.transpose(key, (0, 2, 1, 3)),
        (batch_size * num_heads, seq_len, head_dim))
    chunked_query = self._chunk(query, window_overlap)
    chunked_key = self._chunk(key, window_overlap)

    # matrix multiplication
    # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
    # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
    # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
    chunked_query = tf.cast(chunked_query, dtype=chunked_key.dtype)
    chunked_attention_scores = tf.einsum("bcxd,bcyd->bcxy", chunked_query,
                                         chunked_key)  # multiply

    # convert diagonals into columns
    paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 1], [0, 0]])
    diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
        chunked_attention_scores, paddings)

    # allocate space for the overall attention matrix where the chunks are
    # combined. The last dimension
    # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns
    # are the window_overlap lower triangles (attention from a word to
    # window_overlap previous words). The following column is attention score
    # from each word to itself, then
    # followed by window_overlap columns for the upper triangle.

    # copy parts from diagonal_chunked_attention_scores into the combined matrix
    # of attentions - copying the main diagonal and the upper triangle
    # TODO(crickwu): This code is most likely not very efficient and should be
    # improved.
    diagonal_attn_scores_up_triang = tf.concat(
        [
            diagonal_chunked_attention_scores[:, :, :window_overlap, :
                                              window_overlap + 1],
            diagonal_chunked_attention_scores[:, -1:,
                                              window_overlap:, :window_overlap +
                                              1],
        ],
        axis=1,
    )

    # - copying the lower triangle
    diagonal_attn_scores_low_triang = tf.concat(
        [
            tf.zeros(
                (batch_size * num_heads, 1, window_overlap, window_overlap),
                dtype=diagonal_chunked_attention_scores.dtype,
            ),
            diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1,
                                              window_overlap + 1:],
        ],
        axis=1,
    )
    diagonal_attn_scores_first_chunk = tf.concat(
        [
            tf.roll(
                diagonal_chunked_attention_scores,
                shift=[1, window_overlap],
                axis=[2, 3],
            )[:, :, :window_overlap, :window_overlap],
            tf.zeros(
                (batch_size * num_heads, 1, window_overlap, window_overlap),
                dtype=diagonal_chunked_attention_scores.dtype,
            ),
        ],
        axis=1,
    )
    first_chunk_mask = (
        tf.tile(
            tf.range(chunks_count + 1)[None, :, None, None],
            (batch_size * num_heads, 1, window_overlap, window_overlap),
        ) < 1)

    diagonal_attn_scores_low_triang = tf.where(
        first_chunk_mask,
        diagonal_attn_scores_first_chunk,
        diagonal_attn_scores_low_triang,
    )

    # merging upper and lower triangle
    diagonal_attention_scores = tf.concat(
        [diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang],
        axis=-1)

    # separate batch_size and num_heads dimensions again
    diagonal_attention_scores = tf.transpose(
        tf.reshape(
            diagonal_attention_scores,
            (batch_size, num_heads, seq_len, 2 * window_overlap + 1),
        ),
        (0, 2, 1, 3),
    )

    diagonal_attention_scores = self._mask_invalid_locations(
        diagonal_attention_scores, window_overlap)

    return diagonal_attention_scores

  @staticmethod
  def _mask_invalid_locations(input_tensor, window_overlap):
    # create correct upper triangle bool mask
    mask_2d_upper = tf.reverse(
        tf.linalg.band_part(
            tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
        axis=[0],
    )

    # pad to full matrix
    padding = tf.convert_to_tensor(
        [[0, get_shape_list(input_tensor)[1] - window_overlap],
         [0, get_shape_list(input_tensor)[3] - window_overlap - 1]])

    # create lower mask
    mask_2d = tf.pad(mask_2d_upper, padding)

    # combine with upper mask
    mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

    # broadcast to full matrix
    mask_4d = tf.tile(mask_2d[None, :, None, :],
                      (get_shape_list(input_tensor)[0], 1, 1, 1))

    # inf tensor used for masking
    inf_tensor = -float("inf") * tf.ones_like(input_tensor)

    # mask
    input_tensor = tf.where(
        tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

    return input_tensor

  def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value,
                                              window_overlap):
    """Same as _sliding_chunks_query_key_matmul but for attn_probs and value."""

    batch_size, seq_len, num_heads, head_dim = get_shape_list(value)

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          seq_len % (window_overlap * 2),
          0,
          message="Seq_len has to be multiple of 2 * window_overlap",
      )
      tf.debugging.assert_equal(
          get_shape_list(attn_probs)[:3],
          get_shape_list(value)[:3],
          message="value and attn_probs must have same dims (except head_dim)",
      )
      tf.debugging.assert_equal(
          get_shape_list(attn_probs)[3],
          2 * window_overlap + 1,
          message="attn_probs last dim has to be 2 * window_overlap + 1",
      )

    chunks_count = seq_len // window_overlap - 1

    # group batch_size and num_heads dimensions into one, then chunk seq_len
    # into chunks of size 2 window overlap
    chunked_attn_probs = tf.reshape(
        tf.transpose(attn_probs, (0, 2, 1, 3)),
        (
            batch_size * num_heads,
            seq_len // window_overlap,
            window_overlap,
            2 * window_overlap + 1,
        ),
    )

    # group batch_size and num_heads dimensions into one
    value = tf.reshape(
        tf.transpose(value, (0, 2, 1, 3)),
        (batch_size * num_heads, seq_len, head_dim),
    )

    # pad seq_len with w at the beginning of the sequence and another window
    # overlap at the end
    paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap],
                                     [0, 0]])
    padded_value = tf.pad(value, paddings, constant_values=-1)

    # chunk padded_value into chunks of size 3 window overlap and an overlap of
    # size window overlap
    frame_size = 3 * window_overlap * head_dim
    frame_hop_size = (get_shape_list(padded_value)[1] * head_dim -
                      frame_size) // chunks_count
    chunked_value = tf.signal.frame(
        tf.reshape(padded_value, (batch_size * num_heads, -1)),
        frame_size,
        frame_hop_size,
    )
    chunked_value = tf.reshape(
        chunked_value,
        (batch_size * num_heads, chunks_count + 1, 3 * window_overlap,
         head_dim),
    )

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          get_shape_list(chunked_value),
          [
              batch_size * num_heads, chunks_count + 1, 3 * window_overlap,
              head_dim
          ],
          message="Chunked value has the wrong shape",
      )

    chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
    context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
    context = tf.transpose(
        tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
        (0, 2, 1, 3),
    )

    return context

  @staticmethod
  def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
    """Pads rows and then flips rows and columns."""
    hidden_states_padded = tf.pad(
        hidden_states_padded, paddings
    )  # padding value is not important because it will be overwritten
    batch_size, chunk_size, seq_length, hidden_dim = get_shape_list(
        hidden_states_padded)
    hidden_states_padded = tf.reshape(
        hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

    return hidden_states_padded

  @staticmethod
  def _pad_and_diagonalize(chunked_hidden_states):
    """Shifts every row 1 step right, converting columns into diagonals.

    Example::

          chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                   -1.8348,  0.7672,  0.2986,  0.0285,
                                   -0.7584,  0.4206, -0.0405,  0.1599,
                                   2.0514, -1.1600,  0.5372,  0.2629 ]
          window_overlap = num_rows = 4
         (pad & diagonalize) =>
         [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
           0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
           0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
           0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
    Args:
      chunked_hidden_states: tensor.
    Returns:
      padded_hidden_stategs: tensor.
    """
    total_num_heads, num_chunks, window_overlap, hidden_dim = get_shape_list(
        chunked_hidden_states)
    paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0],
                                     [0, window_overlap + 1]])

    chunked_hidden_states = tf.pad(chunked_hidden_states, paddings)

    chunked_hidden_states = tf.reshape(chunked_hidden_states,
                                       (total_num_heads, num_chunks, -1))
    chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
    chunked_hidden_states = tf.reshape(
        chunked_hidden_states,
        (total_num_heads, num_chunks, window_overlap,
         window_overlap + hidden_dim),
    )
    chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

    return chunked_hidden_states

  @staticmethod
  def _chunk(hidden_states, window_overlap):
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w."""
    batch_size, seq_length, hidden_dim = get_shape_list(hidden_states)
    num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

    # define frame size and frame stride (similar to convolution)
    frame_hop_size = window_overlap * hidden_dim
    frame_size = 2 * frame_hop_size
    hidden_states = tf.reshape(hidden_states,
                               (batch_size, seq_length * hidden_dim))

    # chunk with overlap
    chunked_hidden_states = tf.signal.frame(hidden_states, frame_size,
                                            frame_hop_size)

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          get_shape_list(chunked_hidden_states),
          [batch_size, num_output_chunks, frame_size],
          message=f"Make sure chunking is correctly applied. `Chunked hidden "
          f"states should have output dimension"
          f" {[batch_size, frame_size, num_output_chunks]}, but got "
          f"{get_shape_list(chunked_hidden_states)}.",
      )

    chunked_hidden_states = tf.reshape(
        chunked_hidden_states,
        (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
    )

    return chunked_hidden_states

  @staticmethod
  def _get_global_attn_indices(is_index_global_attn, global_attention_size):
    """Computes global attn indices required throughout forward pass."""
    # All global attention size are fixed through global_attention_size

    batch_size, _ = get_shape_list(is_index_global_attn)

    max_num_global_attn_indices = global_attention_size

    row_indices = tf.range(batch_size)
    row_indices = tf.repeat(
        tf.expand_dims(row_indices, axis=0),
        repeats=[global_attention_size],
        axis=0)
    row_indices = tf.reshape(row_indices,
                             (batch_size * global_attention_size, 1))

    col_indices = tf.range(global_attention_size)
    col_indices = tf.repeat(
        tf.expand_dims(col_indices, axis=1), repeats=[batch_size], axis=0)

    is_index_global_attn_nonzero = tf.concat((row_indices, col_indices), axis=1)

    # this is actually same as `is_index_global_attn_nonzero`,
    # since we assume all global attention are the same size
    is_local_index_global_attn_nonzero = tf.concat((row_indices, col_indices),
                                                   axis=1)

    # empty tensor
    is_local_index_no_global_attn_nonzero = tf.reshape(
        tf.expand_dims(tf.range(0), axis=1), (0, 2))
    return (
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    )

  def _concat_with_global_key_attn_probs(
      self,
      attn_scores,
      key_vectors,
      query_vectors,
      max_num_global_attn_indices,
      is_index_global_attn_nonzero,
      is_local_index_global_attn_nonzero,
      is_local_index_no_global_attn_nonzero,
  ):
    batch_size = get_shape_list(key_vectors)[0]

    # select global key vectors
    global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

    # create only global key vectors
    key_vectors_only_global = tf.scatter_nd(
        is_local_index_global_attn_nonzero,
        global_key_vectors,
        shape=(
            batch_size,
            max_num_global_attn_indices,
            self._num_heads,
            self._key_dim,
        ),
    )

    # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
    attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors,
                                           key_vectors_only_global)

    # (batch_size, max_num_global_attn_indices, seq_len, num_heads)
    attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key,
                                                    (0, 3, 1, 2))
    mask_shape = (
        get_shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            get_shape_list(attn_probs_from_global_key_trans)[-2:])
    mask = tf.ones(mask_shape) * -10000.0
    mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

    # scatter mask
    attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
        attn_probs_from_global_key_trans,
        is_local_index_no_global_attn_nonzero,
        mask,
    )

    # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
    attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans,
                                              (0, 2, 3, 1))

    # concat to attn_probs
    # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
    attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)
    return attn_scores

  def _compute_attn_output_with_global_indices(
      self,
      value_vectors,
      attn_probs,
      max_num_global_attn_indices,
      is_index_global_attn_nonzero,
      is_local_index_global_attn_nonzero,
  ):
    batch_size = get_shape_list(attn_probs)[0]

    # cut local attn probs to global only
    attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

    # select global value vectors
    global_value_vectors = tf.gather_nd(value_vectors,
                                        is_index_global_attn_nonzero)

    # create only global value vectors
    value_vectors_only_global = tf.scatter_nd(
        is_local_index_global_attn_nonzero,
        global_value_vectors,
        shape=(
            batch_size,
            max_num_global_attn_indices,
            self._num_heads,
            self._key_dim,
        ),
    )

    # compute attn output only global
    attn_output_only_global = tf.einsum("blhs,bshd->blhd",
                                        attn_probs_only_global,
                                        value_vectors_only_global)
    # reshape attn probs
    attn_probs_without_global = attn_probs[:, :, :,
                                           max_num_global_attn_indices:]

    # compute attn output with global
    attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
        attn_probs_without_global, value_vectors,
        self._one_sided_attn_window_size)

    return attn_output_only_global + attn_output_without_global

  def _compute_global_attn_output_from_hidden(
      self,
      attn_output,
      hidden_states,
      max_num_global_attn_indices,
      layer_head_mask,
      is_local_index_global_attn_nonzero,
      is_index_global_attn_nonzero,
      is_local_index_no_global_attn_nonzero,
      is_index_masked,
      training,
  ):
    batch_size, seq_len = get_shape_list(hidden_states)[:2]

    # prepare global hidden states
    global_attn_hidden_states = tf.gather_nd(hidden_states,
                                             is_index_global_attn_nonzero)
    global_attn_hidden_states = tf.scatter_nd(
        is_local_index_global_attn_nonzero,
        global_attn_hidden_states,
        shape=(batch_size, max_num_global_attn_indices,
               self._num_heads * self._key_dim),
    )

    # global key, query, value
    global_query_vectors_only_global = self._global_query_dense(
        global_attn_hidden_states)
    global_key_vectors = self._global_key_dense(hidden_states)
    global_value_vectors = self._global_value_dense(hidden_states)

    # normalize
    global_query_vectors_only_global /= tf.math.sqrt(
        tf.cast(self._key_dim, dtype=global_query_vectors_only_global.dtype))
    global_query_vectors_only_global = self.reshape_and_transpose(
        global_query_vectors_only_global, batch_size)
    global_key_vectors = self.reshape_and_transpose(global_key_vectors,
                                                    batch_size)
    global_value_vectors = self.reshape_and_transpose(global_value_vectors,
                                                      batch_size)

    # compute attn scores
    global_attn_scores = tf.matmul(
        global_query_vectors_only_global, global_key_vectors, transpose_b=True)

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          get_shape_list(global_attn_scores),
          [batch_size * self._num_heads, max_num_global_attn_indices, seq_len],
          message=f"global_attn_scores have the wrong size. Size should be"
          f"{(batch_size * self._num_heads, max_num_global_attn_indices, seq_len)}, "
          f"but is {get_shape_list(global_attn_scores)}.",
      )

    global_attn_scores = tf.reshape(
        global_attn_scores,
        (batch_size, self._num_heads, max_num_global_attn_indices, seq_len),
    )
    global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
    mask_shape = (get_shape_list(is_local_index_no_global_attn_nonzero)[0],
                 ) + tuple(get_shape_list(global_attn_scores_trans)[-2:])
    global_attn_mask = tf.ones(mask_shape) * -10000.0
    global_attn_mask = tf.cast(
        global_attn_mask, dtype=global_attn_scores_trans.dtype)

    # scatter mask
    global_attn_scores_trans = tf.tensor_scatter_nd_update(
        global_attn_scores_trans,
        is_local_index_no_global_attn_nonzero,
        global_attn_mask,
    )
    global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))

    # mask global attn scores
    attn_mask = tf.tile(is_index_masked[:, None, None, :],
                        (1, get_shape_list(global_attn_scores)[1], 1, 1))
    global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)
    global_attn_scores = tf.reshape(
        global_attn_scores,
        (batch_size * self._num_heads, max_num_global_attn_indices, seq_len),
    )

    # compute global attn probs
    global_attn_probs_float = tf.nn.softmax(global_attn_scores, axis=-1)

    # apply layer head masking
    if layer_head_mask is not None:
      if tf.executing_eagerly():
        tf.debugging.assert_equal(
            get_shape_list(layer_head_mask),
            [self._num_heads],
            message=f"Head mask for a single layer should be of size "
            f"{(self._num_heads)}, but is {get_shape_list(layer_head_mask)}",
        )
      global_attn_probs_float = tf.reshape(
          layer_head_mask,
          (1, -1, 1, 1)) * tf.reshape(global_attn_probs_float,
                                      (batch_size, self._num_heads,
                                       max_num_global_attn_indices, seq_len))
      global_attn_probs_float = tf.reshape(
          global_attn_probs_float,
          (batch_size * self._num_heads, max_num_global_attn_indices, seq_len))

    # dropout
    global_attn_probs = self._global_dropout_layer(
        global_attn_probs_float, training=training)

    # global attn output
    global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)

    if tf.executing_eagerly():
      tf.debugging.assert_equal(
          get_shape_list(global_attn_output),
          [
              batch_size * self._num_heads, max_num_global_attn_indices,
              self._key_dim
          ],
          message=f"global_attn_output tensor has the wrong size. Size should be "
          f"{(batch_size * self._num_heads, max_num_global_attn_indices, self._key_dim)}, "
          f"but is {get_shape_list(global_attn_output)}.",
      )

    global_attn_output = tf.reshape(
        global_attn_output,
        (batch_size, self._num_heads, max_num_global_attn_indices,
         self._key_dim),
    )

    # get only non zero global attn output
    nonzero_global_attn_output = tf.gather_nd(
        tf.transpose(global_attn_output, (0, 2, 1, 3)),
        is_local_index_global_attn_nonzero,
    )
    nonzero_global_attn_output = tf.reshape(
        nonzero_global_attn_output,
        (get_shape_list(is_local_index_global_attn_nonzero)[0], -1),
    )

    # overwrite values with global attention
    attn_output = tf.tensor_scatter_nd_update(attn_output,
                                              is_index_global_attn_nonzero,
                                              nonzero_global_attn_output)

    global_attn_probs = tf.reshape(
        global_attn_probs,
        (batch_size, self._num_heads, max_num_global_attn_indices, seq_len))

    attn_output = self._output_dense(attn_output)

    return attn_output, global_attn_probs

  def reshape_and_transpose(self, vector, batch_size):
    return tf.reshape(
        tf.transpose(
            tf.reshape(vector,
                       (batch_size, -1, self._num_heads, self._key_dim)),
            (0, 2, 1, 3),
        ),
        (batch_size * self._num_heads, -1, self._key_dim),
    )
