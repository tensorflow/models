# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Keras-based XLNet Model."""

from absl import logging

import tensorflow as tf

from official.nlp.modeling import layers
from official.nlp.modeling.layers import transformer_xl

_SEG_ID_CLS = 2


def _create_causal_attention_mask(
    seq_length,
    memory_length,
    dtype=tf.float32,
    same_length=False):
  """Creates a causal attention mask with a single-sided context.

  When applying the attention mask in `MultiHeadRelativeAttention`, the
  attention scores are of shape `[(batch dimensions), S, S + M]`, where:
  - S = sequence length.
  - M = memory length.

  In a simple case where S = 2, M = 1, here is a simple illustration of the
  `attention_scores` matrix, where `a` represents an attention function:

   token_0   [[a(token_0, mem_0)    a(token_0, token_0)   a(token_0, token_1)],
   token_1    [a(token_1, mem_0)    a(token_1, token_0)   a(token_1, token_1)]]
                      mem_0                token_0               token_1

  For uni-directional attention, we want to mask out values in the attention
  scores that represent a(token_i, token_j) where j > i. We can achieve this by
  concatenating 0s (representing memory positions) with a strictly upper
  triangular matrix of 1s.

  We then flip the matrix values in order to match the representation where
  real values are 1s.

  Arguments:
    seq_length: int, The length of each sequence.
    memory_length: int, The length of memory blocks.
    dtype: dtype of the mask.
    same_length: bool, whether to use the same attention length for each token.

  Returns:
    A unidirectional attention mask of shape
    `[seq_length, seq_length + memory_length]`. E.g.:

    [[1. 1. 1. 0. 0. 0.]
     [1. 1. 1. 1. 0. 0.]
     [1. 1. 1. 1. 1. 0.]
     [1. 1. 1. 1. 1. 1.]]
  """
  ones_matrix = tf.ones([seq_length, seq_length], dtype=dtype)
  upper_triangular = tf.linalg.band_part(ones_matrix, 0, -1)
  diagonal = tf.linalg.band_part(ones_matrix, 0, 0)

  padding = tf.zeros([seq_length, memory_length], dtype=dtype)
  causal_attention_mask = tf.concat(
      [padding, upper_triangular - diagonal], 1)
  if same_length:
    lower_triangular = tf.linalg.band_part(ones_matrix, -1, 0)
    strictly_lower_triangular = lower_triangular - diagonal
    causal_attention_mask = tf.concat(
        [causal_attention_mask[:, :seq_length] + strictly_lower_triangular,
         causal_attention_mask[:, seq_length:]], 1)

  return 1 - causal_attention_mask


def _combine_masks(mask1, mask2, dtype, how="and"):
  """Combines two masks.

  Use "and" if trying to combine two existing masks.
  Use "or" if trying to flip a few positions to "real".

  Args:
    mask1: tf.Tensor, input mask 1
    mask2: tf.Tensor, input mask 2
    dtype: tf.dtype
    how: Which logical operation should run.

  Returns:
    The combined input masks.

  """
  if how == "and":
    operator = tf.math.logical_and
  else:
    operator = tf.math.logical_or
  return tf.cast(operator(
      tf.cast(mask1, tf.bool),
      tf.cast(mask2, tf.bool)), dtype=dtype)


def _compute_attention_mask(
    input_mask,
    permutation_mask,
    attention_type,
    seq_length,
    memory_length,
    batch_size,
    dtype=tf.float32):
  """Combines all input attention masks for XLNet.

  In XLNet modeling, `0` represents tokens that can be attended, and `1`
  represents tokens that cannot be attended.

  For XLNet pre-training and fine tuning, there are a few masks used:
  - Causal attention mask: If the attention type is unidirectional, then all
    tokens after the current position cannot be attended to.
  - Input mask: when generating data, padding is added to a max sequence length
    to make all sequences the same length. This masks out real tokens (`0`) from
    padding tokens (`1`).
  - Permutation mask: during XLNet pretraining, the input sequence is factorized
    into a factorization sequence `z`. During partial prediction, `z` is split
    at a cutting point `c` (an index of the factorization sequence) and
    prediction is only applied to all tokens after `c`. Therefore, tokens at
    factorization positions `i` > `c` can be attended to and tokens at
    factorization positions `i` <= `c` cannot be attended to.

  This function broadcasts and combines all attention masks to produce the
  query attention mask and the content attention mask.

  Args:
    input_mask: Tensor, the input mask related to padding. Input shape:
      `(B, S)`.
    permutation_mask: Tensor, the permutation mask used in partial prediction.
      Input shape: `(B, S, S)`.
    attention_type: str, the attention type. Can be "uni" (directional) or
      "bi" (directional).
    seq_length: int, the length of each sequence.
    memory_length: int the length of memory blocks.
    batch_size: int, the batch size.
    dtype: The dtype of the masks.

  Returns:
    attention_mask, content_attention_mask: The position and context-based
      attention masks and content attention masks, respectively.

  """
  attention_mask = None
  # `1` values mean do not attend to this position.
  if attention_type == "uni":
    causal_attention_mask = _create_causal_attention_mask(
        seq_length=seq_length,
        memory_length=memory_length,
        dtype=dtype)
    causal_attention_mask = causal_attention_mask[None, None, :, :]
    # `causal_attention_mask`: [1, 1, S, S + M]

  # input_mask: [B, S]
  # permutation_mask: [B, S, S]
  if input_mask is not None and permutation_mask is not None:
    data_mask = _combine_masks(input_mask[:, None, :], permutation_mask, dtype)
  elif input_mask is not None and permutation_mask is None:
    data_mask = input_mask[:, None, :]
  elif input_mask is None and permutation_mask is not None:
    data_mask = permutation_mask
  else:
    data_mask = None

  # data_mask: [B, S, S] or [B, 1, S]

  if data_mask is not None:
    # All positions within state can be attended to.
    state_mask = tf.ones([batch_size, tf.shape(data_mask)[1], memory_length],
                         dtype=dtype)
    # state_mask: [B, 1, M] or [B, S, M]
    data_mask = tf.concat([state_mask, data_mask], 2)
    # data_mask: [B, 1, S + M] or [B, S, S + M]

    if attention_type == "uni":
      attention_mask = _combine_masks(causal_attention_mask,
                                      data_mask[:, None, :, :],
                                      dtype=dtype)
    else:
      attention_mask = data_mask[:, None, :, :]

  if attention_mask is not None:
    # Construct the content attention mask.
    # This ensures that the mask allows the model to attend to positions in
    # content positions (e.g. the content diagonal).
    non_target_mask = tf.concat(
        [tf.zeros([seq_length, memory_length], dtype=dtype),
         tf.eye(seq_length, dtype=dtype)], axis=-1)
    content_attention_mask = _combine_masks(
        attention_mask, non_target_mask, how="or", dtype=dtype)
  else:
    content_attention_mask = None

  return attention_mask, content_attention_mask


def _compute_segment_matrix(
    segment_ids,
    memory_length,
    batch_size,
    use_cls_mask):
  """Computes the segment embedding matrix.

  XLNet introduced segment-based attention for attention calculations. This
  extends the idea of relative encodings in Transformer XL by considering
  whether or not two positions are within the same segment, rather than
  which segments they come from.

  This function generates a segment matrix by broadcasting provided segment IDs
  in two different dimensions and checking where values are equal. This output
  matrix shows `True` whenever two tokens are NOT in the same segment and
  `False` whenever they are.

  Args:
    segment_ids: A Tensor of size `[B, S]` that represents which segment
      each token belongs to.
    memory_length: int, the length of memory blocks.
    batch_size: int, the batch size.
    use_cls_mask: bool, whether or not to introduce cls mask in
      input sequences.

  Returns:
    A boolean Tensor of size `[B, S, S + M]`, where `True` means that two
    tokens are NOT in the same segment, and `False` means they are in the same
    segment.

  """
  if segment_ids is None:
    return None

  memory_padding = tf.zeros([batch_size, memory_length], dtype=tf.int32)
  padded_segment_ids = tf.concat([memory_padding, segment_ids], 1)
  # segment_ids: [B, S]
  # padded_segment_ids: [B, S + M]

  if use_cls_mask:
    # `1` indicates not in the same segment.
    # Target result: [B, S, S + M]

    # segment_ids: [B, S]
    # padded_segment_ids: [B, S + M]
    broadcasted_segment_class_indices = (
        tf.equal(segment_ids,
                 tf.constant([_SEG_ID_CLS]))[:, :, None])

    broadcasted_padded_class_indices = (
        tf.equal(
            padded_segment_ids,
            tf.constant([_SEG_ID_CLS]))[:, None, :])

    class_index_matrix = tf.logical_or(broadcasted_segment_class_indices,
                                       broadcasted_padded_class_indices)

    segment_matrix = tf.equal(segment_ids[:, :, None],
                              padded_segment_ids[:, None, :])
    segment_matrix = tf.logical_or(class_index_matrix, segment_matrix)
  else:
    # TODO(allencwang) - address this legacy mismatch from `use_cls_mask`.
    segment_matrix = tf.logical_not(
        tf.equal(segment_ids[:, :, None], padded_segment_ids[:, None, :]))
  return segment_matrix


def _compute_positional_encoding(
    attention_type,
    position_encoding_layer,
    hidden_size,
    batch_size,
    total_length,
    seq_length,
    clamp_length,
    bi_data,
    dtype=tf.float32):
  """Computes the relative position encoding.

  Args:
    attention_type: str, the attention type. Can be "uni" (directional) or
      "bi" (directional).
    position_encoding_layer: An instance of `RelativePositionEncoding`.
    hidden_size: int, the hidden size.
    batch_size: int, the batch size.
    total_length: int, the sequence length added to the memory length.
    seq_length: int, the length of each sequence.
    clamp_length: int, clamp all relative distances larger than clamp_length. -1
      means no clamping.
    bi_data: bool, whether to use bidirectional input pipeline. Usually set to
      True during pretraining and False during finetuning.
    dtype: the dtype of the encoding.

  Returns:
    A Tensor, representing the position encoding.

  """
  freq_seq = tf.range(0, hidden_size, 2.0)
  if dtype is not None and dtype != tf.float32:
    freq_seq = tf.cast(freq_seq, dtype=dtype)

  if attention_type == "bi":
    beg, end = total_length, -seq_length
  elif attention_type == "uni":
    beg, end = total_length, -1
  else:
    raise ValueError("Unknown `attention_type` {}.".format(attention_type))

  if bi_data:
    forward_position_sequence = tf.range(beg, end, -1.0)
    backward_position_sequence = tf.range(-beg, -end, 1.0)

    if dtype is not None and dtype != tf.float32:
      forward_position_sequence = tf.cast(forward_position_sequence,
                                          dtype=dtype)
      backward_position_sequence = tf.cast(backward_position_sequence,
                                           dtype=dtype)

    if clamp_length > 0:
      forward_position_sequence = tf.clip_by_value(
          forward_position_sequence,
          -clamp_length,
          clamp_length)
      backward_position_sequence = tf.clip_by_value(
          backward_position_sequence,
          -clamp_length,
          clamp_length)

    if batch_size is not None:
      forward_positional_encoding = position_encoding_layer(
          forward_position_sequence, batch_size // 2)
      backward_positional_encoding = position_encoding_layer(
          backward_position_sequence, batch_size // 2)
    else:
      forward_positional_encoding = position_encoding_layer(
          forward_position_sequence, None)
      backward_positional_encoding = position_encoding_layer(
          backward_position_sequence, None)

    relative_position_encoding = tf.concat(
        [forward_positional_encoding, backward_positional_encoding], axis=0)
  else:
    forward_position_sequence = tf.range(beg, end, -1.0)
    if dtype is not None and dtype != tf.float32:
      forward_position_sequence = tf.cast(
          forward_position_sequence, dtype=dtype)
    if clamp_length > 0:
      forward_position_sequence = tf.clip_by_value(
          forward_position_sequence,
          -clamp_length,
          clamp_length)

    relative_position_encoding = position_encoding_layer(
        forward_position_sequence, batch_size)
  return relative_position_encoding


class RelativePositionEncoding(tf.keras.layers.Layer):
  """Creates a relative positional encoding.

  This layer creates a relative positional encoding as described in
  "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
  (https://arxiv.org/abs/1901.02860).

  Rather than an absolute position embedding as in Transformer, this
  formulation represents position as the relative distance between tokens using
  sinusoidal positional embeddings.

  Note: This layer is currently experimental.

  Attributes:
    hidden_size: The dimensionality of the input embeddings.
  """

  def __init__(self, hidden_size, **kwargs):
    super(RelativePositionEncoding, self).__init__(**kwargs)
    self._hidden_size = hidden_size
    self._inv_freq = 1.0 / (10000.0**(
        tf.range(0, self._hidden_size, 2.0) / self._hidden_size))

  def call(self, pos_seq, batch_size=None):
    """Implements call() for the layer.

    Arguments:
      pos_seq: A 1-D `Tensor`
      batch_size: The optionally provided batch size that tiles the relative
        positional encoding.

    Returns:
      The relative positional encoding of shape:
        [batch_size, len(pos_seq), hidden_size] if batch_size is provided, else
        [1, len(pos_seq), hidden_size].
    """
    sinusoid_input = tf.einsum("i,d->id", pos_seq, self._inv_freq)
    relative_position_encoding = tf.concat([tf.sin(sinusoid_input),
                                            tf.cos(sinusoid_input)], -1)
    relative_position_encoding = relative_position_encoding[None, :, :]
    if batch_size is not None:
      relative_position_encoding = tf.tile(relative_position_encoding,
                                           [batch_size, 1, 1])
    return relative_position_encoding


@tf.keras.utils.register_keras_serializable(package="Text")
class XLNetBase(tf.keras.layers.Layer):
  """Base XLNet model.

  Attributes:
    vocab_size: int, the number of tokens in vocabulary.
    num_layers: int, the number of layers.
    hidden_size: int, the hidden size.
    num_attention_heads: int, the number of attention heads.
    head_size: int, the dimension size of each attention head.
    inner_size: int, the hidden size in feed-forward layers.
    dropout_rate: float, dropout rate.
    attention_dropout_rate: float, dropout rate on attention probabilities.
    attention_type: str, "uni" or "bi".
    bi_data: bool, whether to use bidirectional input pipeline. Usually set to
      True during pretraining and False during finetuning.
    initializer: A tf initializer.
    two_stream: bool, whether or not to use `TwoStreamRelativeAttention` used
      in the XLNet pretrainer. If `False`, then it will use
      `MultiHeadRelativeAttention` as in Transformer XL.
    tie_attention_biases: bool, whether or not to tie the biases together.
      Usually set to `True`. Used for backwards compatibility.
    memory_length: int, the number of tokens to cache.
    same_length: bool, whether to use the same attention length for each
      token.
    clamp_length: int, clamp all relative distances larger than clamp_length. -1
      means no clamping.
    reuse_length: int, the number of tokens in the currect batch to be cached
      and reused in the future.
    inner_activation: str, "relu" or "gelu".
    use_cls_mask: bool, whether or not cls mask is included in the
      input sequences.
    embedding_width: The width of the word embeddings. If the embedding width
      is not equal to hidden size, embedding parameters will be factorized
      into two matrices in the shape of ["vocab_size", "embedding_width"] and
      ["embedding_width", "hidden_size"] ("embedding_width" is usually much
      smaller than "hidden_size").
    embedding_layer: The word embedding layer. `None` means we will create a
      new embedding layer. Otherwise, we will reuse the given embedding layer.
      This parameter is originally added for ELECTRA model which needs to tie
      the generator embeddings with the discriminator embeddings.
  """

  def __init__(self,
               vocab_size,
               num_layers,
               hidden_size,
               num_attention_heads,
               head_size,
               inner_size,
               dropout_rate,
               attention_dropout_rate,
               attention_type,
               bi_data,
               initializer,
               two_stream=False,
               tie_attention_biases=True,
               memory_length=None,
               clamp_length=-1,
               reuse_length=None,
               inner_activation="relu",
               use_cls_mask=False,
               embedding_width=None,
               **kwargs):
    super(XLNetBase, self).__init__(**kwargs)

    self._vocab_size = vocab_size
    self._initializer = initializer
    self._attention_type = attention_type
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._num_attention_heads = num_attention_heads
    self._head_size = head_size
    self._inner_size = inner_size
    self._inner_activation = inner_activation
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._tie_attention_biases = tie_attention_biases
    self._two_stream = two_stream

    self._memory_length = memory_length
    self._reuse_length = reuse_length
    self._bi_data = bi_data
    self._clamp_length = clamp_length
    self._use_cls_mask = use_cls_mask

    self._segment_embedding = None
    self._mask_embedding = None
    self._embedding_width = embedding_width

    if embedding_width is None:
      embedding_width = hidden_size

    self._embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=self._vocab_size,
        embedding_width=embedding_width,
        initializer=self._initializer,
        dtype=tf.float32,
        name="word_embedding")
    self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    self.embedding_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    self.position_encoding = RelativePositionEncoding(self._hidden_size)

    self._transformer_xl = transformer_xl.TransformerXL(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
        inner_size=inner_size,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        initializer=initializer,
        two_stream=two_stream,
        tie_attention_biases=tie_attention_biases,
        memory_length=memory_length,
        reuse_length=reuse_length,
        inner_activation=inner_activation,
        name="transformer_xl")

  def get_config(self):
    config = {
        "vocab_size":
            self._vocab_size,
        "num_layers":
            self._num_layers,
        "hidden_size":
            self._hidden_size,
        "num_attention_heads":
            self._num_attention_heads,
        "head_size":
            self._head_size,
        "inner_size":
            self._inner_size,
        "dropout_rate":
            self._dropout_rate,
        "attention_dropout_rate":
            self._attention_dropout_rate,
        "attention_type":
            self._attention_type,
        "bi_data":
            self._bi_data,
        "initializer":
            self._initializer,
        "two_stream":
            self._two_stream,
        "tie_attention_biases":
            self._tie_attention_biases,
        "memory_length":
            self._memory_length,
        "clamp_length":
            self._clamp_length,
        "reuse_length":
            self._reuse_length,
        "inner_activation":
            self._inner_activation,
        "use_cls_mask":
            self._use_cls_mask,
        "embedding_width":
            self._embedding_width,
    }
    base_config = super(XLNetBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_embedding_lookup_table(self):
    """Returns the embedding layer weights."""
    return self._embedding_layer.embeddings

  def __call__(self,
               input_ids,
               segment_ids=None,
               input_mask=None,
               state=None,
               permutation_mask=None,
               target_mapping=None,
               masked_tokens=None,
               **kwargs):
    # Uses dict to feed inputs into call() in order to keep state as a python
    # list.
    inputs = {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "input_mask": input_mask,
        "state": state,
        "permutation_mask": permutation_mask,
        "target_mapping": target_mapping,
        "masked_tokens": masked_tokens
    }
    return super(XLNetBase, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_ids = inputs["input_ids"]
    segment_ids = inputs["segment_ids"]
    input_mask = inputs["input_mask"]
    state = inputs["state"]
    permutation_mask = inputs["permutation_mask"]
    target_mapping = inputs["target_mapping"]
    masked_tokens = inputs["masked_tokens"]

    batch_size = tf.shape(input_ids)[0]
    seq_length = tf.shape(input_ids)[1]
    if state is not None:
      memory_length = tf.shape(state[0])[1]
    else:
      memory_length = 0
    total_length = memory_length + seq_length

    if self._two_stream and masked_tokens is None:
      raise ValueError("`masked_tokens` must be provided in order to "
                       "initialize the query stream in "
                       "`TwoStreamRelativeAttention`.")
    if masked_tokens is not None and not self._two_stream:
      logging.warning("`masked_tokens` is provided but `two_stream` is not "
                      "enabled. Please enable `two_stream` to enable two "
                      "stream attention.")

    query_attention_mask, content_attention_mask = _compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type=self._attention_type,
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)
    relative_position_encoding = _compute_positional_encoding(
        attention_type=self._attention_type,
        position_encoding_layer=self.position_encoding,
        hidden_size=self._hidden_size,
        batch_size=batch_size,
        total_length=total_length,
        seq_length=seq_length,
        clamp_length=self._clamp_length,
        bi_data=self._bi_data,
        dtype=tf.float32)
    relative_position_encoding = self.embedding_dropout(
        relative_position_encoding)

    if segment_ids is None:
      segment_embedding = None
      segment_matrix = None
    else:
      if self._segment_embedding is None:
        self._segment_embedding = self.add_weight(
            "seg_embed",
            shape=[self._num_layers, 2, self._num_attention_heads,
                   self._head_size],
            dtype=tf.float32,
            initializer=self._initializer)

      segment_embedding = self._segment_embedding
      segment_matrix = _compute_segment_matrix(
          segment_ids=segment_ids,
          memory_length=memory_length,
          batch_size=batch_size,
          use_cls_mask=self._use_cls_mask)

    word_embeddings = self._embedding_layer(input_ids)
    content_stream = self._dropout(word_embeddings)

    if self._two_stream:
      if self._mask_embedding is None:
        self._mask_embedding = self.add_weight(
            "mask_emb/mask_emb",
            shape=[1, 1, self._hidden_size],
            dtype=tf.float32)
      if target_mapping is None:
        masked_tokens = masked_tokens[:, :, None]
        masked_token_embedding = (
            masked_tokens * self._mask_embedding +
            (1 - masked_tokens) * word_embeddings)
      else:
        masked_token_embedding = tf.tile(
            self._mask_embedding,
            [batch_size, tf.shape(target_mapping)[1], 1])
      query_stream = self._dropout(masked_token_embedding)
    else:
      query_stream = None

    return self._transformer_xl(
        content_stream=content_stream,
        query_stream=query_stream,
        target_mapping=target_mapping,
        state=state,
        relative_position_encoding=relative_position_encoding,
        segment_matrix=segment_matrix,
        segment_embedding=segment_embedding,
        content_attention_mask=content_attention_mask,
        query_attention_mask=query_attention_mask)
