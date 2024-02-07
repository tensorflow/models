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

"""Multi-channel Attention."""
# pylint: disable=g-classes-have-attributes

import math

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.nlp.modeling.layers import masked_softmax


class VotingAttention(tf_keras.layers.Layer):
  """Voting Attention layer.

  Args:
    num_heads: The number of attention heads.
    head_size: Per-head hidden size.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.
  """

  def __init__(self,
               num_heads,
               head_size,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(**kwargs)
    self._num_heads = num_heads
    self._head_size = head_size
    self._kernel_initializer = tf_keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf_keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf_keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf_keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf_keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf_keras.constraints.get(bias_constraint)

  def build(self, unused_input_shapes):
    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    self._query_dense = tf_keras.layers.EinsumDense(
        "BAE,ENH->BANH",
        output_shape=(None, self._num_heads, self._head_size),
        bias_axes="NH",
        name="query",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)
    self._key_dense = tf_keras.layers.EinsumDense(
        "BAE,ENH->BANH",
        output_shape=(None, self._num_heads, self._head_size),
        bias_axes="NH",
        name="key",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)
    super().build(unused_input_shapes)

  def call(self, encoder_outputs, doc_attention_mask):
    num_docs = tf_utils.get_shape_list(encoder_outputs, expected_rank=[4])[1]
    cls_embeddings = encoder_outputs[:, :, 0, :]
    key = self._key_dense(cls_embeddings)
    query = self._query_dense(cls_embeddings)
    doc_attention_mask = tf.cast(doc_attention_mask, tf.float32)

    key = tf.einsum("BANH,BA->BANH", key, doc_attention_mask)
    query = tf.einsum("BANH,BA->BANH", query, doc_attention_mask)
    attention_matrix = tf.einsum("BXNH,BYNH->BNXY", query, key)
    mask = tf.ones([num_docs, num_docs])
    mask = tf.linalg.set_diag(mask, tf.zeros(num_docs))
    attention_matrix = tf.einsum("BNXY,XY->BNXY", attention_matrix, mask)
    doc_attention_probs = tf.einsum("BNAY->BNA", attention_matrix)
    doc_attention_probs = tf.einsum("BNA->BA", doc_attention_probs)
    infadder = (1.0 - doc_attention_mask) * -100000.0
    return tf.nn.softmax(doc_attention_probs + infadder)


class MultiChannelAttention(tf_keras.layers.MultiHeadAttention):
  """Multi-channel Attention layer.

  Introduced in, [Generating Representative Headlines for News Stories
  ](https://arxiv.org/abs/2001.09386). Expects multiple cross-attention
  target sequences.

  Call args:
    query: Query `Tensor` of shape `[B, T, dim]`.
    value: Value `Tensor` of shape `[B, A, S, dim]`, where A denotes the
    context_attention_weights: Context weights of shape `[B, N, T, A]`, where N
      is the number of attention heads. Combines multi-channel sources
      context tensors according to the distribution among channels.
    key: Optional key `Tensor` of shape `[B, A, S, dim]`. If not given, will use
      `value` for both `key` and `value`, which is the most common case.
    attention_mask: A boolean mask of shape `[B, T, S]`, that prevents attention
      to certain positions.
  """

  def _build_attention(self, rank):
    super()._build_attention(rank)  # pytype: disable=attribute-error  # typed-keras
    self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[2])

  def call(self,
           query,
           value,
           key=None,
           context_attention_weights=None,
           attention_mask=None):
    if not self._built_from_signature:
      self._build_from_signature(query, value, key=key)
    if key is None:
      key = value

    # Scalar dimensions referenced here:
    #   B = batch size (number of stories)
    #   A = num_docs (number of docs)
    #   F = target sequence length
    #   T = source sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self._query_dense(query)

    # `key_tensor` = [B, A, T, N, H]
    key_tensor = self._key_dense(key)

    # `value_tensor` = [B, A, T, N, H]
    value_tensor = self._value_dense(value)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BATNH,BFNH->BANFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_dim)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, A, N, F, T]
    attention_probs = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout_layer(attention_probs)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.einsum("BANFT,BATNH->BAFNH", attention_probs,
                              value_tensor)
    attention_output = tf.einsum("BNFA,BAFNH->BFNH", context_attention_weights,
                                 context_layer)
    attention_output = self._output_dense(attention_output)
    return attention_output
