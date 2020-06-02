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
"""Multi-channel decoder."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math

import tensorflow as tf
from official.modeling import tf_utils
from official.nlp.modeling import layers


class DocAttention(tf.keras.layers.Layer):
  """Documents Attention layer."""

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
    super(DocAttention, self).__init__(**kwargs)
    self._num_heads = num_heads
    self._head_size = head_size
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

  def build(self, unused_input_shapes):
    self._query_dense = layers.DenseEinsum(
        output_shape=(self._num_heads, self._head_size),
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        dtype=self.dtype,
        name="encdocatt_query")
    self._key_dense = layers.DenseEinsum(
        output_shape=(self._num_heads, self._head_size),
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        dtype=self.dtype,
        name="encdocatt_key")
    super(DocAttention, self).build(unused_input_shapes)

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


class MultiChannelAttention(layers.MultiHeadAttention):
  """Multi-channel Attention layer."""

  def build(self, input_shape):
    super(MultiChannelAttention, self).build(input_shape)
    self._masked_softmax = layers.MaskedSoftmax(mask_expansion_axes=[2])

  def call(self, inputs, attention_mask=None):
    from_tensor = inputs[0]
    to_tensor = inputs[1]
    doc_attention_probs = inputs[2]

    # Scalar dimensions referenced here:
    #   B = batch size (number of stories)
    #   A = num_docs (number of docs)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self._query_dense(from_tensor)

    # `key_tensor` = [B, A, T, N, H]
    key_tensor = self._key_dense(to_tensor)

    # `value_tensor` = [B, A, T, N, H]
    value_tensor = self._value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BATNH,BFNH->BANFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, A, N, F, T]
    attention_probs = self._masked_softmax([attention_scores, attention_mask])

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout_layer(attention_probs)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.einsum("BANFT,BATNH->BAFNH", attention_probs,
                              value_tensor)
    attention_output = tf.einsum("BNFA,BAFNH->BFNH", doc_attention_probs,
                                 context_layer)
    attention_output = self._output_dense(attention_output)
    return attention_output
