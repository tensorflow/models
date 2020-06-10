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
"""Talking Head Attention layer."""
# pylint: disable=g-classes-have-attributes
import math
import gin
import tensorflow as tf

from official.nlp.modeling.layers import dense_einsum
from official.nlp.modeling.layers import masked_softmax


@tf.keras.utils.register_keras_serializable(package="Text")
@gin.configurable
class TalkingHeadsAttention(tf.keras.layers.Layer):
  """Implements Talking-Heads Attention.

  https://arxiv.org/abs/2003.02436

  Arguments:
    num_heads: Number of attention heads.
    key_size: Size of each attention head.
    dropout: Dropout probability.
    output_shape: The expected shape of an output tensor, besides the batch and
      sequence dims. If not specified, projects back to the key feature dim.
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
               key_size,
               dropout=0.0,
               output_shape=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(TalkingHeadsAttention, self).__init__(**kwargs)
    self._num_heads = num_heads
    self._key_size = key_size
    self._dropout = dropout
    self._output_shape = output_shape
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    self._query_dense = dense_einsum.DenseEinsum(
        output_shape=(self._num_heads, self._key_size),
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="query")

    self._key_dense = dense_einsum.DenseEinsum(
        output_shape=(self._num_heads, self._key_size),
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="key")

    self._value_dense = dense_einsum.DenseEinsum(
        output_shape=(self._num_heads, self._key_size),
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="value")

    self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])

    self._dropout = tf.keras.layers.Dropout(rate=self._dropout)

  def build(self, input_shape):
    if self._output_shape:
      output_shape = self._output_shape
    else:
      input_shape = tf.TensorShape(input_shape[0])
      output_shape = input_shape[-1]
    self._output_dense = dense_einsum.DenseEinsum(
        output_shape=output_shape,
        num_summed_dimensions=2,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        name="attention_output")
    self._pre_softmax_weight = self.add_weight(
        "pre_softmax_weight",
        shape=(self._num_heads, self._num_heads),
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        constraint=self._kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self._post_softmax_weight = self.add_weight(
        "post_softmax_weight",
        shape=(self._num_heads, self._num_heads),
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        constraint=self._kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    super(TalkingHeadsAttention, self).build(input_shape)

  def get_config(self):
    config = {
        "num_heads":
            self._num_heads,
        "key_size":
            self._key_size,
        "dropout":
            self._dropout,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint)
    }
    base_config = super(TalkingHeadsAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, attention_mask=None):
    from_tensor = inputs[0]
    to_tensor = inputs[1]

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = L = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self._query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self._key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self._value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self._key_size)))

    # Apply talking heads before softmax.
    attention_scores = tf.einsum("BNFT,NL->BLFT", attention_scores,
                                 self._pre_softmax_weight)

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = self._masked_softmax(attention_scores, attention_mask)

    # Apply talking heads after softmax.
    attention_probs = tf.einsum("BNFT,NL->BLFT", attention_probs,
                                self._post_softmax_weight)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self._dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    attention_output = tf.einsum("BNFT,BTNH->BFNH", attention_probs,
                                 value_tensor)
    attention_output = self._output_dense(attention_output)
    return attention_output
