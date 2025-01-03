# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Layers for Transformer encoder."""
# pylint: disable=arguments-renamed
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import embedding_layers # import seq_flow_lite module
from layers import normalization_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module
from tf_ops import tf_custom_ops_py # import seq_flow_lite module


class SelfAttention(base_layers.BaseLayer):
  """Self attention encoder (not suitable for causal attention)."""

  def __init__(self,
               model_dimension,
               num_heads,
               attention_dropout_rate=0.0,
               **kwargs):
    self.model_dimension = model_dimension
    self.num_heads = num_heads
    self.filters = model_dimension // num_heads
    self.dense_layers = [
        dense_layers.BaseQDenseVarLen(
            units=self.filters, activation=None, **kwargs)
        for i in range(num_heads * 3)
    ]
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    self.attention_dropout_rate = attention_dropout_rate
    self.qconcat = quantization_layers.ConcatQuantization(axis=2, **kwargs)
    super(SelfAttention, self).__init__(**kwargs)

  def call(self, inputs, mask, inverse_normalizer, attn_mask=None):
    batch_size = self.get_batch_dimension(inputs)
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension

    inputs_rank2 = tf.reshape(inputs, [-1, self.model_dimension])
    mask_rank2 = tf.reshape(mask, [-1, 1])
    tensors = [
        layer(inputs_rank2, mask_rank2, inverse_normalizer)
        for layer in self.dense_layers
    ]
    if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
      tensors = [
          tf.reshape(tensor, [batch_size, -1, self.filters])
          for tensor in tensors
      ]
    context = []
    if attn_mask is None:
      attn_mask = tf.matmul(mask, tf.transpose(mask, [0, 2, 1]))

    if (self.attention_dropout_rate > 0.0 and
        self.parameters.mode == base_layers.TRAIN):
      attn_mask *= self.random_drop_to_zero(attn_mask,
                                            self.attention_dropout_rate)
    invalid_mask = (1 - attn_mask) * self.parameters.invalid_logit
    for _ in range(self.num_heads):
      keys = tensors.pop()
      values = tensors.pop()
      queries = tensors.pop()
      # Attention is not scaled dot product, batch normalization compensates
      # for it.
      if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
        queries = tf.transpose(queries, [0, 2, 1])
        attn_logits = self.qactivation(tf.matmul(keys, queries))
        attn_logits_masked = attn_logits * attn_mask + invalid_mask
        attention = tf.nn.softmax(attn_logits_masked)
        attention = self.qrange_sigmoid(attention, tf_only=True)
        context.append(tf.matmul(attention, values))
      else:
        queries = tf.transpose(queries)
        attn_logits_masked = self.qactivation(tf.matmul(keys, queries))
        attention = tf.nn.softmax(attn_logits_masked)
        attention = self.qrange_sigmoid(attention, tf_only=True)
        ctx = tf.matmul(attention, values)
        ctx = tf.reshape(ctx, [1, -1, self.filters])
        context.append(ctx)
    return self.qconcat(context)


class SelfAttentionV2(base_layers.BaseLayer):
  """Self attention encoder (not suitable for causal attention)."""

  def __init__(self,
               model_dimension,
               num_heads,
               attention_dropout_rate=0.0,
               **kwargs):
    self.model_dimension = model_dimension
    self.num_heads = num_heads
    self.filters = model_dimension // num_heads
    self.dense_layers = dense_layers.BaseQDenseVarLen(
        units=model_dimension * 3, activation=None, **kwargs)
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    self.attention_dropout_rate = attention_dropout_rate
    self.qconcat = quantization_layers.ConcatQuantization(axis=1, **kwargs)
    super(SelfAttentionV2, self).__init__(**kwargs)

  def call(self, inputs, mask, inverse_normalizer, attn_mask=None):
    bsz = self.get_batch_dimension(inputs)
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension

    inputs_rank2 = tf.reshape(inputs, [-1, self.model_dimension])
    mask_rank2 = tf.reshape(mask, [-1, 1])
    tensors = self.dense_layers(inputs_rank2, mask_rank2, inverse_normalizer)
    if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
      tensors = tf.reshape(tensors, [bsz, -1, 3, self.num_heads, self.filters])
      tensors = tf.unstack(tensors, axis=2)
    else:
      tensors = tf.split(tensors, self.num_heads * 3, axis=1)
    if attn_mask is None:
      attn_mask = tf.matmul(mask, mask, transpose_b=True)
    if (self.attention_dropout_rate > 0.0 and
        self.parameters.mode == base_layers.TRAIN):
      attn_mask *= self.random_drop_to_zero(attn_mask,
                                            self.attention_dropout_rate)
    attn_mask = tf.expand_dims(attn_mask, axis=1)
    invalid_mask = (1 - attn_mask) * self.parameters.invalid_logit
    if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
      queries = tf.transpose(tensors[0], [0, 2, 1, 3])
      keys = tf.transpose(tensors[1], [0, 2, 1, 3])
      values = tf.transpose(tensors[2], [0, 2, 1, 3])

      attn_logits = self.qactivation(tf.matmul(queries, keys, transpose_b=True))
      attn_logits_masked = attn_logits * attn_mask + invalid_mask
      attention = tf.nn.softmax(attn_logits_masked)
      attention = self.qrange_sigmoid(attention, tf_only=True)
      result = tf.matmul(attention, values)
      result = tf.transpose(result, [0, 2, 1, 3])
      result = tf.reshape(result, [bsz, -1, self.model_dimension])
      return self.qconcat([result])
    else:
      context = []
      for idx in range(self.num_heads):
        queries = tensors[idx]
        keys = tensors[idx + self.num_heads]
        values = tensors[idx + self.num_heads * 2]
        # Attention is not scaled dot product, batch normalization compensates
        # for it.
        attn_logits_masked = self.qactivation(
            tf.matmul(queries, keys, transpose_b=True))
        attention = tf.nn.softmax(attn_logits_masked)
        attention = self.qrange_sigmoid(attention, tf_only=True)
        context.append(tf.matmul(attention, values))
      result = self.qconcat(context)
      return tf.reshape(result, [1, -1, self.model_dimension])


class TransformerEncoder(base_layers.BaseLayer):
  """Transformer Encoder."""

  def __init__(self,
               model_dimension,
               num_heads,
               intermediate_size,
               initializer_stddev=0.02,
               activation_dropout_rate=0.0,
               attention_dropout_rate=0.0,
               **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.model_dimension = model_dimension
    self.parameters.initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initializer_stddev)
    self.self_attn = SelfAttentionV2(
        model_dimension,
        num_heads,
        attention_dropout_rate=attention_dropout_rate,
        parameters=self.parameters)
    self.prx = dense_layers.BaseQDenseVarLen(
        model_dimension, activation=None, parameters=self.parameters)
    self.upprx = dense_layers.BaseQDenseVarLen(
        intermediate_size, parameters=self.parameters)
    self.downprx = dense_layers.BaseQDenseVarLen(
        model_dimension, activation=None, parameters=self.parameters)
    self.activation_dropout_rate = activation_dropout_rate
    self.ln1 = normalization_layers.LayerNormalization(**kwargs)
    self.ln2 = normalization_layers.LayerNormalization(**kwargs)
    self.q1 = quantization_layers.ActivationQuantization(**kwargs)
    self.q2 = quantization_layers.ActivationQuantization(**kwargs)

  def call(self, inputs, mask, inverse_normalizer, attn_mask=None):
    batch_size = self.get_batch_dimension(inputs)
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension
    mask_rank2 = tf.reshape(mask, [-1, 1])
    assert inputs.get_shape().as_list()[-1] == self.model_dimension
    tensor = self.self_attn(inputs, mask, inverse_normalizer, attn_mask)
    inputs = tf.reshape(inputs, [-1, self.model_dimension])
    tensor = tf.reshape(tensor, [-1, self.model_dimension])
    tensor = self.prx(tensor, mask_rank2, inverse_normalizer)
    if (self.parameters.mode == base_layers.TRAIN and
        self.activation_dropout_rate > 0.0):
      tensor = tf.nn.dropout(tensor, rate=self.activation_dropout_rate)
    inputs_plus_selfattn = self.q1(self.ln1(inputs + tensor))

    ffn_up = self.upprx(inputs_plus_selfattn, mask_rank2, inverse_normalizer)
    ffn_down = self.downprx(ffn_up, mask_rank2, inverse_normalizer)
    if (self.parameters.mode == base_layers.TRAIN and
        self.activation_dropout_rate > 0.0):
      ffn_down = tf.nn.dropout(ffn_down, rate=self.activation_dropout_rate)
    inputs_plus_ffn = self.q2(self.ln2(inputs_plus_selfattn + ffn_down))
    return tf.reshape(inputs_plus_ffn, [batch_size, -1, self.model_dimension])


class TransformerEncoderStack(base_layers.BaseLayer):
  """Transformer Encoder."""

  def __init__(self, num_layers, max_time_step, vocabulary_size, embedding_size,
               model_dimension, num_heads, intermediate_size, **kwargs):
    self.max_time_step = max_time_step
    self.vocabulary_size = vocabulary_size
    self.embedding_size = embedding_size
    activation_dropout_rate = kwargs.pop('activation_dropout_rate', 0.0)
    attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.0)
    self.layers = []
    for _ in range(num_layers):
      self.layers.append(
          TransformerEncoder(
              model_dimension=model_dimension,
              num_heads=num_heads,
              intermediate_size=intermediate_size,
              activation_dropout_rate=activation_dropout_rate,
              attention_dropout_rate=attention_dropout_rate,
              **kwargs))
    self.embedding = embedding_layers.EmbeddingLayer(
        shape=[self.vocabulary_size, self.embedding_size], **kwargs)
    self.positional_embedding = embedding_layers.EmbeddingLayer(
        shape=[self.max_time_step, self.embedding_size], **kwargs)
    self.ln = normalization_layers.LayerNormalization(**kwargs)
    self.qact = quantization_layers.ActivationQuantization(**kwargs)
    super(TransformerEncoderStack, self).__init__(**kwargs)

  def call(self, input_indices, sequence_length):
    mask_rank2 = tf.sequence_mask(
        sequence_length, tf.shape(input_indices)[1], dtype=tf.float32)
    mask_rank3 = tf.expand_dims(mask_rank2, axis=2)
    inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask_rank3))
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      sequence_length = tf.reduce_sum(input_indices + 1 - input_indices)
      pos_indices = tf.range(sequence_length, dtype=tf.int32)
      pos_indices = tf.reshape(pos_indices, [1, -1])
    else:
      pos_indices = tf.cumsum(mask_rank2, axis=1, exclusive=True)
      pos_indices = tf.cast(pos_indices, dtype=tf.int32)

    input_values = self.embedding(input_indices)
    pos_values = self.positional_embedding(pos_indices)
    inputs = self.qact(self.ln(input_values + pos_values))
    attn_mask = tf.matmul(mask_rank3, tf.transpose(mask_rank3, [0, 2, 1]))
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      inputs = inputs * mask_rank3
    for layer in self.layers:
      outputs = layer(inputs, mask_rank3, inverse_normalizer, attn_mask)
      inputs = outputs
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      outputs = outputs * mask_rank3
    return outputs


class TransformerEncoderStackWithInputEmbedding(TransformerEncoderStack):
  """Transformer Encoder."""

  def call(self, inputs, sequence_length):
    mask_rank2 = tf.sequence_mask(
        sequence_length, tf.shape(inputs)[1], dtype=tf.float32)
    mask_rank3 = tf.expand_dims(mask_rank2, axis=2)
    inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask_rank3))
    attn_mask = tf.matmul(mask_rank3, tf.transpose(mask_rank3, [0, 2, 1]))
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      inputs = inputs * mask_rank3
    for layer in self.layers:
      outputs = layer(inputs, mask_rank3, inverse_normalizer, attn_mask)
      inputs = outputs
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      outputs = outputs * mask_rank3
    return outputs


class FunnelAttention(base_layers.BaseLayer):
  """Self attention encoder (not suitable for causal attention)."""

  def __init__(self,
               model_dimension,
               num_heads,
               attention_dropout_rate=0.0,
               **kwargs):
    self.model_dimension = model_dimension
    self.num_heads = num_heads
    self.filters = model_dimension // num_heads
    self.q_dense_layer = dense_layers.BaseQDenseVarLen(
        units=model_dimension, activation=None, **kwargs)
    self.kv_dense_layer = dense_layers.BaseQDenseVarLen(
        units=model_dimension * 2, activation=None, **kwargs)
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    self.attention_dropout_rate = attention_dropout_rate
    self.qconcat = quantization_layers.ConcatQuantization(axis=1, **kwargs)
    super(FunnelAttention, self).__init__(**kwargs)

  def call(self, inputs, mask, inverse_normalizer, memory, memory_mask,
           memory_inverse_normalizer, attn_mask):
    bsz = self.get_batch_dimension(inputs)
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension
    self._assert_rank_and_type(memory, 3)
    self._assert_rank_and_type(memory_mask, 3)
    assert memory.get_shape().as_list()[-1] == self.model_dimension

    inputs_rank2 = tf.reshape(inputs, [-1, self.model_dimension])
    mask_rank2 = tf.reshape(mask, [-1, 1])
    q_tensor = self.q_dense_layer(inputs_rank2, mask_rank2, inverse_normalizer)

    memory_rank2 = tf.reshape(memory, [-1, self.model_dimension])
    memory_mask_rank2 = tf.reshape(memory_mask, [-1, 1])
    kv_tensors = self.kv_dense_layer(memory_rank2, memory_mask_rank2,
                                     inverse_normalizer)
    if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
      q_tensor = tf.reshape(q_tensor, [bsz, -1, self.num_heads, self.filters])
      kv_tensors = tf.reshape(kv_tensors,
                              [bsz, -1, 2, self.num_heads, self.filters])
      kv_tensors = tf.unstack(kv_tensors, axis=2)
    else:
      q_tensor = tf.split(q_tensor, self.num_heads, axis=1)
      kv_tensors = tf.split(kv_tensors, self.num_heads * 2, axis=1)

    attn_mask = tf.expand_dims(attn_mask, axis=1)
    invalid_mask = (1 - attn_mask) * self.parameters.invalid_logit
    if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
      queries = tf.transpose(q_tensor, [0, 2, 1, 3])
      keys = tf.transpose(kv_tensors[0], [0, 2, 1, 3])
      values = tf.transpose(kv_tensors[1], [0, 2, 1, 3])

      attn_logits = self.qactivation(tf.matmul(queries, keys, transpose_b=True))
      attn_logits_masked = attn_logits * attn_mask + invalid_mask
      attention = tf.nn.softmax(attn_logits_masked)
      attention = self.qrange_sigmoid(attention, tf_only=True)
      result = tf.matmul(attention, values)
      result = tf.transpose(result, [0, 2, 1, 3])
      result = tf.reshape(result, [bsz, -1, self.model_dimension])
      return self.qconcat([result])
    else:
      context = []
      for idx in range(self.num_heads):
        queries = q_tensor[idx]
        keys = kv_tensors[idx]
        values = kv_tensors[idx + self.num_heads]
        # Attention is not scaled dot product, batch normalization compensates
        # for it.
        attn_logits_masked = self.qactivation(
            tf.matmul(queries, keys, transpose_b=True))
        attention = tf.nn.softmax(attn_logits_masked)
        attention = self.qrange_sigmoid(attention, tf_only=True)
        context.append(tf.matmul(attention, values))
      result = self.qconcat(context)
      return tf.reshape(result, [1, -1, self.model_dimension])


class FunnelTransformerEncoder(base_layers.BaseLayer):
  """Transformer Encoder."""

  def __init__(self,
               model_dimension,
               num_heads,
               intermediate_size,
               initializer_stddev=0.02,
               activation_dropout_rate=0.0,
               attention_dropout_rate=0.0,
               **kwargs):
    super(FunnelTransformerEncoder, self).__init__(**kwargs)
    self.model_dimension = model_dimension
    self.parameters.initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initializer_stddev)
    self.self_attn = FunnelAttention(
        model_dimension,
        num_heads,
        attention_dropout_rate=attention_dropout_rate,
        parameters=self.parameters)
    self.prx = dense_layers.BaseQDenseVarLen(
        model_dimension, activation=None, parameters=self.parameters)
    self.upprx = dense_layers.BaseQDenseVarLen(
        intermediate_size, parameters=self.parameters)
    self.downprx = dense_layers.BaseQDenseVarLen(
        model_dimension, activation=None, parameters=self.parameters)
    self.activation_dropout_rate = activation_dropout_rate
    self.ln1 = normalization_layers.LayerNormalization(**kwargs)
    self.ln2 = normalization_layers.LayerNormalization(**kwargs)
    self.q1 = quantization_layers.ActivationQuantization(**kwargs)
    self.q2 = quantization_layers.ActivationQuantization(**kwargs)

  def call(self, inputs, mask, inverse_normalizer, memory, memory_mask,
           memory_inverse_normalizer, attn_mask):
    batch_size = self.get_batch_dimension(inputs)
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension
    mask_rank2 = tf.reshape(mask, [-1, 1])
    assert inputs.get_shape().as_list()[-1] == self.model_dimension
    tensor = self.self_attn(inputs, mask, inverse_normalizer, memory,
                            memory_mask, memory_inverse_normalizer, attn_mask)
    inputs = tf.reshape(inputs, [-1, self.model_dimension])
    tensor = tf.reshape(tensor, [-1, self.model_dimension])
    tensor = self.prx(tensor, mask_rank2, inverse_normalizer)
    if (self.parameters.mode == base_layers.TRAIN and
        self.activation_dropout_rate > 0.0):
      tensor = tf.nn.dropout(tensor, rate=self.activation_dropout_rate)
    inputs_plus_selfattn = self.q1(self.ln1(inputs + tensor))

    ffn_up = self.upprx(inputs_plus_selfattn, mask_rank2, inverse_normalizer)
    ffn_down = self.downprx(ffn_up, mask_rank2, inverse_normalizer)
    if (self.parameters.mode == base_layers.TRAIN and
        self.activation_dropout_rate > 0.0):
      ffn_down = tf.nn.dropout(ffn_down, rate=self.activation_dropout_rate)
    inputs_plus_ffn = self.q2(self.ln2(inputs_plus_selfattn + ffn_down))
    return tf.reshape(inputs_plus_ffn, [batch_size, -1, self.model_dimension])


class FunnelTransformerEncoderStack(base_layers.BaseLayer):
  """Transformer Encoder."""

  def __init__(self, num_layers, max_time_step, vocabulary_size, embedding_size,
               model_dimension, num_heads, intermediate_size, **kwargs):
    self.max_time_step = max_time_step
    self.pool_windows = kwargs.pop('pool_windows', [])
    assert len(self.pool_windows) == num_layers
    self.vocabulary_size = vocabulary_size
    activation_dropout_rate = kwargs.pop('activation_dropout_rate', 0.0)
    attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.0)
    self.layers = []
    for _ in range(num_layers):
      self.layers.append(
          FunnelTransformerEncoder(
              model_dimension=model_dimension,
              num_heads=num_heads,
              intermediate_size=intermediate_size,
              activation_dropout_rate=activation_dropout_rate,
              attention_dropout_rate=attention_dropout_rate,
              **kwargs))
    super(FunnelTransformerEncoderStack, self).__init__(**kwargs)

  def call(self, inputs, sequence_length):
    mask_rank2 = tf.sequence_mask(
        sequence_length, tf.shape(inputs)[1], dtype=tf.float32)
    mask_rank3 = tf.expand_dims(mask_rank2, axis=2)
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      inputs = inputs * mask_rank3
    pooled_inputs = inputs
    pooled_mask = mask_rank3
    pooled_inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(pooled_mask))
    memory = pooled_inputs
    memory_mask = pooled_mask
    memory_inverse_normalizer = pooled_inverse_normalizer

    for i, layer in enumerate(self.layers):
      if self.pool_windows[i] > 1:
        pooled_inputs = tf.nn.avg_pool(
            pooled_inputs, [self.pool_windows[i]],
            strides=[self.pool_windows[i]],
            padding='SAME')
        pooled_mask = pooled_mask[:, ::self.pool_windows[i], :]
        pooled_inverse_normalizer = tf.math.reciprocal(
            tf.reduce_sum(pooled_mask))
      attn_mask = tf.matmul(pooled_mask, memory_mask, transpose_b=True)
      pooled_outputs = layer(pooled_inputs, pooled_mask,
                             pooled_inverse_normalizer, memory, memory_mask,
                             memory_inverse_normalizer, attn_mask)
      pooled_inputs = pooled_outputs
      pooled_inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(pooled_mask))
      memory = pooled_inputs
      memory_mask = pooled_mask
      memory_inverse_normalizer = pooled_inverse_normalizer
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      pooled_outputs = pooled_outputs * pooled_mask
    return pooled_outputs, pooled_mask


class DecoderMultiheadAttention(base_layers.BaseLayer):
  """Multihead attention for decoder."""

  def __init__(self,
               model_dimension,
               num_heads,
               attention_dropout_rate=0.0,
               cached_kv=False,
               **kwargs):
    self.model_dimension = model_dimension
    self.num_heads = num_heads
    self.filters = model_dimension // num_heads
    self.cached_kv = cached_kv
    self.q_dense_layers = dense_layers.BaseQDense(
        units=model_dimension,
        activation=None,
        normalize=False,
        bias=False,
        **kwargs)
    self.kv_dense_layers = dense_layers.BaseQDenseVarLen(
        units=model_dimension * 2, activation=None, **kwargs)
    self.qactivation = quantization_layers.ActivationQuantization(**kwargs)
    self.attention_dropout_rate = attention_dropout_rate
    self.qconcat = quantization_layers.ConcatQuantization(axis=1, **kwargs)
    super(DecoderMultiheadAttention, self).__init__(**kwargs)

  def call(self,
           inputs,
           input_mask,
           input_inverse_normalizer,
           memory=None,
           memory_mask=None,
           memory_inverse_normalizer=None,
           attn_mask=None):
    bsz = self.get_batch_dimension(inputs)
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(input_mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension

    inputs_rank2 = tf.reshape(inputs, [-1, self.model_dimension])
    q_tensor = self.q_dense_layers(inputs_rank2)

    if memory is not None:
      self._assert_rank_and_type(memory, 2)
      self._assert_rank_and_type(memory_mask, 2)
      if self.cached_kv:
        # Keys and Values are cached and reused at each layer.
        assert memory.get_shape().as_list()[1] == 2 * self.model_dimension
        kv_tensors = memory
      else:
        kv_tensors = self.kv_dense_layers(memory, memory_mask,
                                          memory_inverse_normalizer)
    else:
      kv_tensors = self.kv_dense_layers(inputs_rank2)
    if self.parameters.mode not in [base_layers.TFLITE, base_layers.PREDICT]:
      q_tensor = tf.reshape(q_tensor, [bsz, -1, self.num_heads, self.filters])
      kv_tensors = tf.reshape(kv_tensors,
                              [bsz, -1, 2, self.num_heads, self.filters])
      kv_tensors = tf.unstack(kv_tensors, axis=2)
    else:
      q_tensor = tf.split(q_tensor, self.num_heads, axis=1)
      kv_tensors = tf.split(kv_tensors, self.num_heads * 2, axis=1)

    if self.parameters.mode in [base_layers.TRAIN, base_layers.EVAL]:
      assert attn_mask is not None
      if (self.attention_dropout_rate > 0.0 and
          self.parameters.mode == base_layers.TRAIN):
        attn_mask *= self.random_drop_to_zero(attn_mask,
                                              self.attention_dropout_rate)
      attn_mask = tf.expand_dims(attn_mask, 1)
      invalid_mask = (1 - attn_mask) * self.parameters.invalid_logit
      queries = tf.transpose(q_tensor, [0, 2, 1, 3])
      keys = tf.transpose(kv_tensors[0], [0, 2, 1, 3])
      values = tf.transpose(kv_tensors[1], [0, 2, 1, 3])

      attn_logits = self.qactivation(tf.matmul(queries, keys, transpose_b=True))
      attn_logits_masked = attn_logits * attn_mask + invalid_mask
      attention = tf.nn.softmax(attn_logits_masked)
      attention = self.qrange_sigmoid(attention, tf_only=True)
      result = tf.matmul(attention, values)
      result = tf.transpose(result, [0, 2, 1, 3])
      result = tf.reshape(result, [bsz, -1, self.model_dimension])
      return self.qconcat([result])
    else:
      # We need to invoke the keras layer before calling APIs that it provides
      # such as quantize_using_range.
      self.qconcat(None)
      context = []
      for head in range(self.num_heads):
        queries = q_tensor[head]
        if self.parameters.mode == base_layers.PREDICT:
          # PREDICT mode assumes callers tile and merge beam size with batch
          # size. Hence extracting the first entry in the tile to compute
          # attention.
          keys = tf.split(kv_tensors[head], bsz, axis=0)
          keys = keys[0]
          values = tf.split(kv_tensors[head + self.num_heads], bsz, axis=0)
          values = values[0]
        else:
          keys = kv_tensors[head]
          values = kv_tensors[head + self.num_heads]
        attn_logits_masked = self.qactivation(
            tf.matmul(queries, keys, transpose_b=True))
        attention = tf.nn.softmax(attn_logits_masked)
        attention = self.qrange_sigmoid(attention, tf_only=True)
        context.append(
            self.qconcat.quantize_using_range(tf.matmul(attention, values)))
      # Concatenating heads along axis 1.
      result = self.qconcat.quantize_using_range(tf.concat(context, axis=1))
      return tf.reshape(result, [-1, 1, self.model_dimension])


class DecoderUniformAttention(base_layers.BaseLayer):
  """Decoder uniform attention."""

  def __init__(self,
               model_dimension,
               max_time_step,
               attention_dropout_rate=0.0,
               beam_size=1,
               **kwargs):
    self.model_dimension = model_dimension
    self.max_time_step = max_time_step
    self.beam_size = beam_size
    self.causal_mask = tf.expand_dims(
        tf.linalg.band_part(tf.ones([max_time_step, max_time_step]), -1, 0), 0)
    self.dense_layers = dense_layers.BaseQDenseVarLen(
        units=model_dimension,
        activation=None,
        normalize=False,
        bias=False,
        rank=3,
        **kwargs)
    self.qoutput = quantization_layers.ActivationQuantization(**kwargs)
    super(DecoderUniformAttention, self).__init__(**kwargs)

  def get_uniform_attention(self, attn_mask=None):
    """Generates uniform attention matrix using `causal_mask`."""
    mask = tf.math.divide_no_nan(
        self.causal_mask,
        tf.reduce_sum(self.causal_mask, axis=-1, keepdims=True))
    if attn_mask is not None:
      self._assert_rank_and_type(attn_mask, 3)
      mask = mask * attn_mask
    return mask

  def call(self,
           inputs,
           mask,
           inverse_normalizer,
           step=None,
           beam_indices=None,
           cache=None,
           attn_mask=None):
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    assert inputs.get_shape().as_list()[-1] == self.model_dimension

    layer_out = self.dense_layers(inputs, mask, inverse_normalizer)

    # TFLite mode is handled with a custom op.
    if self.parameters.mode == base_layers.TFLITE:
      assert beam_indices is not None
      assert step is not None
      layer_out = tf_custom_ops_py.uniform_causal_attn(
          layer_out, step, beam_indices, self.model_dimension, self.beam_size)
    else:
      # Cache is used for TF Predict and Eval modes.
      if cache is None:
        attention_matrix = self.get_uniform_attention(attn_mask)
        layer_out = tf.matmul(attention_matrix, layer_out)
      else:
        assert self.parameters.mode in [base_layers.PREDICT, base_layers.EVAL]
        assert step is not None
        cache['uniform_avg'] = layer_out + cache['uniform_avg']
        layer_out = cache['uniform_avg'] / tf.cast(step, dtype=tf.float32)
    return self.qoutput(layer_out)
