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
# Lint as: python3
"""Implementation of PRADO model."""

import copy
from absl import logging
import numpy as np
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import conv_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import projection_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module
from tf_ops import tf_custom_ops_py # import seq_flow_lite module


class PaddedMaskedVarLenConv(conv_layers.EncoderQConvolutionVarLen):
  """A layer that performs padded masked convolution."""

  def __init__(self, invalid_value, ngram=2, skip_bigram=None, **kwargs):
    self.invalid_value = invalid_value
    assert ngram is None or (ngram >= 1 and ngram <= 5)
    assert skip_bigram is None or skip_bigram == 1 or skip_bigram == 2
    assert bool(ngram is None) != bool(skip_bigram is None)
    self.kwidth = ngram if ngram is not None else (skip_bigram + 2)
    mask = [1] * self.kwidth
    if skip_bigram is not None:
      mask[1], mask[skip_bigram] = 0, 0
    self.mask = np.array(mask, dtype="float32").reshape((1, self.kwidth, 1, 1))
    self.zero_pad = tf.keras.layers.ZeroPadding1D(padding=[0, self.kwidth - 1])
    super(PaddedMaskedVarLenConv, self).__init__(
        ksize=self.kwidth, rank=3, padding="VALID", activation=None, **kwargs)

  def call(self, inputs, mask, inverse_normalizer):
    self._assert_rank_and_type(inputs, 3)
    self._assert_rank_and_type(mask, 3)
    maskr4 = tf.expand_dims(mask, axis=1)
    inputs_padded = self.zero_pad(inputs)
    result = super(PaddedMaskedVarLenConv, self).call(inputs_padded, maskr4,
                                                      inverse_normalizer)
    if self.parameters.mode not in [base_layers.PREDICT, base_layers.TFLITE]:
      return result * mask + (1 - mask) * self.invalid_value
    return result

  def add_qweight(self, shape, num_bits=8):
    weight = super(PaddedMaskedVarLenConv, self).add_qweight(
        shape=shape, num_bits=num_bits)
    return weight * tf.convert_to_tensor(self.mask)


class AttentionPoolReduce(base_layers.BaseLayer):
  """Attention pooling and reduce."""

  def __init__(self, filters, ngram=2, skip_bigram=None, **kwargs):
    super(AttentionPoolReduce, self).__init__(**kwargs)
    self.filters = filters
    self.value = PaddedMaskedVarLenConv(
        0, filters=filters, ngram=ngram, skip_bigram=skip_bigram, **kwargs)
    self.attention_logits = PaddedMaskedVarLenConv(
        self.parameters.invalid_logit,
        filters=filters,
        ngram=ngram,
        skip_bigram=skip_bigram,
        **kwargs)

  def call(self, values_in, attention_in, mask, inverse_normalizer):
    self._assert_rank_and_type(values_in, 3)
    self._assert_rank_and_type(attention_in, 3)
    self._assert_rank_and_type(mask, 3)
    values = self.value(values_in, mask, inverse_normalizer)
    attention_logits = self.attention_logits(attention_in, mask,
                                             inverse_normalizer)

    if self.parameters.mode == base_layers.TFLITE:
      return tf_custom_ops_py.expected_value_op(attention_logits, values)
    else:
      attention_logits = tf.transpose(attention_logits, [0, 2, 1])
      values = tf.transpose(values, [0, 2, 1])
      attention = tf.nn.softmax(attention_logits)
      return tf.reduce_sum(attention * values, axis=2)


class Encoder(tf.keras.layers.Layer):
  """A PRADO keras model."""

  def __init__(self, config, mode):
    super(Encoder, self).__init__()

    def _get_params(varname, default_value=None):
      value = config[varname] if varname in config else default_value
      default = "" if varname in config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    _get_params("labels")
    _get_params("quantize", True)
    _get_params("embedding_regularizer_scale", 35e-3)
    _get_params("embedding_size", 64)
    _get_params("unigram_channels", 0)
    _get_params("bigram_channels", 0)
    _get_params("trigram_channels", 0)
    _get_params("fourgram_channels", 0)
    _get_params("fivegram_channels", 0)
    _get_params("skip1bigram_channels", 0)
    _get_params("skip2bigram_channels", 0)
    _get_params("network_regularizer_scale", 1e-4)
    _get_params("keep_prob", 0.5)
    self.num_classes = len(self.labels)

    self.parameters = base_layers.Parameters(
        mode,
        quantize=self.quantize,
        regularizer_scale=self.embedding_regularizer_scale)
    self.values_fc = dense_layers.BaseQDenseVarLen(
        units=self.embedding_size, rank=3, parameters=self.parameters)
    self.attention_fc = dense_layers.BaseQDenseVarLen(
        units=self.embedding_size, rank=3, parameters=self.parameters)
    self.dropout = tf.keras.layers.Dropout(rate=(1 - self.keep_prob))

    self.parameters = copy.copy(self.parameters)
    self.parameters.regularizer_scale = self.network_regularizer_scale
    self.attention_pool_layers = []
    self._add_attention_pool_layer(self.unigram_channels, 1)
    self._add_attention_pool_layer(self.bigram_channels, 2)
    self._add_attention_pool_layer(self.trigram_channels, 3)
    self._add_attention_pool_layer(self.fourgram_channels, 4)
    self._add_attention_pool_layer(self.fivegram_channels, 5)
    self._add_attention_pool_layer(self.skip1bigram_channels, None, 1)
    self._add_attention_pool_layer(self.skip2bigram_channels, None, 2)

    self.concat_quantizer = quantization_layers.ConcatQuantization(
        axis=1, parameters=self.parameters)
    self.final_fc = dense_layers.BaseQDense(
        units=self.num_classes,
        rank=2,
        parameters=self.parameters,
        activation=None)

  def _add_attention_pool_layer(self, channels, ngram, skip_bigram=None):
    if channels > 0:
      self.attention_pool_layers.append(
          AttentionPoolReduce(
              filters=channels,
              skip_bigram=skip_bigram,
              ngram=ngram,
              parameters=self.parameters))

  def _apply_fc_dropout(self, layer, inputs, mask, inverse_normalizer):
    outputs = layer(inputs, mask, inverse_normalizer)
    if self.parameters.mode == base_layers.TRAIN:
      return self.dropout(outputs)
    return outputs

  def call(self, projection, seq_length):
    mask = tf.sequence_mask(
        seq_length, tf.shape(projection)[1], dtype=tf.float32)
    inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
    maskr3 = tf.expand_dims(mask, axis=2)
    values_in = self._apply_fc_dropout(self.values_fc, projection, mask,
                                       inverse_normalizer)
    attention_in = self._apply_fc_dropout(self.attention_fc, projection, mask,
                                          inverse_normalizer)
    tensors = [
        layer(values_in, attention_in, maskr3, inverse_normalizer)
        for layer in self.attention_pool_layers
    ]
    pre_logits = self.concat_quantizer(tensors)
    return self.final_fc(pre_logits)


class Model(Encoder):

  def __init__(self, config, mode):
    super(Model, self).__init__(config, mode)
    self.projection = projection_layers.ProjectionLayer(config, mode)

  def call(self, inputs):
    projection, seq_length = self.projection(inputs)
    return super(Model, self).call(projection, seq_length)
