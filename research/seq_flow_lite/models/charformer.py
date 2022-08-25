# Copyright 2022 The TensorFlow Authors All Rights Reserved.
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
"""Charformer based model for in-training tokenization."""
from absl import logging
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import embedding_layers # import seq_flow_lite module
from layers import misc_layers # import seq_flow_lite module
from layers import normalization_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module
from models import transformer_encoder # import seq_flow_lite module


class Encoder(tf.keras.layers.Layer):
  """Encoder with GBST and Transformer layers."""

  def __init__(self, config, mode, **kwargs):
    super(Encoder, self).__init__(**kwargs)

    def _get_params(varname, default_value=None):
      value = config[varname] if varname in config else default_value
      default = "" if varname in config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    _get_params("labels", [])
    _get_params("regularizer_scale")
    _get_params("quantize")
    _get_params("feature_size")
    _get_params("bottleneck_size")

    self.max_seq_len = config.get("max_seq_len", 128)
    self.gbst_max_token_len = config.get("gbst_max_token_len", 128)
    # Including 3 additional special token ids (0=padding, 1=EOS, 2=UNK).
    self.vocabulary_size = config.get("vocabulary_size", 259)
    self.parameters = base_layers.Parameters(
        mode, quantize=self.quantize, regularizer_scale=self.regularizer_scale)

    self.embedding = embedding_layers.EmbeddingLayer(
        shape=[self.vocabulary_size, self.feature_size],
        parameters=self.parameters)
    self.gbst_downsample_rate = config.get("gbst_downsample_rate", 1)
    self.positional_embedding = embedding_layers.EmbeddingLayer(
        shape=[self.gbst_max_token_len, self.feature_size],
        parameters=self.parameters)
    self.ln = normalization_layers.LayerNormalization(
        parameters=self.parameters)
    self.qact = quantization_layers.ActivationQuantization(
        parameters=self.parameters)

    self.bottleneck_layer = None
    gbst_size = self.feature_size
    if self.bottleneck_size != self.feature_size:
      self.bottleneck_layer = dense_layers.BaseQDenseVarLen(
          self.bottleneck_size,
          rank=3,
          normalize=False,
          activation=None,
          parameters=self.parameters)
      gbst_size = self.bottleneck_size

    self.gbst_max_subword_block_width = config.get(
        "gbst_max_subword_block_width", 5)
    self.gbst_conv_kernel_size = config.get("gbst_conv_kernel_size", 5)
    self.gbst_block_mixing_mode = config.get("gbst_block_mixing_mode", None)
    self.gbst_layer = misc_layers.GBSTLayerV2(
        feature_size=gbst_size,
        max_seq_len=self.gbst_max_token_len,
        downsample_rate=self.gbst_downsample_rate,
        max_subword_block_width=self.gbst_max_subword_block_width,
        conv_kernel_size=self.gbst_conv_kernel_size,
        block_mixing_mode=self.gbst_block_mixing_mode,
        parameters=self.parameters)

    self.pool_windows = config.get("pool_windows", None)
    if self.pool_windows:
      self.transformer_encoder_layer = transformer_encoder.FunnelTransformerModel(
          config, mode)
    else:
      self.transformer_encoder_layer = transformer_encoder.ModelWithEmbeddings(
          config, mode)
    self.attention_pool = misc_layers.AttentionPooling(
        parameters=self.parameters)
    self.num_classes = len(self.labels)
    if self.num_classes:
      self.final_fc = dense_layers.BaseQDense(
          units=self.num_classes,
          rank=2,
          parameters=self.parameters,
          activation=None)

  def call(self, token_ids, seq_length):
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      mask_rank2 = tf.ones(tf.shape(token_ids), dtype=tf.int32)
      seq_length = tf.reduce_sum(mask_rank2, axis=1)
      pos_indices = tf.cumsum(mask_rank2, axis=1, exclusive=True)
      pos_indices = tf.cast(pos_indices, dtype=tf.int32)
      pos_indices = tf.reshape(pos_indices, [1, -1])
    else:
      mask_rank2 = tf.sequence_mask(
          seq_length, tf.shape(token_ids)[1], dtype=tf.float32)
      pos_indices = tf.cumsum(mask_rank2, axis=1, exclusive=True)
      pos_indices = tf.cast(pos_indices, dtype=tf.int32)

    input_values = self.embedding(token_ids)
    pos_values = self.positional_embedding(pos_indices)
    input_embeds = self.qact(self.ln(input_values + pos_values))

    if self.bottleneck_layer is not None:
      maskr3 = tf.expand_dims(mask_rank2, axis=2)
      maskr3 = tf.cast(maskr3, tf.float32)
      bottleneck_output = self.bottleneck_layer(input_embeds, maskr3)
    else:
      bottleneck_output = input_embeds

    gbst_output = self.gbst_layer(bottleneck_output, seq_length)
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      mask_rank2 = tf.ones(tf.shape(gbst_output)[:-1], dtype=tf.float32)
      seq_length = tf.reduce_sum(mask_rank2, axis=1)
    else:
      seq_length = seq_length / self.gbst_downsample_rate

    if self.pool_windows:
      outputs, mask = self.transformer_encoder_layer(gbst_output,
                                                     seq_length)
      inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))

      pre_logits = self.attention_pool(outputs, mask, inverse_normalizer)
    else:
      outputs = self.transformer_encoder_layer(gbst_output, seq_length)
      mask = tf.sequence_mask(
          seq_length, tf.shape(outputs)[1], dtype=tf.float32)
      inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask))
      maskr3 = tf.expand_dims(mask, axis=2)
      pre_logits = self.attention_pool(outputs, maskr3, inverse_normalizer)
    if self.num_classes:
      return self.final_fc(pre_logits)
    else:
      return pre_logits
