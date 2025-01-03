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
"""ByteQRNN based model for in-training tokenization.

Sample model params:

"feature_size": 128,                  # Embedding size for each byte
"gbst_max_token_len": 1024,           # Max sequence length of bytes in GBST
"gbst_downsample_rate": 1,            # Downsample factor for GBST output
"bottleneck_size": 128,               # Bottleneck size before feeding to QRNN
"qrnn_state_size": 128,               # QRNN layer param
"qrnn_kernel_width": 3,               # QRNN layer param
"qrnn_zoneout_probability": 1e-2,     # QRNN layer param
"distortion_probability": 0.25,       # QRNN layer param
"number_qrnn_layers": 3,              # QRNN layer param
"labels": [],                         # List of labels for getting num classes
"regularizer_scale": 1e-5,            # L2 Regularization scale
"quantize": true,                     # Enable quantization
"multilabel": true,                   # If the output is Multilabel
"""
from absl import logging
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import dense_layers # import seq_flow_lite module
from layers import embedding_layers # import seq_flow_lite module
from layers import misc_layers # import seq_flow_lite module
from layers import qrnn_layers # import seq_flow_lite module


class Encoder(tf.keras.layers.Layer):
  """Encoder with GBST and QRNN layers."""

  def __init__(self, config, mode, **kwargs):
    super(Encoder, self).__init__(**kwargs)

    def _get_params(varname, default_value=None):
      value = config.get(varname, default_value)
      default = "" if varname in config else " (default)"
      logging.info("%s = %s%s", varname, value, default)
      setattr(self, varname, value)

    _get_params("feature_size")
    _get_params("bottleneck_size", self.feature_size)
    _get_params("qrnn_state_size")
    _get_params("qrnn_kernel_width", 3)
    _get_params("qrnn_zoneout_probability")
    _get_params("number_qrnn_layers")
    _get_params("labels", [])
    _get_params("regularizer_scale")
    _get_params("quantize")
    _get_params("gbst_max_token_len", 128)
    _get_params("gbst_downsample_rate", 1)
    _get_params("gbst_max_subword_block_width", 4)
    _get_params("gbst_conv_kernel_size", 5)
    _get_params("gbst_block_mixing_mode")
    _get_params("gbst_add_block_pos_embed", False)
    _get_params("attn_pool_output", True)

    self.num_classes = len(config.get("labels", []))

    self.parameters = base_layers.Parameters(
        mode, quantize=self.quantize, regularizer_scale=self.regularizer_scale)
    # Including 3 additional special token ids (0=padding, 1=EOS, 2=UNK).
    self.vocabulary_size = 259
    self.embedding = embedding_layers.EmbeddingLayer(
        shape=[self.vocabulary_size, self.feature_size],
        parameters=self.parameters)

    self.bottleneck_layer = dense_layers.BaseQDenseVarLen(
        units=self.bottleneck_size,
        rank=3,
        parameters=self.parameters)

    self.gbst_layer = misc_layers.GBSTLayerV2(
        feature_size=self.bottleneck_size,
        max_seq_len=self.gbst_max_token_len,
        downsample_rate=self.gbst_downsample_rate,
        max_subword_block_width=self.gbst_max_subword_block_width,
        conv_kernel_size=self.gbst_conv_kernel_size,
        block_mixing_mode=self.gbst_block_mixing_mode,
        add_block_pos_embed=self.gbst_add_block_pos_embed,
        parameters=self.parameters)

    self.qrnn_stack = qrnn_layers.QRNNBidirectionalStack(
        parameters=self.parameters,
        zoneout_probability=self.qrnn_zoneout_probability,
        kwidth=self.qrnn_kernel_width,
        state_size=self.qrnn_state_size,
        num_layers=self.number_qrnn_layers)
    self.attention_pool = misc_layers.AttentionPooling(
        parameters=self.parameters)

    if self.num_classes:
      self.final_fc = dense_layers.BaseQDense(
          units=self.num_classes,
          rank=2,
          parameters=self.parameters,
          activation=None)

  def call(self, token_ids, seq_length):
    input_embeds = self.embedding(token_ids)
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      mask_rank2 = tf.ones(tf.shape(input_embeds)[:-1], dtype=tf.float32)
      seq_length = tf.reduce_sum(mask_rank2, axis=1)
    else:
      mask_rank2 = tf.sequence_mask(
          seq_length, tf.shape(input_embeds)[1], dtype=tf.float32)
    maskr3 = tf.expand_dims(mask_rank2, axis=2)
    gbst_input = self.bottleneck_layer(input_embeds, maskr3)
    gbst_output = self.gbst_layer(gbst_input, seq_length)
    if self.parameters.mode in [base_layers.PREDICT, base_layers.TFLITE]:
      mask_rank2 = tf.ones(tf.shape(gbst_output)[:-1], dtype=tf.float32)
      seq_length = tf.reduce_sum(mask_rank2, axis=1)
    else:
      seq_length = seq_length / self.gbst_downsample_rate
    mask_rank2 = tf.sequence_mask(
        seq_length, tf.shape(gbst_output)[1], dtype=tf.float32)
    inverse_normalizer = tf.math.reciprocal(tf.reduce_sum(mask_rank2))
    maskr3 = tf.expand_dims(mask_rank2, axis=2)
    outputs = self.qrnn_stack(gbst_output, maskr3, inverse_normalizer)
    if self.attn_pool_output:
      pre_logits = self.attention_pool(outputs, maskr3, inverse_normalizer)
      if self.num_classes:
        return self.final_fc(pre_logits)
      else:
        return pre_logits
    else:
      return outputs

