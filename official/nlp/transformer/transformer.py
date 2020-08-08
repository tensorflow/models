# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.modeling import tf_utils
from official.modeling.activations import attention_initializer
from official.nlp.modeling import layers
from official.nlp.modeling.layers import position_embedding
from official.nlp.modeling.layers import transformer
from official.nlp.modeling.models import seq2seq_transformer
from official.nlp.modeling.ops import beam_search
from official.nlp.transformer import metrics
from official.nlp.transformer import model_utils
from official.nlp.transformer.utils.tokenizer import EOS_ID


# Disable the not-callable lint error, since it claims many objects are not
# callable when they actually are.
# pylint: disable=not-callable


def create_model(params, is_train):
  """Creates transformer model."""
  with tf.name_scope("model"):
    if is_train:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
      internal_model = seq2seq_transformer.Seq2SeqTransformer(
          params, name="transformer_v2")
      logits = internal_model([inputs, targets], training=is_train)
      vocab_size = params["vocab_size"]
      label_smoothing = params["label_smoothing"]
      if params["enable_metrics_in_training"]:
        logits = metrics.MetricLayer(vocab_size)([logits, targets])
      logits = tf.keras.layers.Lambda(lambda x: x, name="logits",
                                      dtype=tf.float32)(logits)
      model = tf.keras.Model([inputs, targets], logits)
      loss = metrics.transformer_loss(
          logits, targets, label_smoothing, vocab_size)
      model.add_loss(loss)
      return model

    else:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      internal_model = seq2seq_transformer.Seq2SeqTransformer(
          params, name="transformer_v2")
      ret = internal_model([inputs], training=is_train)
      outputs, scores = ret["outputs"], ret["scores"]
      return tf.keras.Model(inputs, [outputs, scores])
