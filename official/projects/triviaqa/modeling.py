# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Modeling for TriviaQA."""
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.configs import encoders


class TriviaQaHead(tf.keras.layers.Layer):
  """Computes logits given token and global embeddings."""

  def __init__(self,
               intermediate_size,
               intermediate_activation=tf_utils.get_activation('gelu'),
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               **kwargs):
    super(TriviaQaHead, self).__init__(**kwargs)
    self._attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)
    self._intermediate_dense = tf.keras.layers.Dense(intermediate_size)
    self._intermediate_activation = tf.keras.layers.Activation(
        intermediate_activation)
    self._output_dropout = tf.keras.layers.Dropout(dropout_rate)
    self._output_layer_norm = tf.keras.layers.LayerNormalization()
    self._logits_dense = tf.keras.layers.Dense(2)

  def build(self, input_shape):
    output_shape = input_shape['token_embeddings'][-1]
    self._output_dense = tf.keras.layers.Dense(output_shape)
    super(TriviaQaHead, self).build(input_shape)

  def call(self, inputs, training=None):
    token_embeddings = inputs['token_embeddings']
    token_ids = inputs['token_ids']
    question_lengths = inputs['question_lengths']
    x = self._attention_dropout(token_embeddings, training=training)
    intermediate_outputs = self._intermediate_dense(x)
    intermediate_outputs = self._intermediate_activation(intermediate_outputs)
    outputs = self._output_dense(intermediate_outputs)
    outputs = self._output_dropout(outputs, training=training)
    outputs = self._output_layer_norm(outputs + token_embeddings)
    logits = self._logits_dense(outputs)
    logits -= tf.expand_dims(
        tf.cast(tf.equal(token_ids, 0), tf.float32) + tf.sequence_mask(
            question_lengths, logits.shape[-2], dtype=tf.float32), -1) * 1e6
    return logits


class TriviaQaModel(tf.keras.Model):
  """Model for TriviaQA."""

  def __init__(self, model_config: encoders.EncoderConfig, sequence_length: int,
               **kwargs):
    inputs = dict(
        token_ids=tf.keras.Input((sequence_length,), dtype=tf.int32),
        question_lengths=tf.keras.Input((), dtype=tf.int32))
    encoder = encoders.build_encoder(model_config)
    x = encoder(
        dict(
            input_word_ids=inputs['token_ids'],
            input_mask=tf.cast(inputs['token_ids'] > 0, tf.int32),
            input_type_ids=1 -
            tf.sequence_mask(inputs['question_lengths'], sequence_length,
                             tf.int32)))['sequence_output']
    logits = TriviaQaHead(
        model_config.get().intermediate_size,
        dropout_rate=model_config.get().dropout_rate,
        attention_dropout_rate=model_config.get().attention_dropout_rate)(
            dict(
                token_embeddings=x,
                token_ids=inputs['token_ids'],
                question_lengths=inputs['question_lengths']))
    super(TriviaQaModel, self).__init__(inputs, logits, **kwargs)
    self._encoder = encoder

  @property
  def encoder(self):
    return self._encoder


class SpanOrCrossEntropyLoss(tf.keras.losses.Loss):
  """Cross entropy loss for multiple correct answers.

  See https://arxiv.org/abs/1710.10723.
  """

  def call(self, y_true, y_pred):
    y_pred_masked = y_pred - tf.cast(y_true < 0.5, tf.float32) * 1e6
    or_cross_entropy = (
        tf.math.reduce_logsumexp(y_pred, axis=-2) -
        tf.math.reduce_logsumexp(y_pred_masked, axis=-2))
    return tf.math.reduce_sum(or_cross_entropy, -1)


def smooth_labels(label_smoothing, labels, question_lengths, token_ids):
  mask = 1. - (
      tf.cast(tf.equal(token_ids, 0), tf.float32) +
      tf.sequence_mask(question_lengths, labels.shape[-2], dtype=tf.float32))
  num_classes = tf.expand_dims(tf.math.reduce_sum(mask, -1, keepdims=True), -1)
  labels = (1. - label_smoothing) * labels + (label_smoothing / num_classes)
  return labels * tf.expand_dims(mask, -1)
