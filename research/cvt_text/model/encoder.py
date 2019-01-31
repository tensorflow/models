# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""CNN-BiLSTM sentence encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from base import embeddings
from model import model_helpers


class Encoder(object):
  def __init__(self, config, inputs, pretrained_embeddings):
    self._config = config
    self._inputs = inputs

    self.word_reprs = self._get_word_reprs(pretrained_embeddings)
    self.uni_fw, self.uni_bw = self._get_unidirectional_reprs(self.word_reprs)
    self.uni_reprs = tf.concat([self.uni_fw, self.uni_bw], axis=-1)
    self.bi_fw, self.bi_bw, self.bi_reprs = self._get_bidirectional_reprs(
        self.uni_reprs)

  def _get_word_reprs(self, pretrained_embeddings):
    with tf.variable_scope('word_embeddings'):
      word_embedding_matrix = tf.get_variable(
          'word_embedding_matrix', initializer=pretrained_embeddings)
      word_embeddings = tf.nn.embedding_lookup(
          word_embedding_matrix, self._inputs.words)
      word_embeddings = tf.nn.dropout(word_embeddings, self._inputs.keep_prob)
      word_embeddings *= tf.get_variable('emb_scale', initializer=1.0)

    if not self._config.use_chars:
      return word_embeddings

    with tf.variable_scope('char_embeddings'):
      char_embedding_matrix = tf.get_variable(
          'char_embeddings',
          shape=[embeddings.NUM_CHARS, self._config.char_embedding_size])
      char_embeddings = tf.nn.embedding_lookup(char_embedding_matrix,
                                               self._inputs.chars)
      shape = tf.shape(char_embeddings)
      char_embeddings = tf.reshape(
          char_embeddings,
          shape=[-1, shape[-2], self._config.char_embedding_size])
      char_reprs = []
      for filter_width in self._config.char_cnn_filter_widths:
        conv = tf.layers.conv1d(
            char_embeddings, self._config.char_cnn_n_filters, filter_width)
        conv = tf.nn.relu(conv)
        conv = tf.nn.dropout(tf.reduce_max(conv, axis=1),
                             self._inputs.keep_prob)
        conv = tf.reshape(conv, shape=[-1, shape[1],
                                       self._config.char_cnn_n_filters])
        char_reprs.append(conv)
      return tf.concat([word_embeddings] + char_reprs, axis=-1)

  def _get_unidirectional_reprs(self, word_reprs):
    with tf.variable_scope('unidirectional_reprs'):
      word_lstm_input_size = (
          self._config.word_embedding_size if not self._config.use_chars else
          (self._config.word_embedding_size +
           len(self._config.char_cnn_filter_widths)
           * self._config.char_cnn_n_filters))
      word_reprs.set_shape([None, None, word_lstm_input_size])
      (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
          model_helpers.multi_lstm_cell(self._config.unidirectional_sizes,
                                        self._inputs.keep_prob,
                                        self._config.projection_size),
          model_helpers.multi_lstm_cell(self._config.unidirectional_sizes,
                                        self._inputs.keep_prob,
                                        self._config.projection_size),
          word_reprs,
          dtype=tf.float32,
          sequence_length=self._inputs.lengths,
          scope='unilstm'
      )
      return outputs_fw, outputs_bw

  def _get_bidirectional_reprs(self, uni_reprs):
    with tf.variable_scope('bidirectional_reprs'):
      current_outputs = uni_reprs
      outputs_fw, outputs_bw = None, None
      for size in self._config.bidirectional_sizes:
        (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            model_helpers.lstm_cell(size, self._inputs.keep_prob,
                                    self._config.projection_size),
            model_helpers.lstm_cell(size, self._inputs.keep_prob,
                                    self._config.projection_size),
            current_outputs,
            dtype=tf.float32,
            sequence_length=self._inputs.lengths,
            scope='bilstm'
        )
        current_outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)
      return outputs_fw, outputs_bw, current_outputs
