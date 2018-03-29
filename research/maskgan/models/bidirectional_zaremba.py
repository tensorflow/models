# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Simple bidirectional model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def discriminator(hparams, sequence, is_training, reuse=None):
  """Define the bidirectional Discriminator graph."""
  sequence = tf.cast(sequence, tf.int32)

  if FLAGS.dis_share_embedding:
    assert hparams.dis_rnn_size == hparams.gen_rnn_size, (
        'If you wish to share Discriminator/Generator embeddings, they must be'
        ' same dimension.')
    with tf.variable_scope('gen/rnn', reuse=True):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  with tf.variable_scope('dis', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          hparams.dis_rnn_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and FLAGS.keep_prob < 1:

      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=FLAGS.keep_prob)

    cell_fwd = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.dis_num_layers)],
        state_is_tuple=True)

    cell_bwd = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.dis_num_layers)],
        state_is_tuple=True)

    state_fwd = cell_fwd.zero_state(FLAGS.batch_size, tf.float32)
    state_bwd = cell_bwd.zero_state(FLAGS.batch_size, tf.float32)

    if not FLAGS.dis_share_embedding:
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.dis_rnn_size])

    rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)
    if is_training and FLAGS.keep_prob < 1:
      rnn_inputs = tf.nn.dropout(rnn_inputs, FLAGS.keep_prob)
    rnn_inputs = tf.unstack(rnn_inputs, axis=1)

    with tf.variable_scope('rnn') as vs:
      outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
          cell_fwd, cell_bwd, rnn_inputs, state_fwd, state_bwd, scope=vs)

      # Prediction is linear output for Discriminator.
      predictions = tf.contrib.layers.linear(outputs, 1, scope=vs)

      predictions = tf.transpose(predictions, [1, 0, 2])
      return tf.squeeze(predictions, axis=2)
