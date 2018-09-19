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

"""Critic model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
from regularization import variational_dropout

FLAGS = tf.app.flags.FLAGS


def critic_seq2seq_vd_derivative(hparams, sequence, is_training, reuse=None):
  """Define the Critic graph which is derived from the seq2seq_vd
  Discriminator.  This will be initialized with the same parameters as the
  language model and will share the forward RNN components with the
  Discriminator.   This estimates the V(s_t), where the state
  s_t = x_0,...,x_t-1.
  """
  assert FLAGS.discriminator_model == 'seq2seq_vd'
  sequence = tf.cast(sequence, tf.int32)

  if FLAGS.dis_share_embedding:
    assert hparams.dis_rnn_size == hparams.gen_rnn_size, (
        'If you wish to share Discriminator/Generator embeddings, they must be'
        ' same dimension.')
    with tf.variable_scope('gen/decoder/rnn', reuse=True):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])
  else:
    with tf.variable_scope('dis/decoder/rnn', reuse=True):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.dis_rnn_size])

  with tf.variable_scope(
      'dis/decoder/rnn/multi_rnn_cell', reuse=True) as dis_scope:

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          hparams.dis_rnn_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=True)

    attn_cell = lstm_cell
    if is_training and hparams.dis_vd_keep_prob < 1:

      def attn_cell():
        return variational_dropout.VariationalDropoutWrapper(
            lstm_cell(), FLAGS.batch_size, hparams.dis_rnn_size,
            hparams.dis_vd_keep_prob, hparams.dis_vd_keep_prob)

    cell_critic = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.dis_num_layers)],
        state_is_tuple=True)

  with tf.variable_scope('critic', reuse=reuse):
    state_dis = cell_critic.zero_state(FLAGS.batch_size, tf.float32)

    def make_mask(keep_prob, units):
      random_tensor = keep_prob
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
      return tf.floor(random_tensor) / keep_prob

    if is_training:
      output_mask = make_mask(hparams.dis_vd_keep_prob, hparams.dis_rnn_size)

    with tf.variable_scope('rnn') as vs:
      values = []

      rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        if t == 0:
          rnn_in = tf.zeros_like(rnn_inputs[:, 0])
        else:
          rnn_in = rnn_inputs[:, t - 1]
        rnn_out, state_dis = cell_critic(rnn_in, state_dis, scope=dis_scope)

        if is_training:
          rnn_out *= output_mask

        # Prediction is linear output for Discriminator.
        value = tf.contrib.layers.linear(rnn_out, 1, scope=vs)

        values.append(value)
  values = tf.stack(values, axis=1)
  return tf.squeeze(values, axis=2)
