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

"""Simple RNN model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def generator(hparams,
              inputs,
              targets,
              targets_present,
              is_training,
              is_validating,
              reuse=None):
  """Define the Generator graph.

    G will now impute tokens that have been masked from the input seqeunce.
  """
  tf.logging.warning(
      'Undirectional generative model is not a useful model for this MaskGAN '
      'because future context is needed.  Use only for debugging purposes.')
  init_scale = 0.05
  initializer = tf.random_uniform_initializer(-init_scale, init_scale)
  with tf.variable_scope('gen', reuse=reuse, initializer=initializer):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(hparams.gen_rnn_size,
                                          forget_bias=0.0,
                                          state_is_tuple=True,
                                          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and FLAGS.keep_prob < 1:

      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=FLAGS.keep_prob)

    cell_gen = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.gen_num_layers)],
        state_is_tuple=True)

    initial_state = cell_gen.zero_state(FLAGS.batch_size, tf.float32)

    with tf.variable_scope('rnn'):
      sequence, logits, log_probs = [], [], []
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])
      softmax_w = tf.get_variable('softmax_w',
                                  [hparams.gen_rnn_size, FLAGS.vocab_size])
      softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

      rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)

      if is_training and FLAGS.keep_prob < 1:
        rnn_inputs = tf.nn.dropout(rnn_inputs, FLAGS.keep_prob)

      fake = None
      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        # Input to the model is the first token to provide context.  The
        # model will then predict token t > 0.
        if t == 0:
          # Always provide the real input at t = 0.
          state_gen = initial_state
          rnn_inp = rnn_inputs[:, t]

        # If the input is present, read in the input at t.
        # If the input is not present, read in the previously generated.
        else:
          real_rnn_inp = rnn_inputs[:, t]
          fake_rnn_inp = tf.nn.embedding_lookup(embedding, fake)

          # While validating, the decoder should be operating in teacher
          # forcing regime.  Also, if we're just training with cross_entropy
          # use teacher forcing.
          if is_validating or (is_training and
                               FLAGS.gen_training_strategy == 'cross_entropy'):
            rnn_inp = real_rnn_inp
          else:
            rnn_inp = tf.where(targets_present[:, t - 1], real_rnn_inp,
                               fake_rnn_inp)

        # RNN.
        rnn_out, state_gen = cell_gen(rnn_inp, state_gen)
        logit = tf.matmul(rnn_out, softmax_w) + softmax_b

        # Real sample.
        real = targets[:, t]

        categorical = tf.contrib.distributions.Categorical(logits=logit)
        fake = categorical.sample()
        log_prob = categorical.log_prob(fake)

        # Output for Generator will either be generated or the input.
        #
        # If present:   Return real.
        # If not present:  Return fake.
        output = tf.where(targets_present[:, t], real, fake)

        # Add to lists.
        sequence.append(output)
        log_probs.append(log_prob)
        logits.append(logit)

      # Produce the RNN state had the model operated only
      # over real data.
      real_state_gen = initial_state
      for t in xrange(FLAGS.sequence_length):
        tf.get_variable_scope().reuse_variables()

        rnn_inp = rnn_inputs[:, t]

        # RNN.
        rnn_out, real_state_gen = cell_gen(rnn_inp, real_state_gen)

      final_state = real_state_gen

  return (tf.stack(sequence, axis=1), tf.stack(logits, axis=1), tf.stack(
      log_probs, axis=1), initial_state, final_state)


def discriminator(hparams, sequence, is_training, reuse=None):
  """Define the Discriminator graph."""
  tf.logging.warning(
      'Undirectional Discriminative model is not a useful model for this '
      'MaskGAN because future context is needed.  Use only for debugging '
      'purposes.')
  sequence = tf.cast(sequence, tf.int32)

  with tf.variable_scope('dis', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(hparams.dis_rnn_size,
                                          forget_bias=0.0,
                                          state_is_tuple=True,
                                          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and FLAGS.keep_prob < 1:

      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=FLAGS.keep_prob)

    cell_dis = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.dis_num_layers)],
        state_is_tuple=True)

    state_dis = cell_dis.zero_state(FLAGS.batch_size, tf.float32)

    with tf.variable_scope('rnn') as vs:
      predictions = []
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.dis_rnn_size])

      rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

      if is_training and FLAGS.keep_prob < 1:
        rnn_inputs = tf.nn.dropout(rnn_inputs, FLAGS.keep_prob)

      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        rnn_in = rnn_inputs[:, t]
        rnn_out, state_dis = cell_dis(rnn_in, state_dis)

        # Prediction is linear output for Discriminator.
        pred = tf.contrib.layers.linear(rnn_out, 1, scope=vs)

        predictions.append(pred)
  predictions = tf.stack(predictions, axis=1)
  return tf.squeeze(predictions, axis=2)
