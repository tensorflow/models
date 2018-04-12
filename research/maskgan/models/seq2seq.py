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

"""Simple seq2seq model definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange
from models import attention_utils

# ZoneoutWrapper.
from regularization import zoneout

FLAGS = tf.app.flags.FLAGS


def transform_input_with_is_missing_token(inputs, targets_present):
  """Transforms the inputs to have missing tokens when it's masked out.  The
  mask is for the targets, so therefore, to determine if an input at time t is
  masked, we have to check if the target at time t - 1 is masked out.

  e.g.
    inputs = [a, b, c, d]
    targets = [b, c, d, e]
    targets_present = [1, 0, 1, 0]

  then,
    transformed_input = [a, b, <missing>, d]

  Args:
    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens
      up to, but not including, vocab_size.
    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with
      True representing the presence of the word.

  Returns:
    transformed_input:  tf.int32 Tensor of shape [batch_size, sequence_length]
      which takes on value of inputs when the input is present and takes on
      value=vocab_size to indicate a missing token.
  """
  # To fill in if the input is missing.
  input_missing = tf.constant(
      FLAGS.vocab_size,
      dtype=tf.int32,
      shape=[FLAGS.batch_size, FLAGS.sequence_length])

  # The 0th input will always be present to MaskGAN.
  zeroth_input_present = tf.constant(True, tf.bool, shape=[FLAGS.batch_size, 1])

  # Input present mask.
  inputs_present = tf.concat(
      [zeroth_input_present, targets_present[:, :-1]], axis=1)

  transformed_input = tf.where(inputs_present, inputs, input_missing)
  return transformed_input


def gen_encoder(hparams, inputs, targets_present, is_training, reuse=None):
  """Define the Encoder graph."""
  # We will use the same variable from the decoder.
  if FLAGS.seq2seq_share_embedding:
    with tf.variable_scope('decoder/rnn'):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  with tf.variable_scope('encoder', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.LayerNormBasicLSTMCell(
          hparams.gen_rnn_size, reuse=reuse)

    attn_cell = lstm_cell
    if FLAGS.zoneout_drop_prob > 0.0:

      def attn_cell():
        return zoneout.ZoneoutWrapper(
            lstm_cell(),
            zoneout_drop_prob=FLAGS.zoneout_drop_prob,
            is_training=is_training)

    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.gen_num_layers)],
        state_is_tuple=True)

    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

    # Add a missing token for inputs not present.
    real_inputs = inputs
    masked_inputs = transform_input_with_is_missing_token(
        inputs, targets_present)

    with tf.variable_scope('rnn'):
      hidden_states = []

      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size + 1, hparams.gen_rnn_size])

      real_rnn_inputs = tf.nn.embedding_lookup(embedding, real_inputs)
      masked_rnn_inputs = tf.nn.embedding_lookup(embedding, masked_inputs)

      state = initial_state
      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        rnn_inp = masked_rnn_inputs[:, t]
        rnn_out, state = cell(rnn_inp, state)
        hidden_states.append(rnn_out)
      final_masked_state = state
      hidden_states = tf.stack(hidden_states, axis=1)

      # Produce the RNN state had the model operated only
      # over real data.
      real_state = initial_state
      for t in xrange(FLAGS.sequence_length):
        tf.get_variable_scope().reuse_variables()

        # RNN.
        rnn_inp = real_rnn_inputs[:, t]
        rnn_out, real_state = cell(rnn_inp, real_state)
      final_state = real_state

  return (hidden_states, final_masked_state), initial_state, final_state


def gen_decoder(hparams,
                inputs,
                targets,
                targets_present,
                encoding_state,
                is_training,
                is_validating,
                reuse=None):
  """Define the Decoder graph. The Decoder will now impute tokens that
      have been masked from the input seqeunce.
  """
  gen_decoder_rnn_size = hparams.gen_rnn_size

  with tf.variable_scope('decoder', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.LayerNormBasicLSTMCell(
          gen_decoder_rnn_size, reuse=reuse)

    attn_cell = lstm_cell
    if FLAGS.zoneout_drop_prob > 0.0:

      def attn_cell():
        return zoneout.ZoneoutWrapper(
            lstm_cell(),
            zoneout_drop_prob=FLAGS.zoneout_drop_prob,
            is_training=is_training)

    cell_gen = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.gen_num_layers)],
        state_is_tuple=True)

    # Hidden encoder states.
    hidden_vector_encodings = encoding_state[0]

    # Carry forward the final state tuple from the encoder.
    # State tuples.
    state_gen = encoding_state[1]

    if FLAGS.attention_option is not None:
      (attention_keys, attention_values, _,
       attention_construct_fn) = attention_utils.prepare_attention(
           hidden_vector_encodings,
           FLAGS.attention_option,
           num_units=gen_decoder_rnn_size,
           reuse=reuse)

    with tf.variable_scope('rnn'):
      sequence, logits, log_probs = [], [], []
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, gen_decoder_rnn_size])
      softmax_w = tf.get_variable('softmax_w',
                                  [gen_decoder_rnn_size, FLAGS.vocab_size])
      softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

      rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)

      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        # Input to the Decoder.
        if t == 0:
          # Always provide the real input at t = 0.
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

        if FLAGS.attention_option is not None:
          rnn_out = attention_construct_fn(rnn_out, attention_keys,
                                           attention_values)
        #   # TODO(liamfedus): Assert not "monotonic" attention_type.
        #   # TODO(liamfedus): FLAGS.attention_type.
        #   context_state = revised_attention_utils._empty_state()
        #   rnn_out, context_state = attention_construct_fn(
        #       rnn_out, attention_keys, attention_values, context_state, t)
        logit = tf.matmul(rnn_out, softmax_w) + softmax_b

        # Output for Decoder.
        # If input is present:   Return real at t+1.
        # If input is not present:  Return fake for t+1.
        real = targets[:, t]

        categorical = tf.contrib.distributions.Categorical(logits=logit)
        fake = categorical.sample()
        log_prob = categorical.log_prob(fake)

        output = tf.where(targets_present[:, t], real, fake)

        # Add to lists.
        sequence.append(output)
        log_probs.append(log_prob)
        logits.append(logit)

  return (tf.stack(sequence, axis=1), tf.stack(logits, axis=1), tf.stack(
      log_probs, axis=1))


def generator(hparams,
              inputs,
              targets,
              targets_present,
              is_training,
              is_validating,
              reuse=None):
  """Define the Generator graph."""
  with tf.variable_scope('gen', reuse=reuse):
    encoder_states, initial_state, final_state = gen_encoder(
        hparams, inputs, targets_present, is_training=is_training, reuse=reuse)
    stacked_sequence, stacked_logits, stacked_log_probs = gen_decoder(
        hparams,
        inputs,
        targets,
        targets_present,
        encoder_states,
        is_training=is_training,
        is_validating=is_validating,
        reuse=reuse)
    return (stacked_sequence, stacked_logits, stacked_log_probs, initial_state,
            final_state)
