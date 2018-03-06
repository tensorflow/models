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

import collections
from six.moves import xrange
import tensorflow as tf

# NAS Code..
from nas_utils import configs
from nas_utils import custom_cell
from nas_utils import variational_dropout

FLAGS = tf.app.flags.FLAGS


def get_config():
  return configs.AlienConfig2()


LSTMTuple = collections.namedtuple('LSTMTuple', ['c', 'h'])


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
  tf.logging.info(
      'Undirectional generative model is not a useful model for this MaskGAN '
      'because future context is needed.  Use only for debugging purposes.')
  config = get_config()
  config.keep_prob = [hparams.gen_nas_keep_prob_0, hparams.gen_nas_keep_prob_1]
  configs.print_config(config)

  init_scale = config.init_scale
  initializer = tf.random_uniform_initializer(-init_scale, init_scale)

  with tf.variable_scope('gen', reuse=reuse, initializer=initializer):
    # Neural architecture search cell.
    cell = custom_cell.Alien(config.hidden_size)

    if is_training:
      [h2h_masks, _, _,
       output_mask] = variational_dropout.generate_variational_dropout_masks(
           hparams, config.keep_prob)
    else:
      output_mask = None

    cell_gen = custom_cell.GenericMultiRNNCell([cell] * config.num_layers)
    initial_state = cell_gen.zero_state(FLAGS.batch_size, tf.float32)

    with tf.variable_scope('rnn'):
      sequence, logits, log_probs = [], [], []
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])
      softmax_w = tf.matrix_transpose(embedding)
      softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

      rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)

      if is_training and FLAGS.keep_prob < 1:
        rnn_inputs = tf.nn.dropout(rnn_inputs, FLAGS.keep_prob)

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

        if is_training:
          state_gen = list(state_gen)
          for layer_num, per_layer_state in enumerate(state_gen):
            per_layer_state = LSTMTuple(
                per_layer_state[0], per_layer_state[1] * h2h_masks[layer_num])
            state_gen[layer_num] = per_layer_state

        # RNN.
        rnn_out, state_gen = cell_gen(rnn_inp, state_gen)

        if is_training:
          rnn_out = output_mask * rnn_out

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
  tf.logging.info(
      'Undirectional Discriminative model is not a useful model for this '
      'MaskGAN because future context is needed.  Use only for debugging '
      'purposes.')
  sequence = tf.cast(sequence, tf.int32)

  if FLAGS.dis_share_embedding:
    assert hparams.dis_rnn_size == hparams.gen_rnn_size, (
        'If you wish to share Discriminator/Generator embeddings, they must be'
        ' same dimension.')
    with tf.variable_scope('gen/rnn', reuse=True):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  config = get_config()
  config.keep_prob = [hparams.dis_nas_keep_prob_0, hparams.dis_nas_keep_prob_1]
  configs.print_config(config)

  with tf.variable_scope('dis', reuse=reuse):
    # Neural architecture search cell.
    cell = custom_cell.Alien(config.hidden_size)

    if is_training:
      [h2h_masks, _, _,
       output_mask] = variational_dropout.generate_variational_dropout_masks(
           hparams, config.keep_prob)
    else:
      output_mask = None

    cell_dis = custom_cell.GenericMultiRNNCell([cell] * config.num_layers)
    state_dis = cell_dis.zero_state(FLAGS.batch_size, tf.float32)

    with tf.variable_scope('rnn') as vs:
      predictions = []
      if not FLAGS.dis_share_embedding:
        embedding = tf.get_variable('embedding',
                                    [FLAGS.vocab_size, hparams.dis_rnn_size])

      rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

      if is_training and FLAGS.keep_prob < 1:
        rnn_inputs = tf.nn.dropout(rnn_inputs, FLAGS.keep_prob)

      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        rnn_in = rnn_inputs[:, t]

        if is_training:
          state_dis = list(state_dis)
          for layer_num, per_layer_state in enumerate(state_dis):
            per_layer_state = LSTMTuple(
                per_layer_state[0], per_layer_state[1] * h2h_masks[layer_num])
            state_dis[layer_num] = per_layer_state

        # RNN.
        rnn_out, state_dis = cell_dis(rnn_in, state_dis)

        if is_training:
          rnn_out = output_mask * rnn_out

        # Prediction is linear output for Discriminator.
        pred = tf.contrib.layers.linear(rnn_out, 1, scope=vs)

        predictions.append(pred)
  predictions = tf.stack(predictions, axis=1)
  return tf.squeeze(predictions, axis=2)
