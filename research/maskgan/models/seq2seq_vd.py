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

from six.moves import xrange
import tensorflow as tf

from models import attention_utils
from regularization import variational_dropout

FLAGS = tf.app.flags.FLAGS


def transform_input_with_is_missing_token(inputs, targets_present):
  """Transforms the inputs to have missing tokens when it's masked out.  The
  mask is for the targets, so therefore, to determine if an input at time t is
  masked, we have to check if the target at time t - 1 is masked out.

  e.g.
    inputs = [a, b, c, d]
    targets = [b, c, d, e]
    targets_present = [1, 0, 1, 0]

  which computes,
    inputs_present = [1, 1, 0, 1]

  and outputs,
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


# TODO(adai): IMDB labels placeholder to encoder.
def gen_encoder(hparams, inputs, targets_present, is_training, reuse=None):
  """Define the Encoder graph.

  Args:
    hparams:  Hyperparameters for the MaskGAN.
    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens
      up to, but not including, vocab_size.
    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with
      True representing the presence of the target.
    is_training:  Boolean indicating operational mode (train/inference).
    reuse (Optional):   Whether to reuse the variables.

  Returns:
    Tuple of (hidden_states, final_state).
  """
  # We will use the same variable from the decoder.
  if FLAGS.seq2seq_share_embedding:
    with tf.variable_scope('decoder/rnn'):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  with tf.variable_scope('encoder', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          hparams.gen_rnn_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and hparams.gen_vd_keep_prob < 1:

      def attn_cell():
        return variational_dropout.VariationalDropoutWrapper(
            lstm_cell(), FLAGS.batch_size, hparams.gen_rnn_size,
            hparams.gen_vd_keep_prob, hparams.gen_vd_keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.gen_num_layers)],
        state_is_tuple=True)

    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

    # Add a missing token for inputs not present.
    real_inputs = inputs
    masked_inputs = transform_input_with_is_missing_token(
        inputs, targets_present)

    with tf.variable_scope('rnn') as scope:
      hidden_states = []

      # Split the embedding into two parts so that we can load the PTB
      # weights into one part of the Variable.
      if not FLAGS.seq2seq_share_embedding:
        embedding = tf.get_variable('embedding',
                                    [FLAGS.vocab_size, hparams.gen_rnn_size])
      missing_embedding = tf.get_variable('missing_embedding',
                                          [1, hparams.gen_rnn_size])
      embedding = tf.concat([embedding, missing_embedding], axis=0)

      # TODO(adai): Perhaps append IMDB labels placeholder to input at
      # each time point.
      real_rnn_inputs = tf.nn.embedding_lookup(embedding, real_inputs)
      masked_rnn_inputs = tf.nn.embedding_lookup(embedding, masked_inputs)

      state = initial_state

      def make_mask(keep_prob, units):
        random_tensor = keep_prob
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        random_tensor += tf.random_uniform(
            tf.stack([FLAGS.batch_size, 1, units]))
        return tf.floor(random_tensor) / keep_prob

      if is_training:
        output_mask = make_mask(hparams.gen_vd_keep_prob, hparams.gen_rnn_size)

      hidden_states, state = tf.nn.dynamic_rnn(
          cell, masked_rnn_inputs, initial_state=state, scope=scope)
      if is_training:
        hidden_states *= output_mask

      final_masked_state = state

      # Produce the RNN state had the model operated only
      # over real data.
      real_state = initial_state
      _, real_state = tf.nn.dynamic_rnn(
          cell, real_rnn_inputs, initial_state=real_state, scope=scope)
      final_state = real_state

  return (hidden_states, final_masked_state), initial_state, final_state


# TODO(adai): IMDB labels placeholder to encoder.
def gen_encoder_cnn(hparams, inputs, targets_present, is_training, reuse=None):
  """Define the CNN Encoder graph."""
  del reuse
  sequence = transform_input_with_is_missing_token(inputs, targets_present)

  # TODO(liamfedus): Make this a hyperparameter.
  dis_filter_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

  # Keeping track of l2 regularization loss (optional)
  # l2_loss = tf.constant(0.0)

  with tf.variable_scope('encoder', reuse=True):
    with tf.variable_scope('rnn'):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  cnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

  # Create a convolution layer for each filter size
  conv_outputs = []
  for filter_size in dis_filter_sizes:
    with tf.variable_scope('conv-%s' % filter_size):
      # Convolution Layer
      filter_shape = [
          filter_size, hparams.gen_rnn_size, hparams.dis_num_filters
      ]
      W = tf.get_variable(
          name='W', initializer=tf.truncated_normal(filter_shape, stddev=0.1))
      b = tf.get_variable(
          name='b',
          initializer=tf.constant(0.1, shape=[hparams.dis_num_filters]))
      conv = tf.nn.conv1d(cnn_inputs, W, stride=1, padding='SAME', name='conv')

      # Apply nonlinearity
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

      conv_outputs.append(h)

  # Combine all the pooled features
  dis_num_filters_total = hparams.dis_num_filters * len(dis_filter_sizes)

  h_conv = tf.concat(conv_outputs, axis=2)
  h_conv_flat = tf.reshape(h_conv, [-1, dis_num_filters_total])

  # Add dropout
  if is_training:
    with tf.variable_scope('dropout'):
      h_conv_flat = tf.nn.dropout(h_conv_flat, hparams.gen_vd_keep_prob)

  # Final (unnormalized) scores and predictions
  with tf.variable_scope('output'):
    W = tf.get_variable(
        'W',
        shape=[dis_num_filters_total, hparams.gen_rnn_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
        name='b', initializer=tf.constant(0.1, shape=[hparams.gen_rnn_size]))
    # l2_loss += tf.nn.l2_loss(W)
    # l2_loss += tf.nn.l2_loss(b)
    predictions = tf.nn.xw_plus_b(h_conv_flat, W, b, name='predictions')
    predictions = tf.reshape(
        predictions,
        shape=[FLAGS.batch_size, FLAGS.sequence_length, hparams.gen_rnn_size])
  final_state = tf.reduce_mean(predictions, 1)
  return predictions, (final_state, final_state)


# TODO(adai): IMDB labels placeholder to decoder.
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

  targets = tf.Print(targets, [targets], message='targets', summarize=50)
  if FLAGS.seq2seq_share_embedding:
    with tf.variable_scope('decoder/rnn', reuse=True):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])

  with tf.variable_scope('decoder', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          gen_decoder_rnn_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and hparams.gen_vd_keep_prob < 1:

      def attn_cell():
        return variational_dropout.VariationalDropoutWrapper(
            lstm_cell(), FLAGS.batch_size, hparams.gen_rnn_size,
            hparams.gen_vd_keep_prob, hparams.gen_vd_keep_prob)

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

    def make_mask(keep_prob, units):
      random_tensor = keep_prob
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
      return tf.floor(random_tensor) / keep_prob

    if is_training:
      output_mask = make_mask(hparams.gen_vd_keep_prob, hparams.gen_rnn_size)

    with tf.variable_scope('rnn'):
      sequence, logits, log_probs = [], [], []

      if not FLAGS.seq2seq_share_embedding:
        embedding = tf.get_variable('embedding',
                                    [FLAGS.vocab_size, hparams.gen_rnn_size])
      softmax_w = tf.matrix_transpose(embedding)
      softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

      rnn_inputs = tf.nn.embedding_lookup(embedding, inputs)
      # TODO(adai): Perhaps append IMDB labels placeholder to input at
      # each time point.

      rnn_outs = []

      fake = None
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

          # While validating, the decoder should be operating in teacher
          # forcing regime.  Also, if we're just training with cross_entropy
          # use teacher forcing.
          if is_validating or FLAGS.gen_training_strategy == 'cross_entropy':
            rnn_inp = real_rnn_inp
          else:
            fake_rnn_inp = tf.nn.embedding_lookup(embedding, fake)
            rnn_inp = tf.where(targets_present[:, t - 1], real_rnn_inp,
                               fake_rnn_inp)

        # RNN.
        rnn_out, state_gen = cell_gen(rnn_inp, state_gen)

        if FLAGS.attention_option is not None:
          rnn_out = attention_construct_fn(rnn_out, attention_keys,
                                           attention_values)
        if is_training:
          rnn_out *= output_mask

        rnn_outs.append(rnn_out)
        if FLAGS.gen_training_strategy != 'cross_entropy':
          logit = tf.nn.bias_add(tf.matmul(rnn_out, softmax_w), softmax_b)

          # Output for Decoder.
          # If input is present:   Return real at t+1.
          # If input is not present:  Return fake for t+1.
          real = targets[:, t]

          categorical = tf.contrib.distributions.Categorical(logits=logit)
          if FLAGS.use_gen_mode:
            fake = categorical.mode()
          else:
            fake = categorical.sample()
          log_prob = categorical.log_prob(fake)
          output = tf.where(targets_present[:, t], real, fake)

        else:
          real = targets[:, t]
          logit = tf.zeros(tf.stack([FLAGS.batch_size, FLAGS.vocab_size]))
          log_prob = tf.zeros(tf.stack([FLAGS.batch_size]))
          output = real

        # Add to lists.
        sequence.append(output)
        log_probs.append(log_prob)
        logits.append(logit)

      if FLAGS.gen_training_strategy == 'cross_entropy':
        logits = tf.nn.bias_add(
            tf.matmul(
                tf.reshape(tf.stack(rnn_outs, 1), [-1, gen_decoder_rnn_size]),
                softmax_w), softmax_b)
        logits = tf.reshape(logits,
                            [-1, FLAGS.sequence_length, FLAGS.vocab_size])
      else:
        logits = tf.stack(logits, axis=1)

  return (tf.stack(sequence, axis=1), logits, tf.stack(log_probs, axis=1))


def dis_encoder(hparams, masked_inputs, is_training, reuse=None,
                embedding=None):
  """Define the Discriminator encoder.  Reads in the masked inputs for context
  and produces the hidden states of the encoder."""
  with tf.variable_scope('encoder', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          hparams.dis_rnn_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and hparams.dis_vd_keep_prob < 1:

      def attn_cell():
        return variational_dropout.VariationalDropoutWrapper(
            lstm_cell(), FLAGS.batch_size, hparams.dis_rnn_size,
            hparams.dis_vd_keep_prob, hparams.dis_vd_keep_prob)

    cell_dis = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.dis_num_layers)],
        state_is_tuple=True)

    state_dis = cell_dis.zero_state(FLAGS.batch_size, tf.float32)

    with tf.variable_scope('rnn'):
      hidden_states = []

      missing_embedding = tf.get_variable('missing_embedding',
                                          [1, hparams.dis_rnn_size])
      embedding = tf.concat([embedding, missing_embedding], axis=0)
      masked_rnn_inputs = tf.nn.embedding_lookup(embedding, masked_inputs)

      def make_mask(keep_prob, units):
        random_tensor = keep_prob
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
        return tf.floor(random_tensor) / keep_prob

      if is_training:
        output_mask = make_mask(hparams.dis_vd_keep_prob, hparams.dis_rnn_size)

      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        rnn_in = masked_rnn_inputs[:, t]
        rnn_out, state_dis = cell_dis(rnn_in, state_dis)
        if is_training:
          rnn_out *= output_mask
        hidden_states.append(rnn_out)
      final_state = state_dis

  return (tf.stack(hidden_states, axis=1), final_state)


def dis_decoder(hparams,
                sequence,
                encoding_state,
                is_training,
                reuse=None,
                embedding=None):
  """Define the Discriminator decoder.  Read in the sequence and predict
    at each time point."""
  sequence = tf.cast(sequence, tf.int32)

  with tf.variable_scope('decoder', reuse=reuse):

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          hparams.dis_rnn_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=reuse)

    attn_cell = lstm_cell
    if is_training and hparams.dis_vd_keep_prob < 1:

      def attn_cell():
        return variational_dropout.VariationalDropoutWrapper(
            lstm_cell(), FLAGS.batch_size, hparams.dis_rnn_size,
            hparams.dis_vd_keep_prob, hparams.dis_vd_keep_prob)

    cell_dis = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(hparams.dis_num_layers)],
        state_is_tuple=True)

    # Hidden encoder states.
    hidden_vector_encodings = encoding_state[0]

    # Carry forward the final state tuple from the encoder.
    # State tuples.
    state = encoding_state[1]

    if FLAGS.attention_option is not None:
      (attention_keys, attention_values, _,
       attention_construct_fn) = attention_utils.prepare_attention(
           hidden_vector_encodings,
           FLAGS.attention_option,
           num_units=hparams.dis_rnn_size,
           reuse=reuse)

    def make_mask(keep_prob, units):
      random_tensor = keep_prob
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      random_tensor += tf.random_uniform(tf.stack([FLAGS.batch_size, units]))
      return tf.floor(random_tensor) / keep_prob

    if is_training:
      output_mask = make_mask(hparams.dis_vd_keep_prob, hparams.dis_rnn_size)

    with tf.variable_scope('rnn') as vs:
      predictions = []

      rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

      for t in xrange(FLAGS.sequence_length):
        if t > 0:
          tf.get_variable_scope().reuse_variables()

        rnn_in = rnn_inputs[:, t]
        rnn_out, state = cell_dis(rnn_in, state)

        if FLAGS.attention_option is not None:
          rnn_out = attention_construct_fn(rnn_out, attention_keys,
                                           attention_values)
        if is_training:
          rnn_out *= output_mask

        # Prediction is linear output for Discriminator.
        pred = tf.contrib.layers.linear(rnn_out, 1, scope=vs)
        predictions.append(pred)

  predictions = tf.stack(predictions, axis=1)
  return tf.squeeze(predictions, axis=2)


def discriminator(hparams,
                  inputs,
                  targets_present,
                  sequence,
                  is_training,
                  reuse=None):
  """Define the Discriminator graph."""
  if FLAGS.dis_share_embedding:
    assert hparams.dis_rnn_size == hparams.gen_rnn_size, (
        'If you wish to share Discriminator/Generator embeddings, they must be'
        ' same dimension.')
    with tf.variable_scope('gen/decoder/rnn', reuse=True):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.gen_rnn_size])
  else:
    # Explicitly share the embedding.
    with tf.variable_scope('dis/decoder/rnn', reuse=reuse):
      embedding = tf.get_variable('embedding',
                                  [FLAGS.vocab_size, hparams.dis_rnn_size])

  # Mask the input sequence.
  masked_inputs = transform_input_with_is_missing_token(inputs, targets_present)

  # Confirm masking.
  masked_inputs = tf.Print(
      masked_inputs, [inputs, targets_present, masked_inputs, sequence],
      message='inputs, targets_present, masked_inputs, sequence',
      summarize=10)

  with tf.variable_scope('dis', reuse=reuse):
    encoder_states = dis_encoder(
        hparams,
        masked_inputs,
        is_training=is_training,
        reuse=reuse,
        embedding=embedding)
    predictions = dis_decoder(
        hparams,
        sequence,
        encoder_states,
        is_training=is_training,
        reuse=reuse,
        embedding=embedding)

  # if FLAGS.baseline_method == 'critic':
  #   with tf.variable_scope('critic', reuse=reuse) as critic_scope:
  #     values = tf.contrib.layers.linear(rnn_outs, 1, scope=critic_scope)
  #     values = tf.squeeze(values, axis=2)
  # else:
  #   values = None

  return predictions


# TODO(adai): IMDB labels placeholder to encoder/decoder.
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
            final_state, encoder_states)
