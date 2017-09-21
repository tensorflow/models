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

import tensorflow as tf
from util.nn import fc_layer as fc


def _get_lstm_cell(num_layers, lstm_dim):
  cell_list = [
      tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=True)
      for _ in range(num_layers)
  ]
  cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
  return cell


class AttentionSeq2Seq:

  def __init__(self,
               config,
               text_seq_batch,
               seq_length_batch,
               num_vocab_txt,
               num_vocab_nmn,
               EOS_token,
               decoder_sampling,
               embedding_mat,
               use_gt_layout=None,
               gt_layout_batch=None,
               scope='encoder_decoder',
               reuse=None):
    self.T_decoder = config.T_decoder
    self.encoder_num_vocab = num_vocab_txt
    self.encoder_embed_dim = config.embed_dim_txt
    self.decoder_num_vocab = num_vocab_nmn
    self.decoder_embed_dim = config.embed_dim_nmn
    self.lstm_dim = config.lstm_dim
    self.num_layers = config.num_layers
    self.EOS_token = EOS_token
    self.decoder_sampling = decoder_sampling
    self.embedding_mat = embedding_mat

    with tf.variable_scope(scope, reuse=reuse):
      self._build_encoder(text_seq_batch, seq_length_batch)
      self._build_decoder(use_gt_layout, gt_layout_batch)

  def _build_encoder(self,
                     text_seq_batch,
                     seq_length_batch,
                     scope='encoder',
                     reuse=None):
    lstm_dim = self.lstm_dim
    num_layers = self.num_layers

    with tf.variable_scope(scope, reuse=reuse):
      T = tf.shape(text_seq_batch)[0]
      N = tf.shape(text_seq_batch)[1]
      self.T_encoder = T
      self.N = N

      # text_seq has shape [T, N] and embedded_seq has shape [T, N, D]
      embedded_seq = tf.nn.embedding_lookup(self.embedding_mat, text_seq_batch)
      self.embedded_input_seq = embedded_seq

      # The RNN
      cell = _get_lstm_cell(num_layers, lstm_dim)

      # encoder_outputs has shape [T, N, lstm_dim]
      encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
          cell,
          embedded_seq,
          seq_length_batch,
          dtype=tf.float32,
          time_major=True,
          scope='lstm')
      self.encoder_outputs = encoder_outputs
      self.encoder_states = encoder_states

      # transform the encoder outputs for further attention alignments
      # encoder_outputs_flat has shape [T, N, lstm_dim]
      encoder_h_transformed = fc(
          'encoder_h_transform',
          tf.reshape(encoder_outputs, [-1, lstm_dim]),
          output_dim=lstm_dim)
      encoder_h_transformed = tf.reshape(encoder_h_transformed,
                                         [T, N, lstm_dim])
      self.encoder_h_transformed = encoder_h_transformed

      # seq_not_finished is a shape [T, N, 1] tensor,
      # where seq_not_finished[t, n]
      # is 1 iff sequence n is not finished at time t, and 0 otherwise
      seq_not_finished = tf.less(
          tf.range(T)[:, tf.newaxis, tf.newaxis],
          seq_length_batch[:, tf.newaxis])
      seq_not_finished = tf.cast(seq_not_finished, tf.float32)
      self.seq_not_finished = seq_not_finished

  def _build_decoder(self,
                     use_gt_layout,
                     gt_layout_batch,
                     scope='decoder',
                     reuse=None):
    # The main difference from before is that the decoders now takes another
    # input (the attention) when computing the next step
    # T_max is the maximum length of decoded sequence (including <eos>)
    #
    # This function is for decoding only. It performs greedy search or sampling.
    # the first input is <go> (its embedding vector) and the subsequent inputs
    # are the outputs from previous time step
    # num_vocab does not include <go>
    #
    # use_gt_layout is None or a bool tensor, and gt_layout_batch is a tensor
    # with shape [T_max, N].
    # If use_gt_layout is not None, then when use_gt_layout is true, predict
    # exactly the tokens in gt_layout_batch, regardless of actual probability.
    # Otherwise, if sampling is True, sample from the token probability
    # If sampling is False, do greedy decoding (beam size 1)
    N = self.N
    encoder_states = self.encoder_states
    T_max = self.T_decoder
    lstm_dim = self.lstm_dim
    num_layers = self.num_layers
    EOS_token = self.EOS_token
    sampling = self.decoder_sampling

    with tf.variable_scope(scope, reuse=reuse):
      embedding_mat = tf.get_variable(
          'embedding_mat', [self.decoder_num_vocab, self.decoder_embed_dim])
      # we use a separate embedding for <go>, as it is only used in the
      # beginning of the sequence
      go_embedding = tf.get_variable('go_embedding',
                                     [1, self.decoder_embed_dim])

      with tf.variable_scope('att_prediction'):
        v = tf.get_variable('v', [lstm_dim])
        W_a = tf.get_variable(
            'weights', [lstm_dim, lstm_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable(
            'biases', lstm_dim, initializer=tf.constant_initializer(0.))

      # The parameters to predict the next token
      with tf.variable_scope('token_prediction'):
        W_y = tf.get_variable(
            'weights', [lstm_dim * 2, self.decoder_num_vocab],
            initializer=tf.contrib.layers.xavier_initializer())
        b_y = tf.get_variable(
            'biases',
            self.decoder_num_vocab,
            initializer=tf.constant_initializer(0.))

      # Attentional decoding
      # Loop function is called at time t BEFORE the cell execution at time t,
      # and its next_input is used as the input at time t (not t+1)
      # c.f. https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn
      mask_range = tf.reshape(
          tf.range(self.decoder_num_vocab, dtype=tf.int32), [1, -1])
      all_eos_pred = EOS_token * tf.ones([N], tf.int32)
      all_one_prob = tf.ones([N], tf.float32)
      all_zero_entropy = tf.zeros([N], tf.float32)
      if use_gt_layout is not None:
        gt_layout_mult = tf.cast(use_gt_layout, tf.int32)
        pred_layout_mult = 1 - gt_layout_mult

      def loop_fn(time, cell_output, cell_state, loop_state):
        if cell_output is None:  # time == 0
          next_cell_state = encoder_states
          next_input = tf.tile(go_embedding, [N, 1])
        else:  # time > 0
          next_cell_state = cell_state

          # compute the attention map over the input sequence
          # a_raw has shape [T, N, 1]
          att_raw = tf.reduce_sum(
              tf.tanh(
                  tf.nn.xw_plus_b(cell_output, W_a, b_a) +
                  self.encoder_h_transformed) * v,
              axis=2,
              keep_dims=True)
          # softmax along the first dimension (T) over not finished examples
          # att has shape [T, N, 1]
          att = tf.nn.softmax(att_raw, dim=0) * self.seq_not_finished
          att = att / tf.reduce_sum(att, axis=0, keep_dims=True)
          # d has shape [N, lstm_dim]
          d2 = tf.reduce_sum(att * self.encoder_outputs, axis=0)

          # token_scores has shape [N, num_vocab]
          token_scores = tf.nn.xw_plus_b(
              tf.concat([cell_output, d2], axis=1), W_y, b_y)
          # predict the next token (behavior depending on parameters)
          if sampling:
            # predicted_token has shape [N]
            logits = token_scores
            predicted_token = tf.cast(
                tf.reshape(tf.multinomial(token_scores, 1), [-1]), tf.int32)
          else:
            # predicted_token has shape [N]
            predicted_token = tf.cast(tf.argmax(token_scores, 1), tf.int32)
          if use_gt_layout is not None:
            predicted_token = (gt_layout_batch[time - 1] * gt_layout_mult +
                               predicted_token * pred_layout_mult)

          # token_prob has shape [N], the probability of the predicted token
          # although token_prob is not needed for predicting the next token
          # it is needed in output (for policy gradient training)
          # [N, num_vocab]
          # mask has shape [N, num_vocab]
          mask = tf.equal(mask_range, tf.reshape(predicted_token, [-1, 1]))
          all_token_probs = tf.nn.softmax(token_scores)
          token_prob = tf.reduce_sum(
              all_token_probs * tf.cast(mask, tf.float32), axis=1)
          neg_entropy = tf.reduce_sum(
              all_token_probs * tf.log(all_token_probs), axis=1)

          # is_eos_predicted is a [N] bool tensor, indicating whether
          # <eos> has already been predicted previously in each sequence
          is_eos_predicted = loop_state[2]
          predicted_token_old = predicted_token
          # if <eos> has already been predicted, now predict <eos> with
          # prob 1
          predicted_token = tf.where(is_eos_predicted, all_eos_pred,
                                     predicted_token)
          token_prob = tf.where(is_eos_predicted, all_one_prob, token_prob)
          neg_entropy = tf.where(is_eos_predicted, all_zero_entropy,
                                 neg_entropy)
          is_eos_predicted = tf.logical_or(is_eos_predicted,
                                           tf.equal(predicted_token_old,
                                                    EOS_token))

          # the prediction is from the cell output of the last step
          # timestep (t-1), feed it as input into timestep t
          next_input = tf.nn.embedding_lookup(embedding_mat, predicted_token)

        elements_finished = tf.greater_equal(time, T_max)

        # loop_state is a 5-tuple, representing
        #   1) the predicted_tokens
        #   2) the prob of predicted_tokens
        #   3) whether <eos> has already been predicted
        #   4) the negative entropy of policy (accumulated across timesteps)
        #   5) the attention
        if loop_state is None:  # time == 0
          # Write the predicted token into the output
          predicted_token_array = tf.TensorArray(
              dtype=tf.int32, size=T_max, infer_shape=False)
          token_prob_array = tf.TensorArray(
              dtype=tf.float32, size=T_max, infer_shape=False)
          att_array = tf.TensorArray(
              dtype=tf.float32, size=T_max, infer_shape=False)
          next_loop_state = (predicted_token_array, token_prob_array, tf.zeros(
              [N], dtype=tf.bool), tf.zeros([N], dtype=tf.float32), att_array)
        else:  # time > 0
          t_write = time - 1
          next_loop_state = (
              loop_state[0].write(t_write, predicted_token),
              loop_state[1].write(t_write, token_prob),
              is_eos_predicted,
              loop_state[3] + neg_entropy,
              loop_state[4].write(t_write, att))
        return (elements_finished, next_input, next_cell_state, cell_output,
                next_loop_state)

      # The RNN
      cell = _get_lstm_cell(num_layers, lstm_dim)
      _, _, decodes_ta = tf.nn.raw_rnn(cell, loop_fn, scope='lstm')
      predicted_tokens = decodes_ta[0].stack()
      token_probs = decodes_ta[1].stack()
      neg_entropy = decodes_ta[3]
      # atts has shape [T_decoder, T_encoder, N, 1]
      atts = decodes_ta[4].stack()
      self.atts = atts
      # word_vec has shape [T_decoder, N, D]
      word_vecs = tf.reduce_sum(atts * self.embedded_input_seq, axis=1)

      predicted_tokens.set_shape([None, None])
      token_probs.set_shape([None, None])
      neg_entropy.set_shape([None])
      word_vecs.set_shape([None, None, self.encoder_embed_dim])

      self.predicted_tokens = predicted_tokens
      self.token_probs = token_probs
      self.neg_entropy = neg_entropy
      self.word_vecs = word_vecs
