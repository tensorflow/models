'''
Specifies the SkipThoughtModel
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from gru_with_context import GRUCellWithContext
from data_utils import PAD_ID


class SkipThoughtModel(object):

    def __init__(self, vocab_size, max_sentence_len=30, batch_size=32, learning_rate=0.004, learning_rate_decay_factor=0.99, encoder_cell_size=2400, word_embedding_size=620,
                 decoder_cell_size=2400, max_gradient_norm=5.0, initial_decoder_state=None):
        """Create the model.
        """
        self.max_sentence_len = max_sentence_len
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)

        self.encoder_inputs = tf.placeholder(
            tf.int32, shape=[max_sentence_len, batch_size])
        self.forwards_decoder_inputs = tf.placeholder(
            tf.int32, shape=[max_sentence_len, batch_size])
        self.backwards_decoder_inputs = tf.placeholder(
            tf.int32, shape=[max_sentence_len, batch_size])

        self.encoder_sequence_length = tf.placeholder(
            tf.int32, shape=[batch_size])

        self.forwards_decoder_weights = tf.placeholder(
            tf.float32, shape=[max_sentence_len - 1, batch_size])
        self.backwards_decoder_weights = tf.placeholder(
            tf.float32, shape=[max_sentence_len - 1, batch_size])

        # this will get the embedding var we are using for words
        embedding = tf.get_variable("embedding", [vocab_size, word_embedding_size])

        # get the embeddings
        encoder_emb_inp = tf.nn.embedding_lookup(
            embedding, self.encoder_inputs)
        forwards_decoder_emb_inp = tf.nn.embedding_lookup(
            embedding, self.forwards_decoder_inputs)
        backwards_decoder_emb_inp = tf.nn.embedding_lookup(
            embedding, self.backwards_decoder_inputs)

        # Encoder.
        encoder_cell = tf.nn.rnn_cell.GRUCell(encoder_cell_size)

        _, self.encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,
                                                  dtype=tf.float32, sequence_length=self.encoder_sequence_length, time_major=True, initial_state=encoder_cell.zero_state(self.batch_size, tf.float32))

        decoder_cell = GRUCellWithContext(
            decoder_cell_size, self.encoder_state)

        if initial_decoder_state is None:
            initial_decoder_state = decoder_cell.zero_state(
                self.batch_size, tf.float32)

        # list of [batch_size x output_size]
        forwards_batch_outputs, _ = tf.nn.seq2seq.rnn_decoder(tf.unpack(
            forwards_decoder_emb_inp), initial_decoder_state, decoder_cell, scope="decoder_cell_forwards")

        # TODO add sampled softmax for fast training
        self.forwards_batch_logits = []
        forwards_softmax_w = tf.get_variable(
            "forwards_softmax_w", [decoder_cell_size, vocab_size])
        forwards_softmax_b = tf.get_variable(
            "forwards_softmax_b", [vocab_size])
        for output in forwards_batch_outputs:
            self.forwards_batch_logits.append(
                tf.matmul(output, forwards_softmax_w) + forwards_softmax_b)

        self.forwards_batch_logits_tensor = tf.pack(self.forwards_batch_logits)
        forwards_cost = tf.nn.seq2seq.sequence_loss_by_example(
            self.forwards_batch_logits[:-1],
            tf.unpack(self.forwards_decoder_inputs)[1:],
            tf.unpack(self.forwards_decoder_weights))

        backwards_batch_outputs, _ = tf.nn.seq2seq.rnn_decoder(tf.unpack(
            backwards_decoder_emb_inp), initial_decoder_state, decoder_cell, scope="decoder_cell_backwards")

        self.backwards_batch_logits = []
        backwards_softmax_w = tf.get_variable(
            "backwards_softmax_w", [decoder_cell_size, vocab_size])
        backwards_softmax_b = tf.get_variable(
            "backwards_softmax_b", [vocab_size])
        for output in backwards_batch_outputs:
            self.backwards_batch_logits.append(
                tf.matmul(output, backwards_softmax_w) + backwards_softmax_b)

        self.backwards_batch_logits_tensor = tf.pack(self.backwards_batch_logits)
        backwards_cost = tf.nn.seq2seq.sequence_loss_by_example(
            self.backwards_batch_logits[:-1],
            tf.unpack(self.backwards_decoder_inputs)[1:],
            tf.unpack(self.backwards_decoder_weights))

        self.cost = tf.reduce_sum(forwards_cost) + \
            tf.reduce_sum(backwards_cost)

        opt = tf.train.AdamOptimizer(self.learning_rate)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost, params)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      max_gradient_norm)

        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # add in tensorboard summaries
        tf.scalar_summary("learning_rate", self.learning_rate)
        tf.scalar_summary("forwards_cost", tf.reduce_sum(forwards_cost))
        tf.scalar_summary("backwards_cost", tf.reduce_sum(backwards_cost))

        # Merge all the summaries and write them out to /tmp/mnist_logs (by
        # default)
        self.merged_summaries = tf.merge_all_summaries()

        self.saver = tf.train.Saver(tf.all_variables())

    def prep_data(self, encoder_inputs, forwards_decoder_inputs, backwards_decoder_inputs):
        '''
        Given a set of sentences transform them into a batch of valid inputs.
        Args:
          encoder_inputs: middle sentences (tokenized)
          forwards_decoder_inputs: preceding sentences (tokenized)
          backwards_decoder_inputs: following sentences (tokenized)
        '''
        final_encoder_inputs = np.zeros(
            (self.batch_size, self.max_sentence_len))
        final_forwards_decoder_inputs = np.zeros(
            (self.batch_size, self.max_sentence_len))
        final_backwards_decoder_inputs = np.zeros(
            (self.batch_size, self.max_sentence_len))
        final_encoder_sequence_lengths = np.zeros((self.batch_size))
        final_backwards_decoder_weights = np.zeros(
            (self.batch_size, self.max_sentence_len - 1))
        final_forwards_decoder_weights = np.zeros(
            (self.batch_size, self.max_sentence_len - 1))

        for i, (encoder_input, forwards_decoder_input, backwards_decoder_input) in enumerate(zip(encoder_inputs, forwards_decoder_inputs, backwards_decoder_inputs)):
            encoder_input, forwards_decoder_input, backwards_decoder_input, forwards_decoder_weight, backwards_decoder_weight, encoder_sequence_length = self.prep_datum(
                encoder_input, forwards_decoder_input, backwards_decoder_input)
            final_encoder_inputs[i] = encoder_input
            final_forwards_decoder_inputs[i] = forwards_decoder_input
            final_backwards_decoder_inputs[i] = backwards_decoder_input
            final_encoder_sequence_lengths[i] = encoder_sequence_length
            final_backwards_decoder_weights[i] = backwards_decoder_weight
            final_forwards_decoder_weights[i] = forwards_decoder_weight

        return final_encoder_inputs, final_forwards_decoder_inputs, final_backwards_decoder_inputs, final_backwards_decoder_weights, final_forwards_decoder_weights, final_encoder_sequence_lengths

    def prep_datum(self, encoder_input, forwards_decoder_input, backwards_decoder_input):
        '''
        Given a single sentence transform it into a valid input.
        Args:
          encoder_input: middle sentence (tokenized)
          forwards_decoder_input: preceding sentence (tokenized)
          backwards_decoder_input: following sentence (tokenized)
        '''
        encoder_sequence_length = min(
            len(encoder_input), self.max_sentence_len)

        # create the weights
        forwards_decoder_weight = np.ones(self.max_sentence_len - 1)
        forwards_decoder_weight[len(forwards_decoder_input) - 1:] = 0
        backwards_decoder_weight = np.ones(self.max_sentence_len - 1)
        backwards_decoder_weight[len(backwards_decoder_input) - 1:] = 0

        # pad the inputs
        encoder_input = encoder_input[:self.max_sentence_len] + (
            self.max_sentence_len - len(encoder_input)) * [PAD_ID]
        forwards_decoder_input = forwards_decoder_input[:self.max_sentence_len] + (
            self.max_sentence_len - len(forwards_decoder_input)) * [PAD_ID]
        backwards_decoder_input = backwards_decoder_input[:self.max_sentence_len] + (
            self.max_sentence_len - len(backwards_decoder_input)) * [PAD_ID]

        return encoder_input, forwards_decoder_input, backwards_decoder_input, forwards_decoder_weight, backwards_decoder_weight, encoder_sequence_length

    def step(self, session, results, encoder_input, forwards_decoder_input, backwards_decoder_input, forwards_decoder_weight, backwards_decoder_weight, encoder_sequence_length):
        '''
        Feed valid inputs into the model.
        Args:
          session: tf.session
          results: properties of a SkipThoughtModel that you would like to return
          other: please feed *m.prep_data(...)
        '''
        return session.run(results, feed_dict={
            self.encoder_inputs: encoder_input.T,
            self.forwards_decoder_inputs: forwards_decoder_input.T,
            self.backwards_decoder_inputs: backwards_decoder_input.T,
            self.forwards_decoder_weights: forwards_decoder_weight.T,
            self.backwards_decoder_weights: backwards_decoder_weight.T,
            self.encoder_sequence_length: encoder_sequence_length})
