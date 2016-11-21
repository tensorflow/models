'''
Train, generate, and transform code for the SkipThoughtModel
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

import tensorflow as tf
import numpy as np

from model import SkipThoughtModel
import data_utils


tf.app.flags.DEFINE_float("learning_rate", 0.008, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor",
                          1, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer(
    "encoder_cell_size", 24, "The number of GRUs in an encoder cell.")
tf.app.flags.DEFINE_integer(
    "decoder_cell_size", 24, "The number of GRUs in a decoder cell.")
tf.app.flags.DEFINE_integer("word_embedding_size",
                            48, "The size of the word embeddings.")
tf.app.flags.DEFINE_integer(
    "max_epochs", 2000, "The total number of training epochs.")
tf.app.flags.DEFINE_integer("vocab_size", 200, "The vocabulary size.")
tf.app.flags.DEFINE_integer("max_sentence_len", 30,
                            "The max sentence length we will use.")
tf.app.flags.DEFINE_integer("batch_size",
                            2, "The number of sentences in our batch.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",
                            1000, "The number of steps per checkpoint.")

tf.app.flags.DEFINE_string("model_dir", "model", "Training directory.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory.")
tf.app.flags.DEFINE_string(
    "train_data_name", "books_tiny_p1.txt", "Name of the train data file.")
tf.app.flags.DEFINE_string("relatedness_regression_factors", "relatedness_regression_factors.csv",
                           "Name of the relatedness regression factors data file.")
tf.app.flags.DEFINE_string("relatedness_regression_targets", "relatedness_regression_targets.csv",
                           "Name of the relatedness regression targets data file.")
tf.app.flags.DEFINE_string("summary_dir", "summary", "Summary directory.")

FLAGS = tf.app.flags.FLAGS


def train():
    '''
    Trains the SkipThoughtModel based on hyperparams passed in a flags.
    '''
    train_path, vocab_path, train_ids_path = data_utils.prepare_skip_thought_data(
        FLAGS.data_dir, FLAGS.train_data_name, FLAGS.vocab_size)

    with tf.Session() as sess:

        model = SkipThoughtModel(FLAGS.vocab_size, max_sentence_len=FLAGS.max_sentence_len,
                                 learning_rate=FLAGS.learning_rate,
                                 batch_size=FLAGS.batch_size,
                                 learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                                 encoder_cell_size=FLAGS.encoder_cell_size,
                                 word_embedding_size=FLAGS.word_embedding_size,
                                 decoder_cell_size=FLAGS.decoder_cell_size,
                                 max_gradient_norm=FLAGS.max_gradient_norm,
                                 initial_decoder_state=None)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.initialize_all_variables())

        # add the summary writer
        if tf.gfile.Exists(FLAGS.summary_dir):
            tf.gfile.DeleteRecursively(FLAGS.summary_dir)
        tf.gfile.MakeDirs(FLAGS.summary_dir)
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)
        for epoch in range(FLAGS.max_epochs):
            run_epoch(model, sess, summary_writer, train_ids_path)


def generate(middle_sentence, forwards_sentence, backwards_sentence):
    '''
    Generates forwards and backwards sentences given a middle sentence.
    Args:
      middle_sentence: middle sentence (not tokenized)
      forwards_sentence: preceding sentence (not tokenized)
      backwards_sentence: following sentence (not tokenized)
    '''
    train_path, vocab_path, train_ids_path = data_utils.prepare_skip_thought_data(
        FLAGS.data_dir, FLAGS.train_data_name, FLAGS.vocab_size)

    with tf.Session() as sess:
        m = SkipThoughtModel(FLAGS.vocab_size, max_sentence_len=FLAGS.max_sentence_len,
                             batch_size=FLAGS.batch_size,
                             learning_rate=FLAGS.learning_rate,
                             learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                             encoder_cell_size=FLAGS.encoder_cell_size,
                             word_embedding_size=FLAGS.word_embedding_size,
                             decoder_cell_size=FLAGS.decoder_cell_size,
                             max_gradient_norm=FLAGS.max_gradient_norm,
                             initial_decoder_state=None)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            m.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No model found")
            return

        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        tokenized_middle_sentence = data_utils.sentence_to_token_ids(
            middle_sentence, vocab)
        tokenized_forwards_sentence = data_utils.sentence_to_token_ids(
            forwards_sentence, vocab)
        tokenized_backwards_sentence = data_utils.sentence_to_token_ids(
            " ".join(reversed(backwards_sentence.split())), vocab)

        forwards_batch_logits, backwards_batch_logits = m.step(sess, [m.forwards_batch_logits_tensor, m.backwards_batch_logits_tensor], *m.prep_data(
            [tokenized_middle_sentence], [tokenized_forwards_sentence], [tokenized_backwards_sentence]))

        forwards_logits = forwards_batch_logits[:, 0, :]
        backwards_logits = backwards_batch_logits[:, 0, :]
        print(forwards_logits)
        print(forwards_logits.shape)

        forwards_sentence = map(
            lambda x: rev_vocab[x], map(np.argmax, forwards_logits))
        backwards_sentence = map(
            lambda x: rev_vocab[x], map(np.argmax, backwards_logits))

        print("Generated Forwards Sentence")
        print(" ".join(forwards_sentence))
        print("Generated Backwards Sentence")
        print(" ".join(backwards_sentence))


def convert_relatedness():
    '''
    Converts a file of sentences based on most recent model.
    '''
    train_path, vocab_path, train_ids_path = data_utils.prepare_skip_thought_data(
        FLAGS.data_dir, FLAGS.train_data_name, FLAGS.vocab_size)

    print("Converting Relatedness Data")
    sentence_A_ids, sentence_B_ids, relatedness_scores = data_utils.prepare_relatedness_data(
        FLAGS.data_dir, vocab_path)

    with tf.Session() as sess:
        m = SkipThoughtModel(FLAGS.vocab_size, max_sentence_len=FLAGS.max_sentence_len,
                             batch_size=FLAGS.batch_size,
                             learning_rate=FLAGS.learning_rate,
                             learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                             encoder_cell_size=FLAGS.encoder_cell_size,
                             word_embedding_size=FLAGS.word_embedding_size,
                             decoder_cell_size=FLAGS.decoder_cell_size,
                             max_gradient_norm=FLAGS.max_gradient_norm,
                             initial_decoder_state=None)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            m.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No model found")
            return

        count = 0
        relatedness_regression_factors = []
        relatedness_regression_targets = []
        print("Begining Sentence Embedding")
        for i in xrange(0, len(relatedness_scores), FLAGS.batch_size):
            batch_sentence_A_ids = sentence_A_ids[i:i + FLAGS.batch_size]
            batch_sentence_B_ids = sentence_B_ids[i:i + FLAGS.batch_size]
            batch_relatedness_scores = relatedness_scores[
                i:i + FLAGS.batch_size]
            if len(batch_relatedness_scores) < FLAGS.batch_size:
                break

            sentence_A_embedding = np.array(m.step(sess, [m.encoder_state, tf.no_op()], *m.prep_data(batch_sentence_A_ids,
                                                                                                     batch_sentence_A_ids, batch_sentence_A_ids)))[0]

            sentence_B_embedding = np.array(m.step(sess, [m.encoder_state, tf.no_op()], *m.prep_data(batch_sentence_B_ids,
                                                                                                     batch_sentence_B_ids, batch_sentence_B_ids)))[0]

            factor = np.abs(sentence_A_embedding - sentence_B_embedding)
            factor = np.concatenate(
                [factor, (sentence_A_embedding * sentence_B_embedding)], axis=1)
            if len(relatedness_regression_factors) == 0:
                relatedness_regression_factors = factor
                relatedness_regression_targets = batch_relatedness_scores
                continue

            relatedness_regression_factors = np.concatenate(
                [relatedness_regression_factors, factor])
            relatedness_regression_targets = np.concatenate(
                [relatedness_regression_targets, batch_relatedness_scores])

            count += FLAGS.batch_size
            if count % (10 * FLAGS.batch_size) == 0:
                print("{} Sentences Embedded".format([count]))

        relatedness_factors_dir = os.path.join(
            FLAGS.data_dir, FLAGS.relatedness_regression_factors)
        relatedness_targets_dir = os.path.join(
            FLAGS.data_dir, FLAGS.relatedness_regression_targets)

        data_utils.save_np_array(
            relatedness_factors_dir, relatedness_regression_factors)

        data_utils.save_np_array(
            relatedness_targets_dir, relatedness_regression_targets)


def run_epoch(m, session, summary_writer, sentence_ids_path):
    '''
    Does one complete training run through of a dataset.
    Args:
      m: SkipThoughtModel
      session: tf.session
      summary_writer: a tf.summary_writer
      sentence_ids_path: the path to the tokenized data
    '''
    step_time, loss = 0.0, 0.0
    previous_losses = []
    perplexity = 0.0
    for step, (backwards_decoder_inputs, encoder_inputs, forwards_decoder_inputs) in enumerate(data_utils.sentence_iterator(sentence_ids_path, FLAGS.batch_size)):
        start_time = time.time()
        step_loss, _, step_summary = m.step(session, [m.cost, m.updates, m.merged_summaries], *m.prep_data(encoder_inputs,
                                                                                                           forwards_decoder_inputs, backwards_decoder_inputs))
        step_time += (time.time() - start_time)
        loss += step_loss

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if step % FLAGS.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(
                loss / FLAGS.steps_per_checkpoint) if loss / FLAGS.steps_per_checkpoint < 300 else loss / FLAGS.steps_per_checkpoint
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (m.global_step.eval(), m.learning_rate.eval(),
                             step_time / FLAGS.steps_per_checkpoint, perplexity))

            # Decrease learning rate if no improvement was seen over last 3
            # times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                session.run(m.learning_rate_decay_op)
            previous_losses.append(loss)

            # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.model_dir, "skipthought.ckpt")
            m.saver.save(session, checkpoint_path, global_step=m.global_step)

            summary_writer.add_summary(step_summary, m.global_step.eval())

            # zero timer and loss.
            step_time, loss = 0.0, 0.0

    print("Epoch ended with final loss at {}".format([perplexity]))

if __name__ == '__main__':
    train()

    # middle_sentence = "with clinicals looming to enable her to finish her \
    # nursing degree , she 'd known she would n't be able to work fulltime ."
    # forwards_sentence = "although she loved the freedom and independence of \
    # having her own apartment , there was no way she could afford it and daycare for mason ."
    # backwards_sentence = "now that she was back under their roof , they \
    # seemed to forget she was twenty-five , a mother , and not their little \
    # girl to boss around anymore ."
    # generate(forwards_sentence, middle_sentence, backwards_sentence)
