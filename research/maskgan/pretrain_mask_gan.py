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

"""Pretraining functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

import tensorflow as tf

from data import imdb_loader
from data import ptb_loader

# Data.
from model_utils import model_utils
from models import evaluation_utils

tf.app.flags.DEFINE_integer(
    'gen_pretrain_steps', None,
    'The number of steps to pretrain the generator with cross entropy loss.')
tf.app.flags.DEFINE_integer(
    'dis_pretrain_steps', None,
    'The number of steps to pretrain the discriminator.')

FLAGS = tf.app.flags.FLAGS


def pretrain_generator(sv, sess, model, data, log, id_to_word,
                       data_ngram_counts, is_chief):
  """Pretrain the generator with classic language modeling training."""
  print('\nPretraining generator for %d steps.' % FLAGS.gen_pretrain_steps)
  log.write(
      '\nPretraining generator for %d steps.\n' % FLAGS.gen_pretrain_steps)

  is_pretraining = True

  while is_pretraining:

    costs = 0.
    iters = 0
    if FLAGS.data_set == 'ptb':
      iterator = ptb_loader.ptb_iterator(data, FLAGS.batch_size,
                                         FLAGS.sequence_length,
                                         FLAGS.epoch_size_override)
    elif FLAGS.data_set == 'imdb':
      iterator = imdb_loader.imdb_iterator(data, FLAGS.batch_size,
                                           FLAGS.sequence_length)

    for x, y, _ in iterator:

      # For pretraining with cross entropy loss, we have all tokens in the
      # forward sequence present (all True).
      model_utils.assign_percent_real(sess, model.percent_real_update,
                                      model.new_rate, 1.0)
      p = np.ones(shape=[FLAGS.batch_size, FLAGS.sequence_length], dtype=bool)

      pretrain_feed = {model.inputs: x, model.targets: y, model.present: p}

      [losses, cost_eval, _, step] = sess.run(
          [
              model.fake_cross_entropy_losses, model.avg_log_perplexity,
              model.gen_pretrain_op, model.global_step
          ],
          feed_dict=pretrain_feed)

      costs += cost_eval
      iters += FLAGS.sequence_length

      # Calulate rolling perplexity.
      perplexity = np.exp(costs / iters)

      # Summaries.
      if is_chief and step % FLAGS.summaries_every == 0:
        # Graph summaries.
        summary_str = sess.run(
            model.merge_summaries_op, feed_dict=pretrain_feed)
        sv.SummaryComputed(sess, summary_str)

        # Additional summary.
        for n, data_ngram_count in data_ngram_counts.iteritems():
          avg_percent_captured = evaluation_utils.sequence_ngram_evaluation(
              sess, model.fake_sequence, log, pretrain_feed, data_ngram_count,
              int(n))
          summary_percent_str = tf.Summary(value=[
              tf.Summary.Value(
                  tag='general/%s-grams_percent_correct' % n,
                  simple_value=avg_percent_captured)
          ])
          sv.SummaryComputed(sess, summary_percent_str, global_step=step)

        summary_perplexity_str = tf.Summary(value=[
            tf.Summary.Value(tag='general/perplexity', simple_value=perplexity)
        ])
        sv.SummaryComputed(sess, summary_perplexity_str, global_step=step)

      # Printing and logging
      if is_chief and step % FLAGS.print_every == 0:
        print('global_step: %d' % step)
        print(' generator loss: %.3f' % np.mean(losses))
        print(' perplexity: %.3f' % perplexity)
        log.write('global_step: %d\n' % step)
        log.write(' generator loss: %.3f\n' % np.mean(losses))
        log.write(' perplexity: %.3f\n' % perplexity)

        for n, data_ngram_count in data_ngram_counts.iteritems():
          avg_percent_captured = evaluation_utils.sequence_ngram_evaluation(
              sess, model.fake_sequence, log, pretrain_feed, data_ngram_count,
              int(n))
          print(' percent of %s-grams captured: %.3f.\n' %
                (n, avg_percent_captured))
          log.write(' percent of %s-grams captured: %.3f.\n\n' %
                    (n, avg_percent_captured))

        evaluation_utils.generate_logs(sess, model, log, id_to_word,
                                       pretrain_feed)

      if step >= FLAGS.gen_pretrain_steps:
        is_pretraining = False
        break
  return


def pretrain_discriminator(sv, sess, model, data, log, id_to_word,
                           data_ngram_counts, is_chief):
  print('\nPretraining discriminator for %d steps.' % FLAGS.dis_pretrain_steps)
  log.write(
      '\nPretraining discriminator for %d steps.\n' % FLAGS.dis_pretrain_steps)

  is_pretraining = True

  while is_pretraining:

    cumulative_costs = 0.
    iters = 0
    if FLAGS.data_set == 'ptb':
      iterator = ptb_loader.ptb_iterator(data, FLAGS.batch_size,
                                         FLAGS.sequence_length,
                                         FLAGS.epoch_size_override)
    elif FLAGS.data_set == 'imdb':
      iterator = imdb_loader.imdb_iterator(data, FLAGS.batch_size,
                                           FLAGS.sequence_length)

    for x, y, _ in iterator:
      is_present_rate = FLAGS.is_present_rate
      # is_present_rate = np.random.uniform(low=0.0, high=1.0)
      model_utils.assign_percent_real(sess, model.percent_real_update,
                                      model.new_rate, is_present_rate)
      # Randomly mask out tokens.
      p = model_utils.generate_mask()

      pretrain_feed = {model.inputs: x, model.targets: y, model.present: p}

      [_, dis_loss_eval, gen_log_perplexity_eval, step] = sess.run(
          [
              model.dis_pretrain_op, model.dis_loss, model.avg_log_perplexity,
              model.global_step
          ],
          feed_dict=pretrain_feed)

      cumulative_costs += gen_log_perplexity_eval
      iters += 1

      # Calulate rolling perplexity.
      perplexity = np.exp(cumulative_costs / iters)

      # Summaries.
      if is_chief and step % FLAGS.summaries_every == 0:
        # Graph summaries.
        summary_str = sess.run(
            model.merge_summaries_op, feed_dict=pretrain_feed)
        sv.SummaryComputed(sess, summary_str)

        # Additional summary.
        for n, data_ngram_count in data_ngram_counts.iteritems():
          avg_percent_captured = evaluation_utils.sequence_ngram_evaluation(
              sess, model.fake_sequence, log, pretrain_feed, data_ngram_count,
              int(n))
          summary_percent_str = tf.Summary(value=[
              tf.Summary.Value(
                  tag='general/%s-grams_percent_correct' % n,
                  simple_value=avg_percent_captured)
          ])
          sv.SummaryComputed(sess, summary_percent_str, global_step=step)

        summary_perplexity_str = tf.Summary(value=[
            tf.Summary.Value(tag='general/perplexity', simple_value=perplexity)
        ])
        sv.SummaryComputed(sess, summary_perplexity_str, global_step=step)

      # Printing and logging
      if is_chief and step % FLAGS.print_every == 0:
        print('global_step: %d' % step)
        print(' discriminator loss: %.3f' % dis_loss_eval)
        print(' perplexity: %.3f' % perplexity)
        log.write('global_step: %d\n' % step)
        log.write(' discriminator loss: %.3f\n' % dis_loss_eval)
        log.write(' perplexity: %.3f\n' % perplexity)

        for n, data_ngram_count in data_ngram_counts.iteritems():
          avg_percent_captured = evaluation_utils.sequence_ngram_evaluation(
              sess, model.fake_sequence, log, pretrain_feed, data_ngram_count,
              int(n))
          print(' percent of %s-grams captured: %.3f.\n' %
                (n, avg_percent_captured))
          log.write(' percent of %s-grams captured: %.3f.\n\n' %
                    (n, avg_percent_captured))

        evaluation_utils.generate_logs(sess, model, log, id_to_word,
                                       pretrain_feed)

      if step >= FLAGS.dis_pretrain_steps + int(FLAGS.gen_pretrain_steps or 0):
        is_pretraining = False
        break
  return
