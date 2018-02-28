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

"""Evaluation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
# Dependency imports
import numpy as np
from scipy.special import expit

import tensorflow as tf

from model_utils import helper
from model_utils import n_gram

FLAGS = tf.app.flags.FLAGS


def print_and_log_losses(log, step, is_present_rate, avg_dis_loss,
                         avg_gen_loss):
  """Prints and logs losses to the log file.

  Args:
    log: GFile for logs.
    step: Global step.
    is_present_rate: Current masking rate.
    avg_dis_loss: List of Discriminator losses.
    avg_gen_loss: List of Generator losses.
  """
  print('global_step: %d' % step)
  print(' is_present_rate: %.3f' % is_present_rate)
  print(' D train loss: %.5f' % np.mean(avg_dis_loss))
  print(' G train loss: %.5f' % np.mean(avg_gen_loss))
  log.write('\nglobal_step: %d\n' % step)
  log.write((' is_present_rate: %.3f\n' % is_present_rate))
  log.write(' D train loss: %.5f\n' % np.mean(avg_dis_loss))
  log.write(' G train loss: %.5f\n' % np.mean(avg_gen_loss))


def print_and_log(log, id_to_word, sequence_eval, max_num_to_print=5):
  """Helper function for printing and logging evaluated sequences."""
  indices_arr = np.asarray(sequence_eval)
  samples = helper.convert_to_human_readable(id_to_word, indices_arr,
                                             max_num_to_print)

  for i, sample in enumerate(samples):
    print('Sample', i, '. ', sample)
    log.write('\nSample ' + str(i) + '. ' + sample)
  log.write('\n')
  print('\n')
  log.flush()
  return samples


def zip_seq_pred_crossent(id_to_word, sequences, predictions, cross_entropy):
  """Zip together the sequences, predictions, cross entropy."""
  indices = np.asarray(sequences)

  batch_of_metrics = []

  for ind_batch, pred_batch, crossent_batch in zip(indices, predictions,
                                                   cross_entropy):
    metrics = []

    for index, pred, crossent in zip(ind_batch, pred_batch, crossent_batch):
      metrics.append([str(id_to_word[index]), pred, crossent])

    batch_of_metrics.append(metrics)
  return batch_of_metrics


def zip_metrics(indices, *args):
  """Zip together the indices matrices with the provided metrics matrices."""
  batch_of_metrics = []
  for metrics_batch in zip(indices, *args):

    metrics = []
    for m in zip(*metrics_batch):
      metrics.append(m)
    batch_of_metrics.append(metrics)
  return batch_of_metrics


def print_formatted(present, id_to_word, log, batch_of_tuples):
  """Print and log metrics."""
  num_cols = len(batch_of_tuples[0][0])
  repeat_float_format = '{:<12.3f} '
  repeat_str_format = '{:<13}'

  format_str = ''.join(
      ['[{:<1}]  {:<20}',
       str(repeat_float_format * (num_cols - 1))])

  # TODO(liamfedus): Generalize the logging. This is sloppy.
  header_format_str = ''.join(
      ['[{:<1}]  {:<20}',
       str(repeat_str_format * (num_cols - 1))])
  header_str = header_format_str.format('p', 'Word', 'p(real)', 'log-perp',
                                        'log(p(a))', 'r', 'R=V*(s)', 'b=V(s)',
                                        'A(a,s)')

  for i, batch in enumerate(batch_of_tuples):
    print(' Sample: %d' % i)
    log.write(' Sample %d.\n' % i)
    print('  ', header_str)
    log.write('  ' + str(header_str) + '\n')

    for j, t in enumerate(batch):
      t = list(t)
      t[0] = id_to_word[t[0]]
      buffer_str = format_str.format(int(present[i][j]), *t)
      print('  ', buffer_str)
      log.write('  ' + str(buffer_str) + '\n')
  log.flush()


def generate_RL_logs(sess, model, log, id_to_word, feed):
  """Generate complete logs while running with REINFORCE."""
  # Impute Sequences.
  [
      p,
      fake_sequence_eval,
      fake_predictions_eval,
      _,
      fake_cross_entropy_losses_eval,
      _,
      fake_log_probs_eval,
      fake_rewards_eval,
      fake_baselines_eval,
      cumulative_rewards_eval,
      fake_advantages_eval,
  ] = sess.run(
      [
          model.present,
          model.fake_sequence,
          model.fake_predictions,
          model.real_predictions,
          model.fake_cross_entropy_losses,
          model.fake_logits,
          model.fake_log_probs,
          model.fake_rewards,
          model.fake_baselines,
          model.cumulative_rewards,
          model.fake_advantages,
      ],
      feed_dict=feed)

  indices = np.asarray(fake_sequence_eval)

  # Convert Discriminator linear layer to probability.
  fake_prob_eval = expit(fake_predictions_eval)

  # Add metrics.
  fake_tuples = zip_metrics(indices, fake_prob_eval,
                            fake_cross_entropy_losses_eval, fake_log_probs_eval,
                            fake_rewards_eval, cumulative_rewards_eval,
                            fake_baselines_eval, fake_advantages_eval)

  # real_tuples = zip_metrics(indices, )

  # Print forward sequences.
  tuples_to_print = fake_tuples[:FLAGS.max_num_to_print]
  print_formatted(p, id_to_word, log, tuples_to_print)

  print('Samples')
  log.write('Samples\n')
  samples = print_and_log(log, id_to_word, fake_sequence_eval,
                          FLAGS.max_num_to_print)
  return samples


def generate_logs(sess, model, log, id_to_word, feed):
  """Impute Sequences using the model for a particular feed and send it to
  logs."""
  # Impute Sequences.
  [
      p, sequence_eval, fake_predictions_eval, fake_cross_entropy_losses_eval,
      fake_logits_eval
  ] = sess.run(
      [
          model.present, model.fake_sequence, model.fake_predictions,
          model.fake_cross_entropy_losses, model.fake_logits
      ],
      feed_dict=feed)

  # Convert Discriminator linear layer to probability.
  fake_prob_eval = expit(fake_predictions_eval)

  # Forward Masked Tuples.
  fake_tuples = zip_seq_pred_crossent(id_to_word, sequence_eval, fake_prob_eval,
                                      fake_cross_entropy_losses_eval)

  tuples_to_print = fake_tuples[:FLAGS.max_num_to_print]

  if FLAGS.print_verbose:
    print('fake_logits_eval')
    print(fake_logits_eval)

  for i, batch in enumerate(tuples_to_print):
    print(' Sample %d.' % i)
    log.write(' Sample %d.\n' % i)
    for j, pred in enumerate(batch):
      buffer_str = ('[{:<1}]  {:<20}  {:<7.3f} {:<7.3f}').format(
          int(p[i][j]), pred[0], pred[1], pred[2])
      print('  ', buffer_str)
      log.write('  ' + str(buffer_str) + '\n')
  log.flush()

  print('Samples')
  log.write('Samples\n')
  samples = print_and_log(log, id_to_word, sequence_eval,
                          FLAGS.max_num_to_print)
  return samples


def create_merged_ngram_dictionaries(indices, n):
  """Generate a single dictionary for the full batch.

  Args:
    indices:  List of lists of indices.
    n:  Degree of n-grams.

  Returns:
    Dictionary of hashed(n-gram tuples) to counts in the batch of indices.
  """
  ngram_dicts = []

  for ind in indices:
    ngrams = n_gram.find_all_ngrams(ind, n=n)
    ngram_counts = n_gram.construct_ngrams_dict(ngrams)
    ngram_dicts.append(ngram_counts)

  merged_gen_dict = Counter()
  for ngram_dict in ngram_dicts:
    merged_gen_dict += Counter(ngram_dict)
  return merged_gen_dict


def sequence_ngram_evaluation(sess, sequence, log, feed, data_ngram_count, n):
  """Calculates the percent of ngrams produced in the sequence is present in
  data_ngram_count.

  Args:
    sess: tf.Session.
    sequence: Sequence Tensor from the MaskGAN model.
    log:  gFile log.
    feed: Feed to evaluate.
    data_ngram_count:  Dictionary of hashed(n-gram tuples) to counts in the
      data_set.

  Returns:
    avg_percent_captured: Percent of produced ngrams that appear in the
      data_ngram_count.
  """
  del log
  # Impute sequence.
  [sequence_eval] = sess.run([sequence], feed_dict=feed)
  indices = sequence_eval

  # Retrieve the counts across the batch of indices.
  gen_ngram_counts = create_merged_ngram_dictionaries(
      indices, n=n)
  return n_gram.percent_unique_ngrams_in_train(data_ngram_count,
                                               gen_ngram_counts)
