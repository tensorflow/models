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

"""Random helper functions for converting between indices and one-hot encodings
as well as printing/logging helpers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
import tensorflow as tf


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean/' + name, mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
  tf.summary.scalar('sttdev/' + name, stddev)
  tf.summary.scalar('max/' + name, tf.reduce_max(var))
  tf.summary.scalar('min/' + name, tf.reduce_min(var))
  tf.summary.histogram(name, var)


def zip_seq_pred_crossent(id_to_word, sequences, predictions, cross_entropy):
  """Zip together the sequences, predictions, cross entropy."""
  indices = convert_to_indices(sequences)

  batch_of_metrics = []

  for ind_batch, pred_batch, crossent_batch in zip(indices, predictions,
                                                   cross_entropy):
    metrics = []

    for index, pred, crossent in zip(ind_batch, pred_batch, crossent_batch):
      metrics.append([str(id_to_word[index]), pred, crossent])

    batch_of_metrics.append(metrics)
  return batch_of_metrics


def print_and_log(log, id_to_word, sequence_eval, max_num_to_print=5):
  """Helper function for printing and logging evaluated sequences."""
  indices_eval = convert_to_indices(sequence_eval)
  indices_arr = np.asarray(indices_eval)
  samples = convert_to_human_readable(id_to_word, indices_arr, max_num_to_print)

  for i, sample in enumerate(samples):
    print('Sample', i, '. ', sample)
    log.write('\nSample ' + str(i) + '. ' + sample)
  log.write('\n')
  print('\n')
  log.flush()


def convert_to_human_readable(id_to_word, arr, max_num_to_print):
  """Convert a np.array of indices into words using id_to_word dictionary.
  Return max_num_to_print results.
  """
  assert arr.ndim == 2

  samples = []
  for sequence_id in xrange(min(len(arr), max_num_to_print)):
    buffer_str = ' '.join(
        [str(id_to_word[index]) for index in arr[sequence_id, :]])
    samples.append(buffer_str)
  return samples


def index_to_vocab_array(indices, vocab_size, sequence_length):
  """Convert the indices into an array with vocab_size one-hot encoding."""

  # Extract properties of the indices.
  num_batches = len(indices)
  shape = list(indices.shape)
  shape.append(vocab_size)

  # Construct the vocab_size array.
  new_arr = np.zeros(shape)

  for n in xrange(num_batches):
    indices_batch = indices[n]
    new_arr_batch = new_arr[n]

    # We map all indices greater than the vocabulary size to an unknown
    # character.
    indices_batch = np.where(indices_batch < vocab_size, indices_batch,
                             vocab_size - 1)

    # Convert indices to vocab_size dimensions.
    new_arr_batch[np.arange(sequence_length), indices_batch] = 1
  return new_arr


def convert_to_indices(sequences):
  """Convert a list of size [batch_size, sequence_length, vocab_size] to
  a list of size [batch_size, sequence_length] where the vocab element is
  denoted by the index.
  """
  batch_of_indices = []

  for sequence in sequences:
    indices = []
    for embedding in sequence:
      indices.append(np.argmax(embedding))
    batch_of_indices.append(indices)
  return batch_of_indices


def convert_and_zip(id_to_word, sequences, predictions):
  """Helper function for printing or logging.  Retrieves list of sequences
  and predictions and zips them together.
  """
  indices = convert_to_indices(sequences)

  batch_of_indices_predictions = []

  for index_batch, pred_batch in zip(indices, predictions):
    indices_predictions = []

    for index, pred in zip(index_batch, pred_batch):
      indices_predictions.append([str(id_to_word[index]), pred])
    batch_of_indices_predictions.append(indices_predictions)
  return batch_of_indices_predictions


def recursive_length(item):
  """Recursively determine the total number of elements in nested list."""
  if type(item) == list:
    return sum(recursive_length(subitem) for subitem in item)
  else:
    return 1.


def percent_correct(real_sequence, fake_sequences):
  """Determine the percent of tokens correctly generated within a batch."""
  identical = 0.
  for fake_sequence in fake_sequences:
    for real, fake in zip(real_sequence, fake_sequence):
      if real == fake:
        identical += 1.
  return identical / recursive_length(fake_sequences)
