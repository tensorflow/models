# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions that perform various calculations."""

import collections

import tensorflow as tf

EOS = '<eos>'  # End of sequence symbol
UNK = '<unk>'  # Unknown symbol
N = 'N'
DS = '$'

SPECIAL_SYMBOLS = [UNK, N, DS, EOS]


def build_vocab_id_dict(data_file):
  """Create a dictionary mapping all words in the vocabulary to integer IDs."""
  assert tf.gfile.Exists(data_file), 'File %s does not exist.' % data_file
  with tf.gfile.Open(data_file, 'r') as f:
    unique_words = set(f.read().split())

    # Remove special symbols from the set.
    for symbol in SPECIAL_SYMBOLS:
      if symbol in unique_words:
        unique_words.remove(symbol)
    # Create list of unique words with the special symbols at the front.
    unique_words = SPECIAL_SYMBOLS + sorted(list(unique_words))

    # Create a default dict of word->id so that any words not in the dictionary
    # is mapped to 0 (corresponding to '<unk>').
    vocab_dict = collections.defaultdict(
        int, zip(unique_words, range(len(unique_words))))
    return vocab_dict


def build_reverse_vocab_dict(vocab_dict):
  """Build a dictionary of id->word."""
  reverse_dict = {i: word for word, i in vocab_dict.iteritems()}
  reverse_dict[3] = '.\n'  # Replace '<eos>' with a period with a new line.
  return reverse_dict


def steps_per_epoch(input_word_count, batch_size, unrolled_count):
  """Calculate the number of train/eval steps needed to complete an epoch.

  Args:
    input_word_count: The total number of words in the dataset.
    batch_size: The number of examples per batch.
    unrolled_count: The number of times the RNN will be unrolled for BPTT.
        i.e. the number of words per example in the batch.
  """
  words_per_batch = batch_size * unrolled_count
  return input_word_count // words_per_batch


def perplexity_metric(losses):
  """Returns the perplexity while keeping a running average of the losses."""
  # Use variable scope to avoid variable name collisions.
  with tf.variable_scope('perplexity_metric'):
    total = tf.get_local_variable('total_loss', initializer=0.0)
    count = tf.get_local_variable('count', initializer=0.0)

    # Use control dependencies to ensure that updates are accurate.
    with tf.control_dependencies([losses]):
      update_total_op = tf.assign_add(total, tf.reduce_sum(losses))
      update_count_op = tf.assign_add(
          count, tf.cast(tf.size(losses), tf.float32))

    # Calculate the perplexity
    perplexity = tf.exp(tf.div(total, count))
    # Keep an update of the total loss and count
    update_op = tf.exp(tf.div(update_total_op, update_count_op))
    return perplexity, update_op
