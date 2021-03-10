# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Module for extracting segments from sentences in documents."""

import tensorflow as tf


# Get a random tensor like `positions` and make some decisions
def _get_random(positions, random_fn):
  flat_random = random_fn(
      shape=tf.shape(positions.flat_values),
      minval=0,
      maxval=1,
      dtype=tf.float32)
  return positions.with_flat_values(flat_random)


# For every position j in a row, sample a position preceeding j or
# a position which is [0, j-1]
def _random_int_up_to(maxval, random_fn):
  # Need to cast because the int kernel for uniform doesn't support bcast.
  # We add one because maxval is exclusive, and this will get rounded down
  # when we cast back to int.
  float_maxval = tf.cast(maxval, tf.float32)
  return tf.cast(
      random_fn(
          shape=tf.shape(maxval),
          minval=tf.zeros_like(float_maxval),
          maxval=float_maxval),
      dtype=maxval.dtype)


def _random_int_from_range(minval, maxval, random_fn):
  # Need to cast because the int kernel for uniform doesn't support bcast.
  # We add one because maxval is exclusive, and this will get rounded down
  # when we cast back to int.
  float_minval = tf.cast(minval, tf.float32)
  float_maxval = tf.cast(maxval, tf.float32)
  return tf.cast(
      random_fn(tf.shape(maxval), minval=float_minval, maxval=float_maxval),
      maxval.dtype)


def _sample_from_other_batch(sentences, random_fn):
  """Samples sentences from other batches."""
  # other_batch: <int64>[num_sentences]: The batch to sample from for each
  # sentence.
  other_batch = random_fn(
      shape=[tf.size(sentences)],
      minval=0,
      maxval=sentences.nrows() - 1,
      dtype=tf.int64)

  other_batch += tf.cast(other_batch >= sentences.value_rowids(), tf.int64)

  # other_sentence: <int64>[num_sentences]: The sentence within each batch
  # that we sampled.
  other_sentence = _random_int_up_to(
      tf.gather(sentences.row_lengths(), other_batch), random_fn)
  return sentences.with_values(tf.stack([other_batch, other_sentence], axis=1))


def get_sentence_order_labels(sentences,
                              random_threshold=0.5,
                              random_next_threshold=0.5,
                              random_fn=tf.random.uniform):
  """Extract segments and labels for sentence order prediction (SOP) task.

  Extracts the segment and labels for the sentence order prediction task
  defined in "ALBERT: A Lite BERT for Self-Supervised Learning of Language
  Representations" (https://arxiv.org/pdf/1909.11942.pdf)

  Args:
    sentences: a `RaggedTensor` of shape [batch, (num_sentences)] with string
      dtype.
    random_threshold: (optional) A float threshold between 0 and 1, used to
      determine whether to extract a random, out-of-batch sentence or a
      suceeding sentence. Higher value favors succeeding sentence.
    random_next_threshold: (optional) A float threshold between 0 and 1, used to
      determine whether to extract either a random, out-of-batch, or succeeding
      sentence or a preceeding sentence. Higher value favors preceeding
      sentences.
    random_fn: (optional) An op used to generate random float values.

  Returns:
    a tuple of (preceeding_or_random_next, is_suceeding_or_random) where:
      preceeding_or_random_next: a `RaggedTensor` of strings with the same shape
        as `sentences` and contains either a preceeding, suceeding, or random
        out-of-batch sentence respective to its counterpart in `sentences` and
        dependent on its label in `is_preceeding_or_random_next`.
      is_suceeding_or_random: a `RaggedTensor` of bool values with the
        same shape as `sentences` and is True if it's corresponding sentence in
        `preceeding_or_random_next` is a random or suceeding sentence, False
        otherwise.
  """
  # Create a RaggedTensor in the same shape as sentences ([doc, (sentences)])
  # whose values are index positions.
  positions = tf.ragged.range(sentences.row_lengths())

  row_lengths_broadcasted = tf.expand_dims(positions.row_lengths(),
                                           -1) + 0 * positions
  row_lengths_broadcasted_flat = row_lengths_broadcasted.flat_values

  # Generate indices for all preceeding, succeeding and random.
  # For every position j in a row, sample a position preceeding j or
  # a position which is [0, j-1]
  all_preceding = tf.ragged.map_flat_values(_random_int_up_to, positions,
                                            random_fn)

  # For every position j, sample a position following j, or a position
  # which is [j, row_max]
  all_succeeding = positions.with_flat_values(
      tf.ragged.map_flat_values(_random_int_from_range,
                                positions.flat_values + 1,
                                row_lengths_broadcasted_flat, random_fn))

  # Convert to format that is convenient for `gather_nd`
  rows_broadcasted = tf.expand_dims(tf.range(sentences.nrows()),
                                    -1) + 0 * positions
  all_preceding_nd = tf.stack([rows_broadcasted, all_preceding], -1)
  all_succeeding_nd = tf.stack([rows_broadcasted, all_succeeding], -1)
  all_random_nd = _sample_from_other_batch(positions, random_fn)

  # There's a few spots where there is no "preceding" or "succeeding" item (e.g.
  # first and last sentences in a document). Mark where these are and we will
  # patch them up to grab a random sentence from another document later.
  all_zeros = tf.zeros_like(positions)
  all_ones = tf.ones_like(positions)
  valid_preceding_mask = tf.cast(
      tf.concat([all_zeros[:, :1], all_ones[:, 1:]], -1), tf.bool)
  valid_succeeding_mask = tf.cast(
      tf.concat([all_ones[:, :-1], all_zeros[:, -1:]], -1), tf.bool)

  # Decide what to use for the segment: (1) random, out-of-batch, (2) preceeding
  # item, or (3) succeeding.
  # Should get out-of-batch instead of succeeding item
  should_get_random = ((_get_random(positions, random_fn) > random_threshold)
                       | tf.logical_not(valid_succeeding_mask))
  random_or_succeeding_nd = tf.compat.v1.where(should_get_random, all_random_nd,
                                               all_succeeding_nd)
  # Choose which items should get a random succeeding item. Force positions that
  # don't have a valid preceeding items to get a random succeeding item.
  should_get_random_or_succeeding = (
      (_get_random(positions, random_fn) > random_next_threshold)
      | tf.logical_not(valid_preceding_mask))
  gather_indices = tf.compat.v1.where(should_get_random_or_succeeding,
                                      random_or_succeeding_nd, all_preceding_nd)
  return (tf.gather_nd(sentences,
                       gather_indices), should_get_random_or_succeeding)


def get_next_sentence_labels(sentences,
                             random_threshold=0.5,
                             random_fn=tf.random.uniform):
  """Extracts the next sentence label from sentences.

  Args:
    sentences: A `RaggedTensor` of strings w/ shape [batch, (num_sentences)].
    random_threshold: (optional) A float threshold between 0 and 1, used to
      determine whether to extract a random sentence or the immediate next
      sentence. Higher value favors next sentence.
    random_fn: (optional) An op used to generate random float values.

  Returns:
    A tuple of (next_sentence_or_random, is_next_sentence) where:

    next_sentence_or_random:  A `Tensor` with shape [num_sentences] that
      contains either the subsequent sentence of `segment_a` or a randomly
      injected sentence.
    is_next_sentence: A `Tensor` of bool w/ shape [num_sentences]
      that contains whether or not `next_sentence_or_random` is truly a
      subsequent sentence or not.
  """
  # shift everyone to get the next sentence predictions positions
  positions = tf.ragged.range(sentences.row_lengths())

  # Shift every position down to the right.
  next_sentences_pos = (positions + 1) % tf.expand_dims(sentences.row_lengths(),
                                                        1)
  rows_broadcasted = tf.expand_dims(tf.range(sentences.nrows()),
                                    -1) + 0 * positions
  next_sentences_pos_nd = tf.stack([rows_broadcasted, next_sentences_pos], -1)
  all_random_nd = _sample_from_other_batch(positions, random_fn)

  # Mark the items that don't have a next sentence (e.g. the last
  # sentences in the document). We will patch these up and force them to grab a
  # random sentence from a random document.
  valid_next_sentences = tf.cast(
      tf.concat([
          tf.ones_like(positions)[:, :-1],
          tf.zeros([positions.nrows(), 1], dtype=tf.int64)
      ], -1), tf.bool)

  is_random = ((_get_random(positions, random_fn) > random_threshold)
               | tf.logical_not(valid_next_sentences))
  gather_indices = tf.compat.v1.where(is_random, all_random_nd,
                                      next_sentences_pos_nd)
  return tf.gather_nd(sentences, gather_indices), tf.logical_not(is_random)
