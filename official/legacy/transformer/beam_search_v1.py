# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Beam search to find the translated sequence with the highest probability."""

import tensorflow.compat.v1 as tf
from official.nlp.modeling.ops import beam_search

_StateKeys = beam_search._StateKeys  # pylint: disable=protected-access


class SequenceBeamSearch(beam_search.SequenceBeamSearch):
  """Implementation of beam search loop."""

  def _process_finished_state(self, finished_state):
    alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
    alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
    finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
    finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
    finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

    # Account for corner case where there are no finished sequences for a
    # particular batch item. In that case, return alive sequences for that batch
    # item.
    finished_seq = tf.where(
        tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
    finished_scores = tf.where(
        tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
    return finished_seq, finished_scores


def sequence_beam_search(symbols_to_logits_fn,
                         initial_ids,
                         initial_cache,
                         vocab_size,
                         beam_size,
                         alpha,
                         max_decode_length,
                         eos_id,
                         padded_decode=False):
  """Search for sequence of subtoken ids with the largest probability.

  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape: ids -> A tensor with
        shape [batch_size * beam_size, index]. index -> A scalar. cache -> A
        nested dictionary of tensors [batch_size * beam_size, ...].
      The function must return a tuple of logits and new cache: logits -> A
        tensor with shape [batch * beam_size, vocab_size]. new cache -> A nested
        dictionary with the same shape/structure as the inputted cache.
    initial_ids: An int32 tensor with shape [batch_size]. Starting ids for each
      batch item.
    initial_cache: A dictionary, containing starting decoder variables
      information.
    vocab_size: An integer, the size of the vocabulary, used for topk
      computation.
    beam_size: An integer, the number of beams.
    alpha: A float, defining the strength of length normalization.
    max_decode_length: An integer, the maximum length to decoded a sequence.
    eos_id: An integer, ID of eos token, used to determine when a sequence has
      finished.
    padded_decode: A bool, indicating if max_sequence_length padding is used for
      beam search.

  Returns:
    Top decoded sequences [batch_size, beam_size, max_decode_length]
    sequence scores [batch_size, beam_size]
  """
  sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, beam_size, alpha,
                           max_decode_length, eos_id, padded_decode)
  return sbs.search(initial_ids, initial_cache)
