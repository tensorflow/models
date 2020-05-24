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
"""Beam search in TF v2."""

import tensorflow as tf

from official.nlp.transformer import beam_search_v1 as v1

_StateKeys = v1._StateKeys  # pylint: disable=protected-access


class SequenceBeamSearchV2(v1.SequenceBeamSearch):
  """Implementation of beam search loop in v2."""

  def search(self, initial_ids, initial_cache):
    """Beam search for sequences with highest scores."""
    state, state_shapes = self._create_initial_state(initial_ids, initial_cache)

    finished_state = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(self._continue_search,
                      self._search_step,
                      loop_vars=[state],
                      shape_invariants=[state_shapes],
                      parallel_iterations=1))
    finished_state = finished_state[0]

    alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
    alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
    finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
    finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
    finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

    # 2.0 changes tf.where behavior. Should make parameters broadcastable.
    finished_cond = tf.reduce_any(finished_flags, 1, name="finished_cond")
    seq_cond = _expand_to_same_rank(finished_cond, finished_seq)
    score_cond = _expand_to_same_rank(finished_cond, finished_scores)

    # Account for corner case where there are no finished sequences for a
    # particular batch item. In that case, return alive sequences for that batch
    # item.
    finished_seq = tf.where(seq_cond, finished_seq, alive_seq)
    finished_scores = tf.where(
        score_cond, finished_scores, alive_log_probs)
    return finished_seq, finished_scores


def sequence_beam_search(symbols_to_logits_fn,
                         initial_ids,
                         initial_cache,
                         vocab_size,
                         beam_size,
                         alpha,
                         max_decode_length,
                         eos_id,
                         padded_decode=False,
                         dtype="float32"):
  """Search for sequence of subtoken ids with the largest probability.

  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape:
        ids -> A tensor with shape [batch_size * beam_size, index].
        index -> A scalar.
        cache -> A nested dictionary of tensors [batch_size * beam_size, ...].
      The function must return a tuple of logits and new cache:
        logits -> A tensor with shape [batch * beam_size, vocab_size].
        new cache -> A nested dictionary with the same shape/structure as the
          inputted cache.
    initial_ids: An int32 tensor with shape [batch_size]. Starting ids for
      each batch item.
    initial_cache: A dictionary, containing starting decoder variables
      information.
    vocab_size: An integer, the size of tokens.
    beam_size: An integer, the number of beams.
    alpha: A float, defining the strength of length normalization.
    max_decode_length: An integer, the maximum length to decoded a sequence.
    eos_id: An integer, ID of eos token, used to determine when a sequence has
      finished.
    padded_decode: A bool, indicating if max_sequence_length padding is used
      for beam search.
    dtype: A tensorflow data type used for score computation. The default is
      tf.float32.

  Returns:
    Top decoded sequences [batch_size, beam_size, max_decode_length]
    sequence scores [batch_size, beam_size]
  """
  batch_size = (
      initial_ids.shape.as_list()[0] if padded_decode else
      tf.shape(initial_ids)[0])
  sbs = SequenceBeamSearchV2(symbols_to_logits_fn, vocab_size, batch_size,
                             beam_size, alpha, max_decode_length, eos_id,
                             padded_decode, dtype)
  return sbs.search(initial_ids, initial_cache)


def _expand_to_same_rank(tensor, target):
  """Expands a given tensor to target's rank to be broadcastable.

  Args:
    tensor: input tensor to tile. Shape: [b, d1, ..., da]
    target: target tensor. Shape: [b, d1, ..., da, ..., dn]

  Returns:
    Tiled tensor of shape [b, d1, ..., da, 1, ..., 1] with same rank of target.

  Raises:
    ValueError, if the shape rank of rank tensor/target is None.
  """
  if tensor.shape.rank is None:
    raise ValueError("Expect rank for tensor shape, but got None.")
  if target.shape.rank is None:
    raise ValueError("Expect rank for target shape, but got None.")

  with tf.name_scope("expand_rank"):
    diff_rank = target.shape.rank - tensor.shape.rank
    for _ in range(diff_rank):
      tensor = tf.expand_dims(tensor, -1)
    return tensor
