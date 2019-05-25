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
"""Beam search to find the translated sequence with the highest probability.

Source implementation from Tensor2Tensor:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/beam_search.py
"""

import tensorflow as tf
from tensorflow.python.util import nest

# Default value for INF
INF = 1. * 1e7


class _StateKeys(object):
  """Keys to dictionary storing the state of the beam search loop."""

  # Variable storing the loop index.
  CUR_INDEX = "CUR_INDEX"

  # Top sequences that are alive for each batch item. Alive sequences are ones
  # that have not generated an EOS token. Sequences that reach EOS are marked as
  # finished and moved to the FINISHED_SEQ tensor.
  # Has shape [batch_size, beam_size, CUR_INDEX + 1]
  ALIVE_SEQ = "ALIVE_SEQ"
  # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
  ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
  # Dictionary of cached values for each alive sequence. The cache stores
  # the encoder output, attention bias, and the decoder attention output from
  # the previous iteration.
  ALIVE_CACHE = "ALIVE_CACHE"

  # Top finished sequences for each batch item.
  # Has shape [batch_size, beam_size, CUR_INDEX + 1]. Sequences that are
  # shorter than CUR_INDEX + 1 are padded with 0s.
  FINISHED_SEQ = "FINISHED_SEQ"
  # Scores for each finished sequence. Score = log probability / length norm
  # Shape [batch_size, beam_size]
  FINISHED_SCORES = "FINISHED_SCORES"
  # Flags indicating which sequences in the finished sequences are finished.
  # At the beginning, all of the sequences in FINISHED_SEQ are filler values.
  # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
  FINISHED_FLAGS = "FINISHED_FLAGS"


class SequenceBeamSearch(object):
  """Implementation of beam search loop."""

  def __init__(self, symbols_to_logits_fn, vocab_size, batch_size,
               beam_size, alpha, max_decode_length, eos_id):
    self.symbols_to_logits_fn = symbols_to_logits_fn
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.beam_size = beam_size
    self.alpha = alpha
    self.max_decode_length = max_decode_length
    self.eos_id = eos_id

  def search(self, initial_ids, initial_cache):
    """Beam search for sequences with highest scores."""
    state, state_shapes = self._create_initial_state(initial_ids, initial_cache)

    finished_state = tf.while_loop(
        self._continue_search, self._search_step, loop_vars=[state],
        shape_invariants=[state_shapes], parallel_iterations=1, back_prop=False)
    finished_state = finished_state[0]

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

  def _create_initial_state(self, initial_ids, initial_cache):
    """Return initial state dictionary and its shape invariants.

    Args:
      initial_ids: initial ids to pass into the symbols_to_logits_fn.
        int tensor with shape [batch_size, 1]
      initial_cache: dictionary storing values to be passed into the
        symbols_to_logits_fn.

    Returns:
        state and shape invariant dictionaries with keys from _StateKeys
    """
    # Current loop index (starts at 0)
    cur_index = tf.constant(0)

    # Create alive sequence with shape [batch_size, beam_size, 1]
    alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
    alive_seq = tf.expand_dims(alive_seq, axis=2)

    # Create tensor for storing initial log probabilities.
    # Assume initial_ids are prob 1.0
    initial_log_probs = tf.constant(
        [[0.] + [-float("inf")] * (self.beam_size - 1)])
    alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1])

    # Expand all values stored in the dictionary to the beam size, so that each
    # beam has a separate cache.
    alive_cache = nest.map_structure(
        lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)

    # Initialize tensor storing finished sequences with filler values.
    finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)

    # Set scores of the initial finished seqs to negative infinity.
    finished_scores = tf.ones([self.batch_size, self.beam_size]) * -INF

    # Initialize finished flags with all False values.
    finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)

    # Create state dictionary
    state = {
        _StateKeys.CUR_INDEX: cur_index,
        _StateKeys.ALIVE_SEQ: alive_seq,
        _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
        _StateKeys.ALIVE_CACHE: alive_cache,
        _StateKeys.FINISHED_SEQ: finished_seq,
        _StateKeys.FINISHED_SCORES: finished_scores,
        _StateKeys.FINISHED_FLAGS: finished_flags
    }

    # Create state invariants for each value in the state dictionary. Each
    # dimension must be a constant or None. A None dimension means either:
    #   1) the dimension's value is a tensor that remains the same but may
    #      depend on the input sequence to the model (e.g. batch size).
    #   2) the dimension may have different values on different iterations.
    state_shape_invariants = {
        _StateKeys.CUR_INDEX: tf.TensorShape([]),
        _StateKeys.ALIVE_SEQ: tf.TensorShape([None, self.beam_size, None]),
        _StateKeys.ALIVE_LOG_PROBS: tf.TensorShape([None, self.beam_size]),
        _StateKeys.ALIVE_CACHE: nest.map_structure(
            _get_shape_keep_last_dim, alive_cache),
        _StateKeys.FINISHED_SEQ: tf.TensorShape([None, self.beam_size, None]),
        _StateKeys.FINISHED_SCORES: tf.TensorShape([None, self.beam_size]),
        _StateKeys.FINISHED_FLAGS: tf.TensorShape([None, self.beam_size])
    }

    return state, state_shape_invariants

  def _continue_search(self, state):
    """Return whether to continue the search loop.

    The loops should terminate when
      1) when decode length has been reached, or
      2) when the worst score in the finished sequences is better than the best
         score in the alive sequences (i.e. the finished sequences are provably
         unchanging)

    Args:
      state: A dictionary with the current loop state.

    Returns:
      Bool tensor with value True if loop should continue, False if loop should
      terminate.
    """
    i = state[_StateKeys.CUR_INDEX]
    alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
    finished_scores = state[_StateKeys.FINISHED_SCORES]
    finished_flags = state[_StateKeys.FINISHED_FLAGS]

    not_at_max_decode_length = tf.less(i, self.max_decode_length)

    # Calculate largest length penalty (the larger penalty, the better score).
    max_length_norm = _length_normalization(self.alpha, self.max_decode_length)
    # Get the best possible scores from alive sequences.
    best_alive_scores = alive_log_probs[:, 0] / max_length_norm

    # Compute worst score in finished sequences for each batch element
    finished_scores *= tf.cast(finished_flags,
                               tf.float32)  # set filler scores to zero
    lowest_finished_scores = tf.reduce_min(finished_scores, axis=1)

    # If there are no finished sequences in a batch element, then set the lowest
    # finished score to -INF for that element.
    finished_batches = tf.reduce_any(finished_flags, 1)
    lowest_finished_scores += (1.0 -
                               tf.cast(finished_batches, tf.float32)) * -INF

    worst_finished_score_better_than_best_alive_score = tf.reduce_all(
        tf.greater(lowest_finished_scores, best_alive_scores)
    )

    return tf.logical_and(
        not_at_max_decode_length,
        tf.logical_not(worst_finished_score_better_than_best_alive_score)
    )

  def _search_step(self, state):
    """Beam search loop body.

    Grow alive sequences by a single ID. Sequences that have reached the EOS
    token are marked as finished. The alive and finished sequences with the
    highest log probabilities and scores are returned.

    A sequence's finished score is calculating by dividing the log probability
    by the length normalization factor. Without length normalization, the
    search is more likely to return shorter sequences.

    Args:
      state: A dictionary with the current loop state.

    Returns:
      new state dictionary.
    """
    # Grow alive sequences by one token.
    new_seq, new_log_probs, new_cache = self._grow_alive_seq(state)
    # Collect top beam_size alive sequences
    alive_state = self._get_new_alive_state(new_seq, new_log_probs, new_cache)

    # Combine newly finished sequences with existing finished sequences, and
    # collect the top k scoring sequences.
    finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)

    # Increment loop index and create new state dictionary
    new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
    new_state.update(alive_state)
    new_state.update(finished_state)
    return [new_state]

  def _grow_alive_seq(self, state):
    """Grow alive sequences by one token, and collect top 2*beam_size sequences.

    2*beam_size sequences are collected because some sequences may have reached
    the EOS token. 2*beam_size ensures that at least beam_size sequences are
    still alive.

    Args:
      state: A dictionary with the current loop state.
    Returns:
      Tuple of
      (Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
       Scores of returned sequences [batch_size, 2 * beam_size],
       New alive cache, for each of the 2 * beam_size sequences)
    """
    i = state[_StateKeys.CUR_INDEX]
    alive_seq = state[_StateKeys.ALIVE_SEQ]
    alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
    alive_cache = state[_StateKeys.ALIVE_CACHE]

    beams_to_keep = 2 * self.beam_size

    # Get logits for the next candidate IDs for the alive sequences. Get the new
    # cache values at the same time.
    flat_ids = _flatten_beam_dim(alive_seq)  # [batch_size * beam_size]
    flat_cache = nest.map_structure(_flatten_beam_dim, alive_cache)

    flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, i, flat_cache)

    # Unflatten logits to shape [batch_size, beam_size, vocab_size]
    logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)
    new_cache = nest.map_structure(
        lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size),
        flat_cache)

    # Convert logits to normalized log probs
    candidate_log_probs = _log_prob_from_logits(logits)

    # Calculate new log probabilities if each of the alive sequences were
    # extended # by the the candidate IDs.
    # Shape [batch_size, beam_size, vocab_size]
    log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

    # Each batch item has beam_size * vocab_size candidate sequences. For each
    # batch item, get the k candidates with the highest log probabilities.
    flat_log_probs = tf.reshape(log_probs,
                                [-1, self.beam_size * self.vocab_size])
    topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

    # Extract the alive sequences that generate the highest log probabilities
    # after being extended.
    topk_beam_indices = topk_indices // self.vocab_size
    topk_seq, new_cache = _gather_beams(
        [alive_seq, new_cache], topk_beam_indices, self.batch_size,
        beams_to_keep)

    # Append the most probable IDs to the topk sequences
    topk_ids = topk_indices % self.vocab_size
    topk_ids = tf.expand_dims(topk_ids, axis=2)
    topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
    return topk_seq, topk_log_probs, new_cache

  def _get_new_alive_state(self, new_seq, new_log_probs, new_cache):
    """Gather the top k sequences that are still alive.

    Args:
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor with shape [batch_size, 2 * beam_size, cur_index + 1]
      new_log_probs: Log probabilities of new sequences
        float32 tensor with shape [batch_size, beam_size]
      new_cache: Dict of cached values for each sequence.

    Returns:
      Dictionary with alive keys from _StateKeys:
        {Top beam_size sequences that are still alive (don't end with eos_id)
         Log probabilities of top alive sequences
         Dict cache storing decoder states for top alive sequences}
    """
    # To prevent finished sequences from being considered, set log probs to -INF
    new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
    new_log_probs += tf.cast(new_finished_flags, tf.float32) * -INF

    top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beams(
        [new_seq, new_log_probs, new_cache], new_log_probs, self.batch_size,
        self.beam_size)

    return {
        _StateKeys.ALIVE_SEQ: top_alive_seq,
        _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
        _StateKeys.ALIVE_CACHE: top_alive_cache
    }

  def _get_new_finished_state(self, state, new_seq, new_log_probs):
    """Combine new and old finished sequences, and gather the top k sequences.

    Args:
      state: A dictionary with the current loop state.
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor with shape [batch_size, beam_size, i + 1]
      new_log_probs: Log probabilities of new sequences
        float32 tensor with shape [batch_size, beam_size]

    Returns:
      Dictionary with finished keys from _StateKeys:
        {Top beam_size finished sequences based on score,
         Scores of finished sequences,
         Finished flags of finished sequences}
    """
    i = state[_StateKeys.CUR_INDEX]
    finished_seq = state[_StateKeys.FINISHED_SEQ]
    finished_scores = state[_StateKeys.FINISHED_SCORES]
    finished_flags = state[_StateKeys.FINISHED_FLAGS]

    # First append a column of 0-ids to finished_seq to increment the length.
    # New shape of finished_seq: [batch_size, beam_size, i + 1]
    finished_seq = tf.concat(
        [finished_seq,
         tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)], axis=2)

    # Calculate new seq scores from log probabilities.
    length_norm = _length_normalization(self.alpha, i + 1)
    new_scores = new_log_probs / length_norm

    # Set the scores of the still-alive seq in new_seq to large negative values.
    new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
    new_scores += (1. - tf.cast(new_finished_flags, tf.float32)) * -INF

    # Combine sequences, scores, and flags.
    finished_seq = tf.concat([finished_seq, new_seq], axis=1)
    finished_scores = tf.concat([finished_scores, new_scores], axis=1)
    finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

    # Return the finished sequences with the best scores.
    top_finished_seq, top_finished_scores, top_finished_flags = (
        _gather_topk_beams([finished_seq, finished_scores, finished_flags],
                           finished_scores, self.batch_size, self.beam_size))

    return {
        _StateKeys.FINISHED_SEQ: top_finished_seq,
        _StateKeys.FINISHED_SCORES: top_finished_scores,
        _StateKeys.FINISHED_FLAGS: top_finished_flags
    }


def sequence_beam_search(
    symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
    alpha, max_decode_length, eos_id):
  """Search for sequence of subtoken ids with the largest probability.

  Args:
    symbols_to_logits_fn: A function that takes in ids, index, and cache as
      arguments. The passed in arguments will have shape:
        ids -> [batch_size * beam_size, index]
        index -> [] (scalar)
        cache -> nested dictionary of tensors [batch_size * beam_size, ...]
      The function must return logits and new cache.
        logits -> [batch * beam_size, vocab_size]
        new cache -> same shape/structure as inputted cache
    initial_ids: Starting ids for each batch item.
      int32 tensor with shape [batch_size]
    initial_cache: dict containing starting decoder variables information
    vocab_size: int size of tokens
    beam_size: int number of beams
    alpha: float defining the strength of length normalization
    max_decode_length: maximum length to decoded sequence
    eos_id: int id of eos token, used to determine when a sequence has finished

  Returns:
    Top decoded sequences [batch_size, beam_size, max_decode_length]
    sequence scores [batch_size, beam_size]
  """
  batch_size = tf.shape(initial_ids)[0]
  sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                           beam_size, alpha, max_decode_length, eos_id)
  return sbs.search(initial_ids, initial_cache)


def _log_prob_from_logits(logits):
  return logits - tf.reduce_logsumexp(logits, axis=2, keepdims=True)


def _length_normalization(alpha, length):
  """Return length normalization factor."""
  return tf.pow(((5. + tf.cast(length, tf.float32)) / 6.), alpha)


def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.

  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.

  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)


def _shape_list(tensor):
  """Return a list of the tensor's shape, and ensure no None values in list."""
  # Get statically known shape (may contain None's for unknown dimensions)
  shape = tensor.get_shape().as_list()

  # Ensure that the shape values are not None
  dynamic_shape = tf.shape(tensor)
  for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
    if shape[i] is None:
      shape[i] = dynamic_shape[i]
  return shape


def _get_shape_keep_last_dim(tensor):
  shape_list = _shape_list(tensor)

  # Only the last
  for i in range(len(shape_list) - 1):
    shape_list[i] = None

  if isinstance(shape_list[-1], tf.Tensor):
    shape_list[-1] = None
  return tf.TensorShape(shape_list)


def _flatten_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension.

  Args:
    tensor: Tensor to reshape of shape [A, B, ...]

  Returns:
    Reshaped tensor of shape [A*B, ...]
  """
  shape = _shape_list(tensor)
  shape[0] *= shape[1]
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size].

  Args:
    tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    batch_size: Tensor, original batch size.
    beam_size: int, original beam size.

  Returns:
    Reshaped tensor of shape [batch_size, beam_size, ...]
  """
  shape = _shape_list(tensor)
  new_shape = [batch_size, beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
  """Gather beams from nested structure of tensors.

  Each tensor in nested represents a batch of beams, where beam refers to a
  single search state (beam search involves searching through multiple states
  in parallel).

  This function is used to gather the top beams, specified by
  beam_indices, from the nested tensors.

  Args:
    nested: Nested structure (tensor, list, tuple or dict) containing tensors
      with shape [batch_size, beam_size, ...].
    beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
     value in beam_indices must be between [0, beam_size), and are not
     necessarily unique.
    batch_size: int size of batch
    new_beam_size: int number of beams to be pulled from the nested tensors.

  Returns:
    Nested structure containing tensors with shape
      [batch_size, new_beam_size, ...]
  """
  # Computes the i'th coodinate that contains the batch index for gather_nd.
  # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
  batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

  # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
  # with shape [batch_size, beam_size, 2], where the last dimension contains
  # the (i, j) gathering coordinates.
  coordinates = tf.stack([batch_pos, beam_indices], axis=2)

  return nest.map_structure(
      lambda state: tf.gather_nd(state, coordinates), nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
  """Gather top beams from nested structure."""
  _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
  return _gather_beams(nested, topk_indexes, batch_size, beam_size)
