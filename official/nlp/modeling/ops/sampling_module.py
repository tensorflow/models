# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Sampling module for top_k, top_p and greedy decoding."""

import abc
from typing import Any, Callable, Dict, Optional

import numpy as np
import tensorflow as tf

from official.nlp.modeling.ops import decoding_module


def greedy(log_probs):
  """Returns the top ids and scores based on greedy decoding."""
  log_probs, ids = tf.math.top_k(log_probs, k=1)
  return log_probs, ids


def sample_logits_with_temperature(logits, temperature):
  """Applies a sampling temperature.

     Temperature skews the distribution towards high probability
     tokens and lowers the mass in tail distribution.

  Args:
    logits: Input logits for next token.
    temperature: Tensor for specifying the sampling temperature.

  Returns:
    Logits with applied temperature.
  """
  return logits / temperature


def sample_top_k(logits, top_k):
  """Chooses top_k logits and sets the others to negative infinity.

  Args:
    logits: Input logits for next token.
    top_k: Tensor to specify the top_k values.

  Returns:
    Logits with top_k filtering applied.
  """
  top_k = tf.clip_by_value(
      top_k, clip_value_min=1, clip_value_max=tf.shape(logits)[-1])
  top_k_logits = tf.math.top_k(logits, k=top_k)
  indices_to_remove = logits < tf.expand_dims(top_k_logits[0][..., -1], -1)
  top_k_logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
                                                np.NINF)
  return top_k_logits


def sample_top_p(logits, top_p):
  """Chooses most probable logits with cumulative probabilities upto top_p.

  Sets the remaining logits to negative infinity.

  Args:
    logits: Input logits for next token.
    top_p: Float tensor with a value >=0 and < 1.0

  Returns:
    Logits with top_p filtering applied.
  """
  sorted_indices = tf.argsort(logits, direction="DESCENDING")
  # Flatten logits as tf.gather on TPU needs axis to be compile time constant.
  logits_shape = decoding_module.shape_list(logits)
  range_for_gather = tf.expand_dims(tf.range(0, logits_shape[0]), axis=1)
  range_for_gather = tf.tile(range_for_gather * logits_shape[1],
                             [1, logits_shape[1]]) + sorted_indices
  flattened_logits = tf.reshape(logits, [-1])
  flattened_sorted_indices = tf.reshape(range_for_gather, [-1])
  sorted_logits = tf.reshape(
      tf.gather(flattened_logits, flattened_sorted_indices),
      [logits_shape[0], logits_shape[1]])
  cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

  # Remove tokens with cumulative probability above the threshold.
  sorted_indices_to_remove = cumulative_probs > top_p

  # Shift the indices to the right to keep the first token above threshold.
  sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
  sorted_indices_to_remove = tf.concat([
      tf.zeros_like(sorted_indices_to_remove[:, :1]),
      sorted_indices_to_remove[:, 1:]
  ], -1)

  # Scatter sorted indices to original indexes.
  indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove,
                                                      sorted_indices)
  top_p_logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
                                                np.NINF)
  return top_p_logits


def scatter_values_on_batch_indices(values, batch_indices):
  """Scatter `values` into a tensor using `batch_indices`.

  Args:
    values: tensor of shape [batch_size, vocab_size] containing the values to
      scatter
    batch_indices: tensor of shape [batch_size, vocab_size] containing the
      indices to insert (should be a permutation in range(0, n))

  Returns:
    Tensor of shape [batch_size, vocab_size] with values inserted at
    batch_indices
  """
  tensor_shape = decoding_module.shape_list(batch_indices)
  broad_casted_batch_dims = tf.reshape(
      tf.broadcast_to(
          tf.expand_dims(tf.range(tensor_shape[0]), axis=-1), tensor_shape),
      [1, -1])
  pair_indices = tf.transpose(
      tf.concat([broad_casted_batch_dims,
                 tf.reshape(batch_indices, [1, -1])], 0))
  return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), tensor_shape)


def set_tensor_by_indices_to_value(input_tensor, indices, value):
  """Where indices is True, set the value in input_tensor to value.

  Args:
    input_tensor: float (batch_size, dim)
    indices: bool (batch_size, dim)
    value: float scalar

  Returns:
    output_tensor: same shape as input_tensor.
  """
  value_tensor = tf.zeros_like(input_tensor) + value
  output_tensor = tf.where(indices, value_tensor, input_tensor)
  return output_tensor


class SamplingModule(decoding_module.DecodingModule, metaclass=abc.ABCMeta):
  """Implementation for sampling strategies (go/decoding-tf-nlp)."""

  def __init__(self,
               symbols_to_logits_fn,
               vocab_size: int,
               max_decode_length: int,
               eos_id: int,
               padded_decode: bool,
               length_normalization_fn: Optional[Callable[[int, tf.DType],
                                                          float]] = None,
               top_k=0,
               top_p=1.0,
               sample_temperature=0.0,
               enable_greedy: bool = True,
               dtype: tf.DType = tf.float32,
               decoding_name: Optional[str] = None,
               extra_cache_output: bool = False):
    """Initialize sampling module."""
    self.symbols_to_logits_fn = symbols_to_logits_fn
    self.length_normalization_fn = length_normalization_fn
    self.eos_id = eos_id
    self.padded_decode = padded_decode
    self.dtype = tf.as_dtype(dtype)
    self.vocab_size = tf.convert_to_tensor(vocab_size, dtype=tf.int32)
    self.max_decode_length = max_decode_length
    self.top_k = tf.convert_to_tensor(top_k, dtype=tf.int32)
    self.top_p = tf.convert_to_tensor(top_p, dtype=tf.float32)
    self.sample_temperature = tf.convert_to_tensor(
        sample_temperature, dtype=tf.float32)
    self.enable_greedy = enable_greedy
    self.decoding_name = decoding_name
    self.extra_cache_output = extra_cache_output
    super(SamplingModule, self).__init__(
        length_normalization_fn=length_normalization_fn,
        dtype=dtype,
        decoding_name=decoding_name,
        extra_cache_output=extra_cache_output)

  def _grow_alive_seq(self,
                      state: Dict[str, Any],
                      batch_size: int) -> decoding_module.InternalState:
    """Grow alive sequences by one token.

    This function will implement the decoding strategies like top_p, top_k
    and greedy for the choosing the next logit.

    Args:
      state: A dictionary with the current loop state.
      batch_size: The given batch size

    Returns:
      Tuple of
      (Top sequences [batch, curr_index + 1] or [batch, max_decode_length + 1],
       Scores of returned sequences [batch, 1],
       New ids [batch, 1],
       New alive cache)
    """
    i = state[decoding_module.StateKeys.CUR_INDEX]
    alive_seq = state[decoding_module.StateKeys.ALIVE_SEQ]
    alive_log_probs = state[decoding_module.StateKeys.ALIVE_LOG_PROBS]
    alive_cache = state[decoding_module.StateKeys.ALIVE_CACHE]

    if self.padded_decode:
      ids = tf.slice(alive_seq, [0, i], [batch_size, 1])
    else:
      ids = alive_seq

    new_logits, new_cache = self.symbols_to_logits_fn(ids, i, alive_cache)
    candidate_log_probs = decoding_module.log_prob_from_logits(
        new_logits)
    original_log_probs = candidate_log_probs + alive_log_probs

    topk_log_probs, topk_ids = None, None
    if self.enable_greedy:
      topk_log_probs, topk_ids = greedy(original_log_probs)
    else:
      temperature_fn = sample_logits_with_temperature
      sampled_logits = tf.cond(
          self.sample_temperature > 0.0,
          lambda: temperature_fn(new_logits, self.sample_temperature),
          lambda: new_logits)
      sampled_logits = tf.cond(
          self.top_k > 0,
          lambda: sample_top_k(sampled_logits, self.top_k),
          lambda: sampled_logits)
      sampled_logits = tf.cond(
          self.top_p < 1,
          lambda: sample_top_p(sampled_logits, self.top_p),
          lambda: sampled_logits)
      topk_ids = tf.random.categorical(
          sampled_logits, dtype=tf.int32, num_samples=1)
      topk_log_probs = tf.gather(
          original_log_probs, topk_ids, axis=1, batch_dims=1)
    if self.padded_decode:
      topk_seq = tf.transpose(alive_seq, perm=[1, 0])
      topk_seq = tf.tensor_scatter_nd_update(
          topk_seq, [[i + 1]], tf.expand_dims(tf.squeeze(topk_ids, -1), 0))
      topk_seq = tf.transpose(topk_seq, perm=[1, 0])
    else:
      topk_seq = tf.concat([alive_seq, topk_ids], axis=-1)
    return topk_seq, topk_log_probs, topk_ids, new_cache

  def _create_initial_state(
      self,
      initial_ids: tf.Tensor,
      initial_cache: Dict[str, tf.Tensor],
      batch_size: int,
      initial_log_probs: Optional[tf.Tensor] = None
  ) -> decoding_module.InitialState:
    """Return initial state dictionary and its shape invariants."""
    for key, value in initial_cache.items():
      for inner_value in tf.nest.flatten(value):
        if inner_value.dtype != self.dtype:
          raise TypeError(
              "initial_cache element for key '%s' has dtype %s that does not "
              "match sampling_module's dtype of %s. Value: %s" %
              (key, value.dtype.name, self.dtype.name, inner_value))

    # Current loop index (starts at 0)
    cur_index = tf.constant(0)

    # Alive sequence with shape [batch_size, 1]
    alive_seq = initial_ids
    alive_seq = tf.expand_dims(alive_seq, axis=-1)
    if self.padded_decode:
      alive_seq = tf.tile(alive_seq, [1, self.max_decode_length + 1])

    # Initial log probabilities with shape [batch_size, 1].
    if initial_log_probs is None:
      initial_log_probs = tf.constant([[0.]], dtype=self.dtype)
      alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])
    else:
      alive_log_probs = initial_log_probs

    alive_cache = initial_cache

    # Initialize tensor storing finished sequences [batch_size, 1, 1].
    finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)

    # Set scores of the initial finished seqs to negative infinity.
    finished_scores = tf.zeros([batch_size, 1], dtype=self.dtype)

    # Initialize finished flags with all False values.
    finished_flags = tf.zeros([batch_size, 1], tf.bool)

    # Create state dictionary and state shapes.
    state = {
        decoding_module.StateKeys.CUR_INDEX: cur_index,
        decoding_module.StateKeys.ALIVE_SEQ: alive_seq,
        decoding_module.StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
        decoding_module.StateKeys.ALIVE_CACHE: alive_cache,
        decoding_module.StateKeys.FINISHED_SEQ: finished_seq,
        decoding_module.StateKeys.FINISHED_SCORES: finished_scores,
        decoding_module.StateKeys.FINISHED_FLAGS: finished_flags
    }

    if self.padded_decode:
      state_shape_invariants = {
          decoding_module.StateKeys.CUR_INDEX:
              tf.TensorShape([]),
          decoding_module.StateKeys.ALIVE_SEQ:
              tf.TensorShape([batch_size, self.max_decode_length + 1]),
          decoding_module.StateKeys.ALIVE_LOG_PROBS:
              tf.TensorShape([batch_size, 1]),
          decoding_module.StateKeys.ALIVE_CACHE:
              tf.nest.map_structure(lambda state: state.get_shape(),
                                    alive_cache),
          decoding_module.StateKeys.FINISHED_SEQ:
              tf.TensorShape([batch_size, self.max_decode_length + 1]),
          decoding_module.StateKeys.FINISHED_SCORES:
              tf.TensorShape([batch_size, 1]),
          decoding_module.StateKeys.FINISHED_FLAGS:
              tf.TensorShape([batch_size, 1])
      }
    else:
      state_shape_invariants = {
          decoding_module.StateKeys.CUR_INDEX:
              tf.TensorShape([]),
          decoding_module.StateKeys.ALIVE_SEQ:
              tf.TensorShape([None, None]),
          decoding_module.StateKeys.ALIVE_LOG_PROBS:
              tf.TensorShape([None, 1]),
          decoding_module.StateKeys.ALIVE_CACHE:
              tf.nest.map_structure(decoding_module.get_shape_keep_last_dim,
                                    alive_cache),
          decoding_module.StateKeys.FINISHED_SEQ:
              tf.TensorShape([None, None]),
          decoding_module.StateKeys.FINISHED_SCORES:
              tf.TensorShape([None, 1]),
          decoding_module.StateKeys.FINISHED_FLAGS:
              tf.TensorShape([None, 1])
      }

    if self.extra_cache_output:
      state.update(
          {decoding_module.StateKeys.INITIAL_OUTPUT_CACHE: alive_cache})
      if self.padded_decode:
        state_shape_invariants.update({
            decoding_module.StateKeys.INITIAL_OUTPUT_CACHE:
                tf.nest.map_structure(lambda state: state.get_shape(),
                                      alive_cache)
        })
      else:
        state_shape_invariants.update({
            decoding_module.StateKeys.INITIAL_OUTPUT_CACHE:
                tf.nest.map_structure(decoding_module.get_shape_keep_last_dim,
                                      alive_cache),
        })

    return state, state_shape_invariants

  def _get_new_alive_state(self, new_seq: tf.Tensor, new_log_probs: tf.Tensor,
                           new_finished_flags: tf.Tensor,
                           new_cache: Dict[str, tf.Tensor]) -> Dict[str, Any]:
    """Gather the sequences that are still alive.

    This function resets the sequences in the alive_state that are finished.

    Args:
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor with shape [batch_size, cur_index + 1]
      new_log_probs: Log probabilities of new sequences float32 tensor with
        shape [batch_size, 1]
      new_finished_flags: A boolean Tensor indicates which sequences are live
        inside the beam.
      new_cache: Dict of cached values for each sequence.

    Returns:
      Dictionary with alive keys.
    """
    new_seq = tf.multiply(
        new_seq, tf.cast(tf.logical_not(new_finished_flags), new_seq.dtype))
    return {
        decoding_module.StateKeys.ALIVE_SEQ: new_seq,
        decoding_module.StateKeys.ALIVE_LOG_PROBS: new_log_probs,
        decoding_module.StateKeys.ALIVE_CACHE: new_cache
    }

  def _get_new_finished_state(self, state: Dict[str, Any], new_seq: tf.Tensor,
                              new_log_probs: tf.Tensor,
                              new_finished_flags: tf.Tensor,
                              batch_size: int) -> Dict[str, tf.Tensor]:
    """Combine new and old finished sequences.

    Args:
      state: A dictionary with the current loop state.
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor [batch, curr_index + 1] or [batch, max_decode_length + 1].
      new_log_probs: Log probabilities of new sequences float32 tensor with
        shape [batch, 1].
      new_finished_flags: A boolean Tensor indicates which sequences are live.
      batch_size: The given batch size.

    Returns:
      Dictionary with finished keys from StateKeys.
    """
    i = state[decoding_module.StateKeys.CUR_INDEX]
    finished_seq = state[decoding_module.StateKeys.FINISHED_SEQ]
    finished_scores = state[decoding_module.StateKeys.FINISHED_SCORES]
    finished_flags = state[decoding_module.StateKeys.FINISHED_FLAGS]

    if not self.padded_decode:
      finished_seq = tf.concat(
          [finished_seq, tf.zeros([batch_size, 1], tf.int32)], axis=-1)
    new_scores = new_log_probs
    if self.length_normalization_fn is not None:
      length_norm = self.length_normalization_fn(i + 1, self.dtype)
      new_scores = new_log_probs / length_norm
    new_seq = tf.multiply(
        new_seq, tf.cast(tf.logical_not(finished_flags), new_seq.dtype))
    new_scores = tf.multiply(
        new_scores, tf.cast(tf.logical_not(finished_flags), new_scores.dtype))

    finished_seq += tf.multiply(new_seq,
                                tf.cast(new_finished_flags, new_seq.dtype))
    finished_scores += tf.multiply(
        new_scores, tf.cast(new_finished_flags, new_scores.dtype))
    new_finished_flags = tf.logical_or(new_finished_flags, finished_flags)
    return {
        decoding_module.StateKeys.FINISHED_SEQ: finished_seq,
        decoding_module.StateKeys.FINISHED_SCORES: finished_scores,
        decoding_module.StateKeys.FINISHED_FLAGS: new_finished_flags
    }

  def _process_finished_state(
      self, finished_state: Dict[str, Any]) -> decoding_module.Output:
    """Process the alive/finished state to return final sequences and scores."""
    alive_seq = finished_state[decoding_module.StateKeys.ALIVE_SEQ]
    alive_log_probs = finished_state[decoding_module.StateKeys.ALIVE_LOG_PROBS]
    finished_seq = finished_state[decoding_module.StateKeys.FINISHED_SEQ]
    finished_scores = finished_state[decoding_module.StateKeys.FINISHED_SCORES]
    finished_flags = finished_state[decoding_module.StateKeys.FINISHED_FLAGS]
    finished_cond = tf.reduce_any(finished_flags, 1, name="finished_cond")
    if self.length_normalization_fn is not None:
      length_norm = self.length_normalization_fn(self.max_decode_length + 1,
                                                 self.dtype)
      alive_log_probs = alive_log_probs / length_norm
    seq_cond = decoding_module.expand_to_same_rank(finished_cond, finished_seq)
    score_cond = decoding_module.expand_to_same_rank(finished_cond,
                                                     finished_scores)
    finished_seq = tf.where(seq_cond, finished_seq, alive_seq)
    finished_scores = tf.where(score_cond, finished_scores, alive_log_probs)
    if self.extra_cache_output:
      return finished_seq, finished_scores, finished_state[
          decoding_module.StateKeys.INITIAL_OUTPUT_CACHE]
    return finished_seq, finished_scores

  def _continue_search(self, state) -> tf.Tensor:
    i = state[decoding_module.StateKeys.CUR_INDEX]
    # Have we reached max decoding length?
    not_at_end = tf.less(i, self.max_decode_length)
    # Have all sampled sequences reached an EOS?
    all_has_eos = tf.reduce_all(
        state[decoding_module.StateKeys.FINISHED_FLAGS],
        axis=None,
        name="search_finish_cond")
    return tf.logical_and(not_at_end, tf.logical_not(all_has_eos))

  def _finished_flags(self, topk_ids, state) -> tf.Tensor:
    new_finished_flags = tf.equal(topk_ids, self.eos_id)
    new_finished_flags = tf.logical_or(
        new_finished_flags, state[decoding_module.StateKeys.FINISHED_FLAGS])
    return new_finished_flags
