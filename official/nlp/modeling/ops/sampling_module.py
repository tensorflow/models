# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Sampling module for top_k, top_p and greedy decoding."""

import abc
from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf

from official.nlp.modeling.ops import decoding_module


class SamplingModule(decoding_module.DecodingModule, metaclass=abc.ABCMeta):
  """Implementation for sampling stratgies (go/decoding-tf-nlp)."""

  def __init__(self,
               symbols_to_logits_fn,
               length_normalization_fn: Callable[[int, tf.DType], float],
               vocab_size: int,
               max_decode_length: int,
               eos_id: int,
               padded_decode: bool,
               top_k: tf.Tensor = None,
               sample_temperature: tf.Tensor = None,
               dtype: tf.DType = tf.float32):
    """Initialize sampling module."""
    self.symbols_to_logits_fn = symbols_to_logits_fn
    self.vocab_size = vocab_size
    self.length_normalization_fn = length_normalization_fn
    self.max_decode_length = max_decode_length
    self.eos_id = eos_id
    self.padded_decode = padded_decode
    self.dtype = tf.as_dtype(dtype)
    self.top_k = top_k
    self.sample_temperature = sample_temperature
    super(SamplingModule, self).__init__(
        length_normalization_fn=length_normalization_fn, dtype=dtype)

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
    candidate_log_probs = decoding_module.DecodingModule._log_prob_from_logits(
        new_logits)
    original_log_probs = candidate_log_probs + alive_log_probs
    probs = original_log_probs

    topk_log_probs, topk_ids = None, None
    if not self.do_sample:
      topk_log_probs, topk_ids = self._greedy(probs)
    else:
      temperature_fn = self.sample_logits_with_temperature
      probs = tf.cond(self.sample_temperature > 0.0,
                      lambda: temperature_fn(probs, self.sample_temperature),
                      lambda: probs)
      probs = tf.cond(self.top_k is not None and self.top_k > 1,
                      lambda: self._sample_top_k(probs, self.top_k),
                      lambda: probs)
      topk_ids = tf.random.categorical(probs, dtype=tf.int32, num_samples=1)
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

  def _create_initial_state(self,
                            initial_ids: tf.Tensor,
                            initial_cache: Dict[str, tf.Tensor],
                            batch_size: int) -> decoding_module.InitialState:
    """Return initial state dictionary and its shape invariants."""
    for key, value in initial_cache.items():
      for inner_value in tf.nest.flatten(value):
        if inner_value.dtype != self.dtype:
          raise TypeError(
              "initial_cache element for key '%s' has dtype %s that does not "
              "match SequenceBeamSearch's dtype of %s. Value: %s" %
              (key, value.dtype.name, self.dtype.name, inner_value))

    # Current loop index (starts at 0)
    cur_index = tf.constant(0)

    # Alive sequence with shape [batch_size, 1]
    alive_seq = initial_ids
    alive_seq = tf.expand_dims(alive_seq, axis=-1)
    if self.padded_decode:
      alive_seq = tf.tile(alive_seq, [1, self.max_decode_length + 1])

    # Initial log probabilities with shape [batch_size, 1].
    initial_log_probs = tf.constant([[0.]], dtype=self.dtype)
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

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
              tf.TensorShape(
                  [batch_size, self.max_decode_length + 1]),
          decoding_module.StateKeys.ALIVE_LOG_PROBS:
              tf.TensorShape([batch_size, 1]),
          decoding_module.StateKeys.ALIVE_CACHE:
              tf.nest.map_structure(lambda state: state.get_shape(),
                                    alive_cache),
          decoding_module.StateKeys.FINISHED_SEQ:
              tf.TensorShape(
                  [batch_size, self.max_decode_length + 1]),
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
              tf.nest.map_structure(
                  decoding_module.DecodingModule._get_shape_keep_last_dim,
                  alive_cache),
          decoding_module.StateKeys.FINISHED_SEQ:
              tf.TensorShape([None, None]),
          decoding_module.StateKeys.FINISHED_SCORES:
              tf.TensorShape([None, 1]),
          decoding_module.StateKeys.FINISHED_FLAGS:
              tf.TensorShape([None, 1])
      }

    return state, state_shape_invariants

  def _get_new_alive_state(
      self,
      new_seq: tf.Tensor,
      new_log_probs: tf.Tensor,
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

  def _get_new_finished_state(self,
                              state: Dict[str, Any],
                              new_seq: tf.Tensor,
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
    seq_cond = decoding_module.DecodingModule._expand_to_same_rank(
        finished_cond, finished_seq)
    score_cond = decoding_module.DecodingModule._expand_to_same_rank(
        finished_cond, finished_scores)
    finished_seq = tf.where(seq_cond, finished_seq, alive_seq, finished_scores)
    finished_scores = tf.where(score_cond, finished_scores, alive_log_probs)
    return finished_seq, finished_scores

  def _continue_search(self, state) -> tf.Tensor:
    i = state[decoding_module.StateKeys.CUR_INDEX]
    return tf.less(i, self.max_decode_length)

  def _finished_flags(self, topk_ids, state) -> tf.Tensor:
    new_finished_flags = tf.equal(topk_ids, self.eos_id)
    new_finished_flags = tf.logical_or(
        new_finished_flags, state[decoding_module.StateKeys.FINISHED_FLAGS])
    return new_finished_flags

  @property
  def do_sample(self) -> bool:
    """Returns True if top_p or top_k is enabled."""
    # TODO(poorvap) : Add the check for top_p.
    if self.top_k is not None:
      return True
    return False

  @staticmethod
  def _greedy(log_probs):
    """Returns the top ids and scores based on greedy decoding."""
    log_probs, ids = tf.nn.top_k(log_probs, k=1)
    return log_probs, ids

  @staticmethod
  def sample_logits_with_temperature(logits, temperature):
    """Applies a sampling temperature.

       Temperature of [0, 1) skews the distribution towards high probability
       tokens and lowers the mass in tail distribution.

    Args:
      logits: Input logits for next token.
      temperature: Tensor for specifying the sampling temperature.

    Returns:
      Logits with applied temperature.
    """
    return logits / temperature

  @staticmethod
  def _sample_top_k(logits, top_k):
    """Chooses top_k logits and sets the others to negative infinity.

    Args:
      logits: Input logits for next token.
      top_k: Tensor to specify the top_k values.

    Returns:
      Logits with top_k filtering apploed.
    """
    top_k_logits = tf.math.top_k(logits, k=top_k)
    indices_to_remove = logits < top_k_logits[0][..., -1, None]
    top_k_logits = SamplingModule._set_tensor_by_indices_to_value(
        logits, indices_to_remove, np.NINF)
    return top_k_logits

  @staticmethod
  def _set_tensor_by_indices_to_value(input_tensor, indices, value):
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







