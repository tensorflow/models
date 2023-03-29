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

"""Base class for Decoding Strategies (beam_search, top_k, top_p and greedy)."""

import abc
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf

from tensorflow.python.framework import dtypes
from official.modeling import tf_utils

Output = Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
InternalState = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict]
InitialState = Tuple[Dict[str, Any], Dict[str, Any]]


class StateKeys:
  """Keys to dictionary storing the state of Decoding loop."""

  # Variable storing the loop index.
  CUR_INDEX = "CUR_INDEX"

  # Top sequences that are alive for each batch item. Alive sequences are ones
  # that have not generated an EOS token. Sequences that reach EOS are marked as
  # finished and moved to the FINISHED_SEQ tensor.
  # Has shape [batch_size, beam_size, CUR_INDEX + 1] for SequenceBeamSearch and
  # [batch_size, CUR_INDEX + 1] otherwise.
  ALIVE_SEQ = "ALIVE_SEQ"
  # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
  ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
  # Dictionary of cached values for each alive sequence. The cache stores
  # the encoder output, attention bias, and the decoder attention output from
  # the previous iteration.
  ALIVE_CACHE = "ALIVE_CACHE"

  # The initial model state/cache after model processing the initial token.
  # The cache will be filled if extra_cache_output is true.
  INITIAL_OUTPUT_CACHE = "INITIAL_OUTPUT_CACHE"

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


def log_prob_from_logits(logits):
  return logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)


def shape_list(tensor):
  """Return a list of the tensor's shape, and ensure no None values in list."""
  return tf_utils.get_shape_list(tensor)


def get_shape_keep_last_dim(tensor):
  shape_list_obj = shape_list(tensor)
  for i in range(len(shape_list_obj) - 1):
    shape_list_obj[i] = None

  if isinstance(shape_list_obj[-1], tf.Tensor):
    shape_list_obj[-1] = None
  return tf.TensorShape(shape_list_obj)


def expand_to_same_rank(tensor, target):
  """Expands a given tensor to target's rank to be broadcastable.

  Args:
    tensor: input tensor to tile. Shape: [b, d1, ..., da]
    target: target tensor. Shape: [b, d1, ..., da, ..., dn]

  Returns:
    Tiled tensor of shape [b, d1, ..., da, 1, ..., 1] with same rank of target

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


class DecodingModule(tf.Module, metaclass=abc.ABCMeta):
  """A base class for the API required for decoding (go/decoding-tf-nlp)."""

  def __init__(self,
               length_normalization_fn: Callable[[int, tf.DType], float],
               dtype: tf.DType = tf.float32,
               decoding_name: Optional[str] = None,
               extra_cache_output: bool = False):
    """Initialize the Decoding Module.

    Args:
      length_normalization_fn: Closure for returning length normalization
      parameter. Function accepts input as length, dtype and returns float.
      dtype: A tensorflow data type used for score computation. The default is
        tf.float32.
      decoding_name: an optional name for the decoding loop tensors.
      extra_cache_output: If true, the first cache will be in the states.
    """
    self.length_normalization_fn = length_normalization_fn
    self.dtype = tf.as_dtype(dtype)
    self.decoding_name = decoding_name

  def generate(self,
               initial_ids: tf.Tensor,
               initial_cache: Dict[str, tf.Tensor],
               initial_log_probs: Optional[tf.Tensor] = None) -> Output:
    """Implements the decoding strategy (beam_search or sampling).

    Args:
      initial_ids: initial ids to pass into the symbols_to_logits_fn. int tensor
        with shape [batch_size, 1]
      initial_cache: dictionary for caching model outputs from previous step.
      initial_log_probs: Optionally initial log probs if there is a prefix
        sequence we want to start to decode from.

    Returns:
      Tuple of tensors representing
        finished_sequence: shape [batch, max_seq_length]
        finished_scores: [batch]
        first_cache: The cache after init token
    """
    batch_size = (
        initial_ids.shape.as_list()[0]
        if self.padded_decode else tf.shape(initial_ids)[0])

    state, state_shapes = self._create_initial_state(initial_ids, initial_cache,
                                                     batch_size,
                                                     initial_log_probs)

    def _generate_step(state):
      topk_seq, topk_log_probs, topk_ids, new_cache = self._grow_alive_seq(
          state, batch_size)
      new_finished_flags = self._finished_flags(topk_ids, state)
      alive_state = self._get_new_alive_state(topk_seq,
                                              topk_log_probs,
                                              new_finished_flags,
                                              new_cache)
      finished_state = self._get_new_finished_state(state,
                                                    topk_seq,
                                                    topk_log_probs,
                                                    new_finished_flags,
                                                    batch_size)
      new_state = {
          StateKeys.CUR_INDEX: state[StateKeys.CUR_INDEX] + 1
      }
      new_state.update(alive_state)
      new_state.update(finished_state)
      if self.extra_cache_output:
        i = state[StateKeys.CUR_INDEX]
        old_cache = state[StateKeys.INITIAL_OUTPUT_CACHE]

        def update_with_cache(new_state, cache):
          """Updates new_state with cache."""
          new_state.update({StateKeys.INITIAL_OUTPUT_CACHE: cache})

        tf.cond(
            tf.equal(i, 0), lambda: update_with_cache(new_state, new_cache),
            lambda: update_with_cache(new_state, old_cache))
      return [new_state]

    finished_state = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
            self._continue_search,
            _generate_step,
            loop_vars=[state],
            shape_invariants=[state_shapes],
            parallel_iterations=1,
            name=self.decoding_name))
    final_state = self._process_finished_state(finished_state[0])
    return final_state

  @abc.abstractmethod
  def _create_initial_state(
      self,
      initial_ids: tf.Tensor,
      initial_cache: Dict[str, tf.Tensor],
      batch_size: int,
      initial_log_probs: Optional[tf.Tensor] = None) -> InitialState:
    """Return initial state dictionary and its shape invariants."""
    pass

  @abc.abstractmethod
  def _grow_alive_seq(self,
                      state: Dict[str, Any],
                      batch_size: int) -> InternalState:
    """Grow alive sequences by one token.

    Args:
      state: A dictionary with the current loop state.
      batch_size: The given batch size

    Returns:
      Tuple of
      (Top sequences,
       Scores of returned sequences,
       New ids,
       New alive cache)
    """
    pass

  @abc.abstractmethod
  def _get_new_alive_state(
      self,
      new_seq: tf.Tensor,
      new_log_probs: tf.Tensor,
      new_finished_flags: tf.Tensor,
      new_cache: Dict[str, tf.Tensor]) -> Dict[str, Any]:
    """Gather the sequences that are still alive.

    Args:
      new_seq: New sequences generated by growing the current alive sequences
        int32 tensor with shape
      new_log_probs: Log probabilities of new sequences float32 tensor with
        shape
      new_finished_flags: A boolean Tensor indicates which sequences are live.
      new_cache: Dict of cached values for each sequence.

    Returns:
      Dictionary with alive keys from StateKeys.
    """
    pass

  @abc.abstractmethod
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
        int32 tensor.
      new_log_probs: Log probabilities of new sequences float32 tensor with
        shape.
      new_finished_flags: A boolean Tensor indicates which sequences are live.
      batch_size: The given batch size.

    Returns:
      Dictionary with finished keys from StateKeys.
    """
    pass

  @abc.abstractmethod
  def _process_finished_state(self, finished_state: Dict[str, Any]) -> Output:
    """Process the alive/finished state to return final sequences and scores."""
    pass

  @abc.abstractmethod
  def _continue_search(self, state: Dict[str, Any]) -> tf.Tensor:
    """Returns a bool tensor if the decoding loop should continue."""
    pass

  @abc.abstractmethod
  def _finished_flags(self,
                      topk_ids: tf.Tensor,
                      state: Dict[str, Any]) -> tf.Tensor:
    """Calculate the finished flags."""
    pass

  def inf(self):
    """Returns a value close to infinity, but is still finite in `dtype`.

    This is useful to get a very large value that is still zero when multiplied
    by zero. The floating-point "Inf" value is NaN when multiplied by zero.

    Returns:
      A very large value.
    """
    if self.dtype == dtypes.float32 or self.dtype == dtypes.bfloat16:
      return 1e7
    elif self.dtype == dtypes.float16:
      return dtypes.float16.max
    else:
      raise AssertionError("Invalid dtype: %s" % self.dtype)
