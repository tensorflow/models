# Copyright 2020 The Orbit Authors. All Rights Reserved.
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
"""Utilities for creating loop functions."""

from orbit.utils import tpu_summaries

import tensorflow as tf


def create_loop_fn(step_fn):
  """Creates a multiple steps function driven by the python while loop.

  Args:
    step_fn: A function which takes `iterator` as input.

  Returns:
    A callable defined as the `loop_fn` defination below.
  """

  def loop_fn(iterator, num_steps, state=None, reduce_fn=None):
    """A loop function with multiple steps.

    Args:
      iterator: A nested structure of tf.data `Iterator` or
        `DistributedIterator`.
      num_steps: The number of steps in the loop. If `num_steps==-1`, will
        iterate until exausting the iterator.
      state: An optional initial state before running the loop.
      reduce_fn: a callable defined as `def reduce_fn(state, value)`, where
        `value` is the outputs from `step_fn`.

    Returns:
      The updated state.
    """
    try:
      step = 0
      # To make sure the OutOfRangeError exception can be handled well with
      # async remote eager, we need to wrap the loop body in a `async_scope`.
      with tf.experimental.async_scope():
        while (num_steps == -1 or step < num_steps):
          outputs = step_fn(iterator)
          if reduce_fn is not None:
            state = reduce_fn(state, outputs)
          step += 1
        return state
    except (StopIteration, tf.errors.OutOfRangeError):
      tf.experimental.async_clear_error()
      return state

  return loop_fn


def create_tf_while_loop_fn(step_fn):
  """Create a multiple steps function driven by tf.while_loop on the host.

  Args:
    step_fn: A function which takes `iterator` as input.

  Returns:
    A callable defined as the `loop_fn` defination below.
  """

  def loop_fn(iterator, num_steps):
    """A loop function with multiple steps.

    Args:
      iterator: A nested structure of tf.data `Iterator` or
        `DistributedIterator`.
      num_steps: The number of steps in the loop. Must be a tf.Tensor.
    """
    if not isinstance(num_steps, tf.Tensor):
      raise ValueError("`num_steps` should be an `tf.Tensor`. Python object "
                       "may cause retracing.")

    for _ in tf.range(num_steps):
      step_fn(iterator)

  return loop_fn


class LoopFnWithSummaries(tpu_summaries.OptionalSummariesFunction):
  """Implements a two-program approach for optimizing summaries on TPU.

  This version works with the result of `create_tf_while_loop_fn`.
  """

  def __call__(self, iterator, num_steps):
    if tf.summary.should_record_summaries():
      output = self.with_summaries(iterator, tf.constant(1))
      num_steps -= 1
    if num_steps >= 1:
      output = self.without_summaries(iterator, num_steps)
    return output
