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
  """Creates a loop function driven by a Python `while` loop.

  Args:
    step_fn: A function taking a nested structure of `tf.data.Iterator` or
      `DistributedIterator`. There are no constraints on the return value of the
      function (except that it must be compatible with any `reduce_fn` provided
      to the returned `loop_fn`).

  Returns:
    A loop function taking required `iterator` and `num_steps` parameters, as
    well as optional `state` and `reduce_fn` parameters for accumulating state
    over multiple iterations of the loop. See the `loop_fn` definition below for
    additional details.
  """

  def loop_fn(iterator, num_steps, state=None, reduce_fn=None):
    """Makes `num_steps` calls to `step_fn(iterator)`.

    Additionally, state may be accumulated across iterations of the loop.
    Conceptually, state accumulation is handled roughly as follows:

        for _ in range(num_steps):
          step_outputs  = step_fn(iterator)
          state = reduce_fn(state, step_outputs)
        return state

    However, the implementation is slightly more complicated in order to support
    looping until the iterator is exhausted (when `num_steps == -1`) and to
    properly catch exceptions when running under async remote eager (as is the
    case in TPU training setups involving separate coordinator/worker machines).

    Args:
      iterator: A nested structure of `tf.data.Iterator` or
        `DistributedIterator`.
      num_steps: The number of steps in the loop. If `num_steps == -1`, will
        iterate until exausting the iterator.
      state: An optional initial state before running the loop.
      reduce_fn: A callable taking two inputs, `state` and `value`, where
        `state` is the previous output from `reduce_fn`, and `value` is the
        output from `step_fn`.

    Returns:
      The final state returned by `reduce_fn`, or `None` if `state` and
      `reduce_fn` are not provided.
    """
    try:
      step = 0
      # To make sure the OutOfRangeError exception can be handled well under
      # async remote eager, we need to wrap the loop body in `async_scope`.
      with tf.experimental.async_scope():
        while num_steps == -1 or step < num_steps:
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
  """Creates a loop function compatible with TF's AutoGraph loop conversion.

  Args:
    step_fn: A function taking a nested structure of `tf.data.Iterator` or
      `DistributedIterator`. Currently, any return values are ignored.

  Returns:
    A loop function taking required `iterator` and `num_steps` parameters. If
    called inside a `tf.function`, the loop will be converted by AutoGraph into
    a `tf.while_loop` construct. See the `loop_fn` definition below for
    additional details.
  """

  def loop_fn(iterator, num_steps):
    """Makes `num_steps` calls to `step_fn(iterator)`.

    Args:
      iterator: A nested structure of `tf.data.Iterator` or
        `DistributedIterator`.
      num_steps: The number of steps in the loop. Should be passed as a
        `tf.Tensor`. Iterating until iterator exhaustion is not supported.
    """
    if not isinstance(num_steps, tf.Tensor):
      raise ValueError(
          "`num_steps` should be a `tf.Tensor`. Passing a Python value can "
          "cause unnecessary retracing when wrapped by `tf.function`.")

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
