# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Context functions.

Given the current contexts, timer and context sampler, returns new contexts
  after an environment step. This can be used to define a high-level policy
  that controls contexts as its actions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin.tf
import utils as uvf_utils


@gin.configurable
def periodic_context_fn(contexts, timer, sampler_fn, period=1):
  """Periodically samples contexts.

  Args:
    contexts: a list of [num_context_dims] tensor variables representing
      current contexts.
    timer: a scalar integer tensor variable holding the current time step.
    sampler_fn: a sampler function that samples a list of [num_context_dims]
      tensors.
    period: (integer) period of update.
  Returns:
    a list of [num_context_dims] tensors.
  """
  contexts = list(contexts[:])  # create copy
  return tf.cond(tf.mod(timer, period) == 0, sampler_fn, lambda: contexts)


@gin.configurable
def timer_context_fn(contexts,
                     timer,
                     sampler_fn,
                     period=1,
                     timer_index=-1,
                     debug=False):
  """Samples contexts based on timer in contexts.

  Args:
    contexts: a list of [num_context_dims] tensor variables representing
      current contexts.
    timer: a scalar integer tensor variable holding the current time step.
    sampler_fn: a sampler function that samples a list of [num_context_dims]
      tensors.
    period: (integer) period of update; actual period = `period` + 1.
    timer_index: (integer) Index of context list that present timer.
    debug: (boolean) Print debug messages.
  Returns:
    a list of [num_context_dims] tensors.
  """
  contexts = list(contexts[:])  # create copy
  cond = tf.equal(contexts[timer_index][0], 0)
  def reset():
    """Sample context and reset the timer."""
    new_contexts = sampler_fn()
    new_contexts[timer_index] = tf.zeros_like(
        contexts[timer_index]) + period
    return new_contexts
  def update():
    """Decrement the timer."""
    contexts[timer_index] -= 1
    return contexts
  values = tf.cond(cond, reset, update)
  if debug:
    values[0] = uvf_utils.tf_print(
        values[0],
        values + [timer],
        'timer_context_fn',
        first_n=200,
        name='timer_context_fn:contexts')
  return values


@gin.configurable
def relative_context_transition_fn(
    contexts, timer, sampler_fn,
    k=2, state=None, next_state=None,
    **kwargs):
  """Contexts updated to be relative to next state.
  """
  contexts = list(contexts[:])  # create copy
  assert len(contexts) == 1
  new_contexts = [
      tf.concat(
          [contexts[0][:k] + state[:k] - next_state[:k],
           contexts[0][k:]], -1)]
  return new_contexts


@gin.configurable
def relative_context_multi_transition_fn(
    contexts, timer, sampler_fn,
    k=2, states=None,
    **kwargs):
  """Given contexts at first state and sequence of states, derives sequence of all contexts.
  """
  contexts = list(contexts[:])  # create copy
  assert len(contexts) == 1
  contexts = [
      tf.concat(
          [tf.expand_dims(contexts[0][:, :k] + states[:, 0, :k], 1) - states[:, :, :k],
           contexts[0][:, None, k:] * tf.ones_like(states[:, :, :1])], -1)]
  return contexts
