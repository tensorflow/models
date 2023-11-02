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

"""Library for rematerialization.

Incubates a version of tf.recompute_grad that is XLA compatible.
"""
import collections
import os
import threading
from typing import Deque, List, NamedTuple, Optional, Sequence

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras


class RecomputeContext(
    NamedTuple('RecomputeContext', [
        ('is_recomputing', bool),
        ('seed', tf.Tensor),
        ('children', Deque['RecomputeContext']),
    ])):
  """Context for recomputation.

  Attributes:
    is_recomputing: Whether we are in a recomputation phase.
    seed: Scalar integer tensor that should be used with stateless random ops
      for deterministic behavior and correct computation of the gradient.
    children: Nested `RecomputeContext` instances. Used internally by
      `recompute_grad` to track nested instances of `RecomputeContext`.
  """

  def __enter__(self):
    return _context_stack.push(self)

  def __exit__(self, exc_type, exc_value, traceback):
    _context_stack.pop(self)


# Simplified version of `_DefaultStack` in
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py.
class _ContextStack(threading.local):
  """A thread-local stack for providing implicit recompute contexts."""

  def __init__(self):
    super(_ContextStack, self).__init__()
    self._stack = []

  def top(self) -> Optional[RecomputeContext]:
    return self._stack[-1] if self._stack else None

  def push(self, context: RecomputeContext):
    self._stack.append(context)
    return context

  def pop(self, context: RecomputeContext):
    if self._stack[-1] is not context:
      raise AssertionError('Nesting violated for RecomputeContext.')
    self._stack.pop()


_context_stack = _ContextStack()


def get_recompute_context() -> Optional[RecomputeContext]:
  """Returns the current recomputing context if it exists."""
  return _context_stack.top()


# Adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/control_flow_util.py.
def _get_containing_xla_context(graph: tf.Graph) -> Optional[object]:
  """Returns the first ancestor `XLAControlFlowContext` in the `graph`."""
  ctxt = graph._get_control_flow_context()  # pylint: disable=protected-access
  while ctxt:
    if ctxt.IsXLAContext():
      return ctxt
    ctxt = ctxt.outer_context
  return None


def _in_xla_context(graph: Optional[tf.Graph] = None) -> bool:
  """Detects whether we are in an XLA context."""
  if '--tf_xla_auto_jit=2' in os.environ.get('TF_XLA_FLAGS', ''):
    return True
  graph = tf.compat.v1.get_default_graph() if graph is None else graph
  while True:
    if _get_containing_xla_context(graph) is not None:
      return True
    try:
      graph = graph.outer_graph
    except AttributeError:
      return False


def _force_data_dependency(
    first_compute: Sequence[tf.Tensor],
    then_compute: Sequence[tf.Tensor]) -> List[tf.Tensor]:
  """Force all of `then_compute` to depend on all of `first_compute`.

  Uses a dummy data dependency, which is useful when running on TPUs because
  XLA ignores control dependencies. Only supports float arguments.

  Args:
    first_compute: Sequence of `Tensor`s to be executed before `then_compute`.
    then_compute: Sequence of `Tensor`s to executed after `first_compute`.

  Returns:
    Sequence of `Tensor`s with same length of `then_compute`.

  Raises:
    ValueError: if ranks are unknown or types are not floating.
  """

  def _first_element(x):
    if x.shape.ndims is None:
      raise ValueError('Rank of Tensor %s must be known' % x)
    ndims = x.shape.ndims
    begin = tf.zeros(ndims, dtype=tf.int32)
    size = tf.ones(ndims, dtype=tf.int32)
    return tf.reshape(tf.slice(x, begin, size), [])

  first_compute_sum = tf.add_n(
      [_first_element(x) for x in first_compute if x is not None])
  dtype = first_compute_sum.dtype
  if not dtype.is_floating:
    raise ValueError('_force_data_dependency only supports floating dtypes.')
  zero = np.finfo(dtype.as_numpy_dtype).tiny * first_compute_sum
  return [
      x + tf.cast(zero, x.dtype) if x is not None else None
      for x in then_compute
  ]


def _make_seed_if_none(seed: Optional[tf.Tensor]) -> tf.Tensor:
  """Uses the global generator to make a seed if necessary."""
  if seed is not None:
    return seed
  generator = tf.random.experimental.get_global_generator()
  # The two seeds for stateless random ops don't have individual semantics and
  # are scrambled together, so providing one seed is fine. This makes it easier
  # for users to provide a local seed without worrying about integer overflow.
  # See `make_seeds` in
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/stateful_random_ops.py.
  try:
    return generator.uniform_full_int([], tf.int32, name='recompute_grad_seed')
  except (RuntimeError, TypeError, ValueError, tf.errors.NotFoundError) as e:
    # For a number of reasons, the above operation can fail like using multiple
    # graphs or toggling between eager and graph modes. Reset the generator.
    logging.warn('Resetting the generator. %s: %s', type(e), e)
    tf.random.experimental.set_global_generator(None)
    generator = tf.random.experimental.get_global_generator()
    return generator.uniform_full_int([], tf.int32, name='recompute_grad_seed')


def recompute_grad(f, seed=None):
  """An eager-compatible version of recompute_grad.

  For f(*args, **kwargs), this supports gradients with respect to args, or to
  gradients with respect to any variables residing in the kwarg 'variables'.
  Note that for keras layer and model objects, this is handled automatically.

  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.

  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
    seed: Optional seed for random ops. `seed` should an integer scalar
      `Tensor`. When compiling to XLA, `seed` must have dtype `tf.int32`. If
      `seed` is not provided one will be generated.

  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  """

  @tf.custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    # Detect when we're nested and in the backwards pass, so we don't generate
    # an additional seed.
    parent_context = get_recompute_context()
    if parent_context is not None and parent_context.is_recomputing:
      # Use the cached context in the recomputation phase.
      with parent_context.children.popleft()._replace(
          is_recomputing=True) as context:
        result = f(*args, **kwargs)
    else:
      with RecomputeContext(
          is_recomputing=False,
          seed=_make_seed_if_none(seed),
          children=collections.deque()) as context:
        result = f(*args, **kwargs)
        # In the forward pass, build up a tree of recomputation contexts.
        if parent_context is not None and not parent_context.is_recomputing:
          parent_context.children.append(context)

    def grad(*dresult, **grad_kwargs):
      """Gradient function calculation for inner function."""
      variables = grad_kwargs.pop('variables', None)
      if grad_kwargs:
        raise ValueError('Found unexpected kwargs for `grad`: ',
                         list(grad_kwargs.keys()))
      inputs, seed = list(args), context.seed
      if _in_xla_context():
        inputs = _force_data_dependency(
            tf.nest.flatten(dresult), inputs + [seed])
        seed = inputs.pop()
      with tf.GradientTape() as tape:
        tape.watch(inputs)
        if variables is not None:
          tape.watch(variables)
        with tf.control_dependencies(dresult):
          with context._replace(is_recomputing=True, seed=seed):
            result = f(*inputs, **kwargs)
      kw_vars = []
      if variables is not None:
        kw_vars = list(variables)
      grads = tape.gradient(
          result, list(inputs) + kw_vars, output_gradients=dresult)
      return grads[:len(inputs)], grads[len(inputs):]

    return result, grad

  return inner
