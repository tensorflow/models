# Copyright 2018 Google, Inc. All Rights Reserved.
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

"""Utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import tensorflow as tf
import sonnet as snt
import itertools
import functools

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.ops import variable_scope as variable_scope_ops
from sonnet.python.modules import util as snt_util

from tensorflow.python.util import nest


def eqzip(*args):
  """Zip but raises error if lengths don't match.

  Args:
    *args: list of lists or tuples
  Returns:
    list: the result of zip
  Raises:
    ValueError: when the lengths don't match
  """

  sizes = [len(x) for x in args]
  if not all([sizes[0] == x for x in sizes]):
    raise ValueError("Lists are of different sizes. \n %s"%str(sizes))
  return zip(*args)


@contextlib.contextmanager
def assert_no_new_variables():
  """Ensure that no tf.Variables are constructed inside the context.

  Yields:
    None
  Raises:
    ValueError: if there is a variable created.
  """
  num_vars = len(tf.global_variables())
  old_variables = tf.global_variables()
  yield
  if len(tf.global_variables()) != num_vars:
    new_vars = set(tf.global_variables()) - set(old_variables)
    tf.logging.error("NEW VARIABLES CREATED")
    tf.logging.error(10*"=")
    for v in new_vars:
      tf.logging.error(v)

    raise ValueError("Variables created inside an "
                     "assert_no_new_variables context")
  if old_variables != tf.global_variables():
    raise ValueError("Variables somehow changed inside an "
                     "assert_no_new_variables context."
                     "This means something modified the tf.global_variables()")


def get_variables_in_modules(module_list):
  var_list = []
  for m in module_list:
    var_list.extend(snt.get_variables_in_module(m))
  return var_list


def state_barrier_context(state):
  """Return a context manager that prevents interior ops from running
  unless the whole state has been computed.

  This is to prevent assign race conditions.
  """
  tensors = [x for x in nest.flatten(state) if type(x) == tf.Tensor]
  tarray = [x.flow for x in nest.flatten(state) if hasattr(x, "flow")]
  return tf.control_dependencies(tensors + tarray)


def _identity_fn(tf_entity):
  if hasattr(tf_entity, "identity"):
    return tf_entity.identity()
  else:
    return tf.identity(tf_entity)


def state_barrier_result(state):
  """Return the same state, but with a control dependency to prevent it from
  being partially computed
  """
  with state_barrier_context(state):
    return nest.map_structure(_identity_fn, state)


def train_iterator(num_iterations):
  """Iterator that returns an index of the current step.
  This iterator runs forever if num_iterations is None
  otherwise it runs for some fixed amount of steps.
  """
  if num_iterations is None:
    return itertools.count()
  else:
    return xrange(num_iterations)


def print_op(op, msg):
  """Print a string and return an op wrapped in a control dependency to make
  sure it ran."""
  print_op = tf.Print(tf.constant(0), [tf.constant(0)], msg)
  return tf.group(op, print_op)


class MultiQueueRunner(tf.train.QueueRunner):
  """A QueueRunner with multiple queues """
  def __init__(self, queues, enqueue_ops):
    close_op = tf.group(* [q.close() for q in queues])
    cancel_op = tf.group(
        * [q.close(cancel_pending_enqueues=True) for q in queues])
    queue_closed_exception_types = (errors.OutOfRangeError,)

    enqueue_op = tf.group(*enqueue_ops, name="multi_enqueue")

    super(MultiQueueRunner, self).__init__(
        queues[0],
        enqueue_ops=[enqueue_op],
        close_op=close_op,
        cancel_op=cancel_op,
        queue_closed_exception_types=queue_closed_exception_types)


# This function is not elegant, but I tried so many other ways to get this to
# work and this is the only one that ended up not incuring significant overhead
# or obscure tensorflow bugs.
def sample_n_per_class(dataset, samples_per_class):
  """Create a new callable / dataset object that returns batches of each with
  samples_per_class per label.

  Args:
    dataset: fn
    samples_per_class: int
  Returns:
    function, [] -> batch where batch is the same type as the return of
    dataset().
  """

  with tf.control_dependencies(None), tf.name_scope(None):
    with tf.name_scope("queue_runner/sample_n_per_class"):
      batch = dataset()
      num_classes = batch.label_onehot.shape.as_list()[1]
      batch_size = num_classes * samples_per_class

      flatten = nest.flatten(batch)
      queues = []
      enqueue_ops = []
      capacity = samples_per_class * 20
      for i in xrange(num_classes):
        queue = tf.FIFOQueue(
            capacity=capacity,
            shapes=[f.shape.as_list()[1:] for f in flatten],
            dtypes=[f.dtype for f in flatten])
        queues.append(queue)

        idx = tf.where(tf.equal(batch.label, i))
        sub_batch = []
        to_enqueue = []
        for elem in batch:
          new_e = tf.gather(elem, idx)
          new_e = tf.squeeze(new_e, 1)
          to_enqueue.append(new_e)

        remaining = (capacity - queue.size())
        to_add = tf.minimum(tf.shape(idx)[0], remaining)

        def _enqueue():
          return queue.enqueue_many([t[:to_add] for t in to_enqueue])

        enqueue_op = tf.cond(
            tf.equal(to_add, 0), tf.no_op, _enqueue)
        enqueue_ops.append(enqueue_op)

      # This has caused many deadlocks / issues. This is some logging to at least
      # shed light to what is going on.
      print_lam = lambda: tf.Print(tf.constant(0.0), [q.size() for q in queues], "MultiQueueRunner queues status. Has capacity %d"%capacity)
      some_percent_of_time = tf.less(tf.random_uniform([]), 0.0005)
      maybe_print = tf.cond(some_percent_of_time, print_lam, lambda: tf.constant(0.0))
      with tf.control_dependencies([maybe_print]):
        enqueue_ops = [tf.group(e) for e in enqueue_ops]
      qr = MultiQueueRunner(queues=queues, enqueue_ops=enqueue_ops)
      tf.train.add_queue_runner(qr)

  def dequeue_batch():
    with tf.name_scope("sample_n_per_batch/dequeue/"):
      entries = []
      for q in queues:
        entries.append(q.dequeue_many(samples_per_class))

      flat_batch = [tf.concat(x, 0) for x in zip(*entries)]
      idx = tf.random_shuffle(tf.range(batch_size))
      flat_batch = [tf.gather(f, idx, axis=0) for f in flat_batch]
      return nest.pack_sequence_as(batch, flat_batch)

  return dequeue_batch

def structure_map_multi(func, values):
  all_values = [nest.flatten(v) for v in values]
  rets = []
  for pair in zip(*all_values):
    rets.append(func(pair))
  return nest.pack_sequence_as(values[0], rets)

def structure_map_split(func, value):
  vv = nest.flatten(value)
  rets = []
  for v in vv:
    rets.append(func(v))
  return [nest.pack_sequence_as(value, r) for r in zip(*rets)]

def assign_variables(targets, values):
  return tf.group(*[t.assign(v) for t,v in eqzip(targets, values)],
                  name="assign_variables")


def create_variables_in_class_scope(method):
  """Force the variables constructed in this class to live in the sonnet module.
  Wraps a method on a sonnet module.

  For example the following will create two different variables.
  ```
  class Mod(snt.AbstractModule):
    @create_variables_in_class_scope
    def dynamic_thing(self, input, name):
      return snt.Linear(name)(input)
  mod.dynamic_thing(x, name="module_nameA")
  mod.dynamic_thing(x, name="module_nameB")
  # reuse
  mod.dynamic_thing(y, name="module_nameA")
  ```
  """
  @functools.wraps(method)
  def wrapper(obj, *args, **kwargs):
    def default_context_manager(reuse=None):
      variable_scope = obj.variable_scope
      return tf.variable_scope(variable_scope, reuse=reuse)

    variable_scope_context_manager = getattr(obj, "_enter_variable_scope",
                                             default_context_manager)
    graph = tf.get_default_graph()

    # Temporarily enter the variable scope to capture it
    with variable_scope_context_manager() as tmp_variable_scope:
      variable_scope = tmp_variable_scope

    with variable_scope_ops._pure_variable_scope(
        variable_scope, reuse=tf.AUTO_REUSE) as pure_variable_scope:

      name_scope = variable_scope.original_name_scope
      if name_scope[-1] != "/":
        name_scope += "/"

      with tf.name_scope(name_scope):
        sub_scope = snt_util.to_snake_case(method.__name__)
        with tf.name_scope(sub_scope) as scope:
          out_ops = method(obj, *args, **kwargs)
          return out_ops

  return wrapper

